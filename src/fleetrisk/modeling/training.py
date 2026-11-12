import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import mlflow
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import Engine, text

from fleetrisk.common.io import atomic_json, checksum, dumps
from fleetrisk.config import Settings
from fleetrisk.data.contracts import get_contract
from fleetrisk.data.db import one
from fleetrisk.evaluation.metrics import bootstrap_intervals, evaluate, subgroup_metrics
from fleetrisk.features.registry import (
    FEATURE_VERSION,
    feature_frame,
    feature_schema,
    preprocessing,
)

SEED = 42
TARGETS = {
    "ltfs": (
        "loan_default: source-defined first-EMI default",
        "first EMI due date; exact delinquency tolerance unverified",
    ),
    "home_credit_application": (
        "TARGET: source-defined payment difficulties",
        "source-defined first Y installments; X and Y undisclosed",
    ),
}


def partition(y: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
    if len(y) < 100 or min(np.bincount(y, minlength=2)) < 20:
        raise ValueError("At least 100 rows and 20 observations of each class are required")
    indices = np.arange(len(y))
    remaining, test = train_test_split(indices, test_size=0.15, stratify=y, random_state=SEED)
    remaining, calibration = train_test_split(
        remaining, test_size=0.10 / 0.85, stratify=y[remaining], random_state=SEED
    )
    train, validation = train_test_split(
        remaining, test_size=0.15 / 0.75, stratify=y[remaining], random_state=SEED
    )
    return {"train": train, "validation": validation, "calibration": calibration, "test": test}


def revision() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def train(
    engine: Engine,
    settings: Settings,
    snapshot_id: str,
    *,
    quick: bool = False,
    bootstrap_replicates: int = 500,
) -> dict[str, Any]:
    with engine.connect() as connection:
        snapshot = one(connection, "SELECT * FROM dataset_snapshots WHERE id=:id", id=snapshot_id)
    if snapshot is None:
        raise ValueError("Unknown dataset snapshot")
    if snapshot["provenance"] == "synthetic" and not settings.allow_synthetic:
        raise ValueError("Synthetic training requires explicit fixture mode")
    if quick and snapshot["provenance"] != "synthetic":
        raise ValueError(
            "Quick training is fixture-only; empirical runs use the configured full budget"
        )
    dataset = snapshot["source_id"]
    if dataset not in TARGETS:
        raise ValueError("This table is not a supported supervised population")
    path = Path(snapshot["parquet_path"])
    if checksum(path) != snapshot["parquet_sha256"]:
        raise ValueError("Snapshot integrity check failed")
    if snapshot["row_count"] > 2_000_000:
        raise ValueError(
            "Training memory budget exceeded; explicitly redesign computation, never silently sample"
        )
    contract = get_contract(dataset)
    assert contract.target is not None
    names = (
        list(contract.key) + list(contract.numeric) + list(contract.categorical) + [contract.target]
    )
    frame = pl.read_parquet(path, columns=names).sort(list(contract.key)).to_pandas()
    X = feature_frame(frame, dataset)
    y = frame[contract.target].to_numpy(dtype=int)
    split = partition(y)
    keys = [dumps([row[k] for k in contract.key]) for row in frame.to_dict(orient="records")]
    experiment_id, model_id = str(uuid4()), str(uuid4())
    directory = settings.artifact_root.resolve() / model_id
    directory.mkdir(parents=True, exist_ok=False)
    split_manifest = {
        "seed": SEED,
        "strategy": "stratified_application_random; not chronological",
        "borrower_grouping": "not established; source application keys only",
        "partitions": {name: [keys[i] for i in ids] for name, ids in split.items()},
    }
    atomic_json(directory / "splits.json", split_manifest)
    schema = feature_schema(dataset)
    atomic_json(directory / "feature_schema.json", schema)
    feature_id = hashlib.sha256(f"{snapshot_id}:{FEATURE_VERSION}".encode()).hexdigest()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    definitions: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "prior": (DummyClassifier(strategy="prior"), {}),
        "logistic": (
            LogisticRegression(max_iter=1000, random_state=SEED),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=150,
                n_jobs=4,
                verbosity=-1,
                random_state=SEED,
                deterministic=True,
                force_col_wise=True,
            ),
            {
                "model__num_leaves": [7, 15, 31],
                "model__min_child_samples": [20, 50],
                "model__reg_lambda": [0.0, 1.0],
            },
        ),
    }
    train_ids, cal_ids, val_ids, test_ids = (
        split[name] for name in ("train", "calibration", "validation", "test")
    )
    candidates: dict[str, Any] = {}
    raw_models: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    for name, (model, grid) in definitions.items():
        pipe = Pipeline([("preprocess", preprocessing(dataset)), ("model", model)])
        if grid:
            search_grid = {k: v[:1] for k, v in grid.items()} if quick else grid
            search = GridSearchCV(
                pipe,
                search_grid,
                cv=cv,
                scoring="neg_log_loss",
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(X.iloc[train_ids], y[train_ids])
            pipe, parameters[name] = search.best_estimator_, search.best_params_
        else:
            pipe.fit(X.iloc[train_ids], y[train_ids])
            parameters[name] = {"strategy": "prior"}
        raw_models[name] = pipe
        candidates[f"{name}:none"] = pipe
        if name != "prior":
            for method in ("sigmoid", "isotonic"):
                if method == "isotonic" and (
                    len(cal_ids) < 1000 or min(np.bincount(y[cal_ids], minlength=2)) < 100
                ):
                    continue
                calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method=method)
                calibrated.fit(X.iloc[cal_ids], y[cal_ids])
                candidates[f"{name}:{method}"] = calibrated
    validation = {
        name: evaluate(y[val_ids], pipe.predict_proba(X.iloc[val_ids])[:, 1])
        for name, pipe in candidates.items()
    }
    selected = min(
        candidates,
        key=lambda name: (
            validation[name]["log_loss"],
            validation[name]["brier"],
            list(candidates).index(name),
        ),
    )
    chosen = candidates[selected]
    # Selection is now fixed. Test results cannot change which artifact is persisted.
    test = {
        name: evaluate(y[test_ids], pipe.predict_proba(X.iloc[test_ids])[:, 1])
        for name, pipe in candidates.items()
    }
    probabilities = chosen.predict_proba(X.iloc[test_ids])[:, 1]
    selected_model_name, calibration_method = selected.split(":")
    raw_model = raw_models[selected_model_name]
    bundle = {"raw": raw_model, "inference": chosen}
    joblib.dump(bundle, directory / "pipeline.joblib")
    dataset_label = (
        "SYNTHETIC ENGINEERING ONLY"
        if snapshot["provenance"] == "synthetic"
        else "REAL SOURCE RESEARCH; NOT PRODUCTION VALIDATION"
    )
    report = {
        "provenance": snapshot["provenance"],
        "interpretation": dataset_label,
        "selected": selected,
        "selection_rule": "validation log loss, then Brier, then simpler model",
        "validation": validation,
        "test": test,
        "uncertainty": bootstrap_intervals(y[test_ids], probabilities, bootstrap_replicates),
        "majority_baseline": {
            "class": int(y[train_ids].mean() >= 0.5),
            "test_accuracy": float((y[test_ids] == int(y[train_ids].mean() >= 0.5)).mean()),
        },
        "subgroups": subgroup_metrics(
            y[test_ids], probabilities, frame.iloc[test_ids][contract.categorical[0]].to_numpy()
        ),
        "error_analysis": {
            "mean_positive_probability": float(probabilities[y[test_ids] == 1].mean()),
            "mean_negative_probability": float(probabilities[y[test_ids] == 0].mean()),
        },
        "isotonic": "eligible"
        if any(":isotonic" in k for k in candidates)
        else "insufficient_calibration_support",
    }
    atomic_json(directory / "metrics.json", report)
    _plots(directory, test, selected)
    target, horizon = TARGETS[dataset]
    metadata = {
        "dataset_snapshot": snapshot_id,
        "source_sha256": snapshot["source_sha256"],
        "feature_version": FEATURE_VERSION,
        "source_revision": revision(),
        "seed": SEED,
        "hyperparameters": parameters,
        "quick_fixture_mode": quick,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("scikit-learn", "lightgbm", "numpy", "pandas", "polars", "joblib")
        },
    }
    files = {file.name: checksum(file) for file in directory.iterdir() if file.is_file()}
    manifest = {
        "artifact_format": "fleetrisk-pipeline-v1",
        "model_id": model_id,
        "version": "1",
        "dataset": dataset,
        "target": target,
        "horizon": horizon,
        "provenance": snapshot["provenance"],
        "feature_schema": schema,
        "calibration": calibration_method,
        "selected": selected,
        "files": files,
        **metadata,
    }
    atomic_json(directory / "manifest.json", manifest)
    (directory / "report.md").write_text(
        f"# Model experiment\n\n{dataset_label}\n\nTarget: {target}. Horizon: {horizon}.\n\n"
        f"Selected: {selected}, using validation proper scoring rules. Final test was held out.\n\n"
        f"Selected test metrics:\n\n```json\n{dumps(test[selected])}\n```\n\n"
        "See metrics.json for all frozen comparisons, uncertainty, support, and error analysis. "
        "No causal, lifetime-PD, recovery-model, or accounting-compliance claim.\n",
        encoding="utf-8",
    )
    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment("fleetrisk-phase1")
    with mlflow.start_run(run_name=model_id) as run:
        mlflow.set_tags(
            {"provenance": snapshot["provenance"], "snapshot_id": snapshot_id, "model_id": model_id}
        )
        mlflow.log_params({"selected": selected, "seed": SEED, "feature_version": FEATURE_VERSION})
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in test[selected].items()
                if k in ("roc_auc", "average_precision", "brier", "log_loss") and v is not None
            }
        )
        for file in directory.iterdir():
            if file.is_file():
                mlflow.log_artifact(str(file))
        metadata["mlflow_run_id"] = run.info.run_id
    with engine.begin() as connection:
        connection.execute(
            text(Path("db/queries/materialize_features.sql").read_text()),
            {
                "feature_id": feature_id,
                "snapshot_id": snapshot_id,
                "version": FEATURE_VERSION,
                "schema": dumps(schema),
            },
        )
        connection.execute(
            text(
                "INSERT INTO experiments(id,snapshot_id,feature_snapshot_id,metadata) VALUES (:id,:snapshot,:feature,CAST(:metadata AS jsonb))"
            ),
            {
                "id": experiment_id,
                "snapshot": snapshot_id,
                "feature": feature_id,
                "metadata": dumps(metadata),
            },
        )
        connection.execute(
            text("""INSERT INTO model_versions(id,experiment_id,version,target,horizon,provenance,artifact_path,manifest,validated)
            VALUES (:id,:experiment,'1',:target,:horizon,:provenance,:path,CAST(:manifest AS jsonb),true)"""),
            {
                "id": model_id,
                "experiment": experiment_id,
                "target": target,
                "horizon": horizon,
                "provenance": snapshot["provenance"],
                "path": model_id,
                "manifest": dumps(manifest),
            },
        )
    return {
        "model_id": model_id,
        "experiment_id": experiment_id,
        "directory": str(directory),
        "selected": selected,
        "provenance": snapshot["provenance"],
        "metrics": test[selected],
    }


def _plots(directory: Path, results: dict[str, Any], selected: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", label="identity")
    for name, result in results.items():
        curve = result["calibration"]
        ax.plot(curve["mean_probability"], curve["observed_rate"], marker=".", label=name)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed event fraction",
        title=f"Frozen test comparisons; selected {selected}",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "calibration.png", dpi=130)
    plt.close(fig)
