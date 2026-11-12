from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")


def evaluate(
    y: NDArray[Any], probabilities: NDArray[Any], threshold: float = 0.5
) -> dict[str, Any]:
    y = np.asarray(y)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Labels and probabilities must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary labels or probabilities")
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid classification threshold")
    predicted = p >= threshold
    both = len(np.unique(y)) == 2
    observed, estimated = calibration_curve(y, p, n_bins=10, strategy="quantile")
    order = np.argsort(p, kind="stable")
    deciles = [
        {
            "decile": i + 1,
            "count": len(indices),
            "mean_probability": float(p[indices].mean()),
            "label_rate": float(y[indices].mean()),
        }
        for i, indices in enumerate(np.array_split(order, min(10, len(y))))
    ]
    return {
        "n": len(y),
        "positive_count": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "calibration": {"mean_probability": estimated.tolist(), "observed_rate": observed.tolist()},
        "risk_deciles": deciles,
    }


def bootstrap_intervals(
    y: NDArray[Any], p: NDArray[Any], replicates: int = 500, seed: int = 42
) -> dict[str, Any]:
    if replicates < 2 or len(np.unique(y)) != 2:
        raise ValueError("Bootstrap requires both classes and at least two replicates")
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == label) for label in (0, 1)]
    samples: dict[str, list[float]] = {
        name: [] for name in ("roc_auc", "average_precision", "brier", "log_loss")
    }
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        a, b = y[indices], p[indices]
        for name, value in {
            "roc_auc": roc_auc_score(a, b),
            "average_precision": average_precision_score(a, b),
            "brier": brier_score_loss(a, b),
            "log_loss": log_loss(a, b, labels=[0, 1]),
        }.items():
            samples[name].append(float(value))
    return {
        "method": "stratified percentile bootstrap; conditional on class counts and frozen model",
        "replicates": replicates,
        "seed": seed,
        "confidence": 0.95,
        "intervals": {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in samples.items()},
    }


def subgroup_metrics(y: NDArray[Any], p: NDArray[Any], groups: NDArray[Any]) -> dict[str, Any]:
    output = {}
    for group in sorted({str(g) for g in groups}):
        mask = np.array([str(g) == group for g in groups])
        labels = y[mask]
        supported = (
            len(labels) >= 200 and int(labels.sum()) >= 20 and int((labels == 0).sum()) >= 20
        )
        output[group] = (
            evaluate(labels, p[mask])
            if supported
            else {"n": len(labels), "status": "insufficient_support"}
        )
    return output
