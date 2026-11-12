import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from apps.api.main import create_app
from fleetrisk.common.errors import FleetRiskError
from fleetrisk.config import Settings
from fleetrisk.data.db import one, ready, rows
from fleetrisk.data.ingestion import ingest
from fleetrisk.features.registry import FEATURE_VERSION
from fleetrisk.governance.registry import load_model
from fleetrisk.modeling.training import train

pytestmark = pytest.mark.postgres


def test_migration_roundtrip(postgres) -> None:
    assert ready(postgres)
    command.downgrade(Config("alembic.ini"), "base")
    command.upgrade(Config("alembic.ini"), "head")
    assert ready(postgres)


def test_idempotency_and_concurrency(postgres, settings: Settings, source: Path) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: ingest(postgres, settings, "ltfs", "ltfs.csv", "synthetic"), range(2)
            )
        )
    assert sorted(r["status"] for r in results) == ["completed", "duplicate"]
    with postgres.connect() as connection:
        assert one(connection, "SELECT count(*) AS n FROM applications")["n"] == 300
        assert one(connection, "SELECT count(*) AS n FROM dataset_snapshots")["n"] == 1


def test_interrupted_loading_rolls_back_and_replays(
    postgres, settings: Settings, source: Path
) -> None:
    with pytest.raises(RuntimeError, match="Injected"):
        ingest(postgres, settings, "ltfs", "ltfs.csv", "synthetic", fail_after_batches=1)
    with postgres.connect() as connection:
        assert one(connection, "SELECT count(*) AS n FROM applications")["n"] == 0
        assert one(connection, "SELECT count(*) AS n FROM dataset_snapshots")["n"] == 0
        failed = one(connection, "SELECT * FROM ingestions")
        assert failed["status"] == "failed"
    result = ingest(
        postgres, settings, "ltfs", "ltfs.csv", "synthetic", replay_of=str(failed["id"])
    )
    assert result["status"] == "completed"


def test_invalid_schema_leaves_failure_evidence(postgres, settings: Settings, source: Path) -> None:
    pl.read_csv(source).drop("asset_cost").write_csv(source)
    with pytest.raises(FleetRiskError, match="contract"):
        ingest(postgres, settings, "ltfs", "ltfs.csv", "synthetic")
    with postgres.connect() as connection:
        assert one(connection, "SELECT status FROM ingestions")["status"] == "failed"
        assert one(connection, "SELECT count(*) AS n FROM quality_results")["n"] == 1


def test_constraints_and_analytical_sql(postgres, settings: Settings, source: Path) -> None:
    result = ingest(postgres, settings, "ltfs", "ltfs.csv", "synthetic")
    with postgres.connect() as connection:
        cohorts = rows(
            connection,
            Path("db/queries/cohorts.sql").read_text(),
            snapshot_id=result["snapshot_id"],
        )
        assert sum(row["applications"] for row in cohorts) == 300
        assert not rows(connection, Path("db/queries/deduplication.sql").read_text())
        plan = rows(
            connection,
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT * FROM applications WHERE snapshot_id=:id",
            id=result["snapshot_id"],
        )
        assert plan
    with pytest.raises(IntegrityError), postgres.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO applications(snapshot_id,source_key,target,attributes) VALUES (:id,'invalid',2,'{}')"
            ),
            {"id": result["snapshot_id"]},
        )
    with pytest.raises(IntegrityError), postgres.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO applications(snapshot_id,source_key,target,attributes) VALUES ('missing','invalid',1,'{}')"
            )
        )


def test_model_api_persistence_and_artifact_controls(
    postgres, settings: Settings, source: Path
) -> None:
    ingestion = ingest(postgres, settings, "ltfs", "ltfs.csv", "synthetic")
    trained = train(
        postgres, settings, ingestion["snapshot_id"], quick=True, bootstrap_replicates=20
    )
    model_id = trained["model_id"]
    model, bundle = load_model(postgres, settings, model_id)
    assert set(bundle) == {"raw", "inference"}
    schema = model["manifest"]["feature_schema"]
    row = pl.read_csv(source).row(0, named=True)
    payload = {
        "model_id": model_id,
        "application_key": "application-1",
        "dataset": "ltfs",
        "provenance": "synthetic",
        "feature_schema_version": FEATURE_VERSION,
        "features": {name: row[name] for name in schema["numeric"] + schema["categorical"]},
    }
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 200
        first = client.post("/api/v1/risk/score", json=payload)
        assert first.status_code == 200, first.text
        second = client.post("/api/v1/risk/score", json=payload)
        assert second.status_code == 200
        assert first.json()["raw_probability"] == second.json()["raw_probability"]
        persisted = client.get(f"/api/v1/risk/predictions/{first.json()['id']}")
        assert persisted.json()["model_id"] == model_id
        summary = client.get(f"/api/v1/portfolio/summary?model_id={model_id}").json()
        assert sum(band["applications"] for band in summary["bands"]) == 1
        invalid = client.post(
            "/api/v1/risk/score",
            json={**payload, "features": {**payload["features"], "loan_default": 1}},
        )
        assert invalid.status_code == 422
        assert (
            client.post(
                "/api/v1/risk/score", json={**payload, "model_id": str(uuid4())}
            ).status_code
            == 404
        )
        assert (
            client.post("/api/v1/risk/score", json={**payload, "provenance": "real"}).status_code
            == 422
        )
        manifest = Path(trained["directory"]) / "manifest.json"
        original = manifest.read_bytes()
        try:
            changed = json.loads(original)
            changed["dependencies"]["scikit-learn"] = "incompatible"
            manifest.write_text(json.dumps(changed))
            assert client.post("/api/v1/risk/score", json=payload).status_code == 409
        finally:
            manifest.write_bytes(original)
    settings.allow_synthetic = False
    with pytest.raises(FleetRiskError, match="Fixture"):
        load_model(postgres, settings, model_id)


def test_auth_and_missing_model_behavior(postgres, settings: Settings) -> None:
    settings.api_key = "test-only-secret"
    settings.model_id = str(uuid4())
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 404
        assert client.get("/api/v1/models").status_code == 401
        assert (
            client.get("/api/v1/models", headers={"X-API-Key": settings.api_key}).status_code == 200
        )
