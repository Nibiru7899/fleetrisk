"""Run the real HTTP + PostgreSQL proof. Fixture metrics are engineering evidence only."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import polars as pl
from alembic import command
from alembic.config import Config

from fleetrisk.common.io import atomic_json
from fleetrisk.config import Settings
from fleetrisk.data.db import make_engine, one
from fleetrisk.data.fixtures import write_ltfs_fixture
from fleetrisk.features.registry import FEATURE_VERSION
from fleetrisk.modeling.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixture", "real"], required=True)
    parser.add_argument("--source", help="Real LTFS file relative to configured source root")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--quick", action="store_true", help="Fixture-only reduced tuning budget")
    args = parser.parse_args()
    if args.mode == "real" and not args.source:
        parser.error("Real mode requires --source and permitted source files")
    if args.mode == "real" and args.quick:
        parser.error("Quick mode cannot be used for empirical experiments")
    os.environ["FLEETRISK_ALLOW_SYNTHETIC"] = "true" if args.mode == "fixture" else "false"
    settings = Settings()
    if args.mode == "fixture":
        source = "fixtures/ltfs.csv"
        write_ltfs_fixture(settings.source_root / source)
    else:
        source = args.source
    command.upgrade(Config("alembic.ini"), "head")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Proof port is occupied; refusing to use an unrelated server")
    output = Path("reports/runtime")
    output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "mode": args.mode,
        "provenance": "synthetic" if args.mode == "fixture" else "real",
        "checks": {},
    }
    headers = {"X-API-Key": settings.api_key} if settings.api_key else {}
    with (output / f"api-{args.mode}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "apps.api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{args.port}", headers=headers, timeout=120
            ) as client:
                for _ in range(150):
                    if process.poll() is not None:
                        raise RuntimeError("API exited; inspect proof log")
                    try:
                        if client.get("/health/live").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError("API did not become live")

                def request(method: str, endpoint: str, expected: int = 200, **kwargs):
                    response = client.request(method, endpoint, **kwargs)
                    assert response.status_code == expected, (
                        endpoint,
                        response.status_code,
                        response.text,
                    )
                    evidence["checks"][f"{method} {endpoint}"] = response.status_code
                    return response.json()

                request("GET", "/health/ready")
                ingestion = {
                    "dataset": "ltfs",
                    "relative_path": source,
                    "provenance": evidence["provenance"],
                }
                first = request("POST", "/api/v1/data/ingestions", 201, json=ingestion)
                second = request("POST", "/api/v1/data/ingestions", 201, json=ingestion)
                assert (
                    second["status"] == "duplicate"
                    and first["snapshot_id"] == second["snapshot_id"]
                )
                request("GET", f"/api/v1/data/ingestions/{first['id']}")
                engine = make_engine(settings)
                trained = train(engine, settings, first["snapshot_id"], quick=args.quick)
                model_id = trained["model_id"]
                evidence["model_id"] = model_id
                evidence["snapshot_id"] = first["snapshot_id"]
                request("GET", "/api/v1/models")
                model = request("GET", f"/api/v1/models/{model_id}")
                with engine.connect() as connection:
                    snapshot = one(
                        connection,
                        "SELECT * FROM dataset_snapshots WHERE id=:id",
                        id=first["snapshot_id"],
                    )
                row = pl.read_parquet(snapshot["parquet_path"]).row(0, named=True)
                schema = model["manifest"]["feature_schema"]
                payload = {
                    "model_id": model_id,
                    "application_key": "vertical-slice-application",
                    "dataset": "ltfs",
                    "provenance": evidence["provenance"],
                    "feature_schema_version": FEATURE_VERSION,
                    "features": {
                        name: row[name] for name in schema["numeric"] + schema["categorical"]
                    },
                }
                p1 = request("POST", "/api/v1/risk/score", json=payload)
                p2 = request("POST", "/api/v1/risk/score", json=payload)
                assert (p1["raw_probability"], p1["calibrated_probability"]) == (
                    p2["raw_probability"],
                    p2["calibrated_probability"],
                )
                persisted = request("GET", f"/api/v1/risk/predictions/{p1['id']}")
                assert persisted["model_id"] == model_id and persisted["model_version"] == "1"
                invalid = {**payload, "features": {}}
                request("POST", "/api/v1/risk/score", 422, json=invalid)
                request(
                    "POST", "/api/v1/risk/score", 404, json={**payload, "model_id": str(uuid4())}
                )
                manifest_path = Path(trained["directory"]) / "manifest.json"
                original = manifest_path.read_bytes()
                try:
                    changed = json.loads(original)
                    changed["artifact_format"] = "incompatible-v999"
                    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
                    request("POST", "/api/v1/risk/score", 409, json=payload)
                finally:
                    manifest_path.write_bytes(original)
                summary = request("GET", f"/api/v1/portfolio/summary?model_id={model_id}")
                assert sum(band["applications"] for band in summary["bands"]) == 1
                assumptions = {
                    "event_definition": "illustrative_scenario_event",
                    "horizon": "12 months",
                    "provenance": "assumed",
                }
                financial = request(
                    "POST",
                    "/api/v1/financial/expected-loss",
                    json={
                        "scenario_name": "HTTP scenario",
                        "pd": {**assumptions, "value": "0.2"},
                        "lgd": {**assumptions, "value": "0.5"},
                        "ead": {**assumptions, "amount": "1000", "currency": "INR"},
                    },
                )
                assert financial["expected_loss"] == "100.00"
                with engine.connect() as connection:
                    recorded = one(
                        connection,
                        "SELECT count(*) AS n FROM predictions WHERE model_id=:id",
                        id=model_id,
                    )
                    assert recorded["n"] == 2
                evidence["invariants"] = [
                    "consistent_predictions",
                    "correct_persisted_model",
                    "strict_feature_schema",
                    "incompatible_artifact_rejected",
                    "unknown_model_controlled",
                    "idempotent_ingestion",
                    "latest_application_portfolio",
                    "financial_scenario_arithmetic",
                ]
                evidence["status"] = (
                    "PASS_ENGINEERING_ONLY" if args.mode == "fixture" else "PASS_REAL_DATA_FLOW"
                )
                atomic_json(output / f"vertical-slice-{args.mode}.json", evidence)
                print(json.dumps(evidence, indent=2))
                engine.dispose()
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
