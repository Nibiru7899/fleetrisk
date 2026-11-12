import importlib.metadata
import json
from typing import Any

import joblib
from sqlalchemy import Engine

from fleetrisk.common.errors import FleetRiskError
from fleetrisk.common.io import checksum
from fleetrisk.config import Settings, contained_path
from fleetrisk.data.db import one
from fleetrisk.features.registry import feature_schema


def model_record(engine: Engine, model_id: str) -> dict[str, Any]:
    with engine.connect() as connection:
        record = one(connection, "SELECT * FROM model_versions WHERE id=:id", id=model_id)
    if record is None:
        raise FleetRiskError("model_not_found", "Model is not registered", 404)
    return record


def load_model(
    engine: Engine, settings: Settings, model_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = model_record(engine, model_id)
    if not record["validated"]:
        raise FleetRiskError("model_unavailable", "Model has not passed artifact validation", 503)
    if record["provenance"] == "synthetic" and not settings.allow_synthetic:
        raise FleetRiskError(
            "synthetic_disabled", "Fixture models cannot serve ordinary scoring", 409
        )
    try:
        directory = contained_path(settings.artifact_root, record["artifact_path"])
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest != record["manifest"]:
            raise ValueError("Manifest differs from trusted registry record")
        if manifest["artifact_format"] != "fleetrisk-pipeline-v1" or manifest["model_id"] != str(
            record["id"]
        ):
            raise ValueError("Artifact format or identity mismatch")
        if (
            manifest["provenance"] != record["provenance"]
            or manifest["target"] != record["target"]
            or manifest["horizon"] != record["horizon"]
        ):
            raise ValueError("Artifact semantic mismatch")
        if manifest["feature_schema"] != feature_schema(manifest["dataset"]):
            raise ValueError("Feature schema incompatible with this runtime")
        for name, version in manifest["dependencies"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Dependency compatibility mismatch: {name}")
        for filename, expected in manifest["files"].items():
            file = contained_path(directory, filename)
            if checksum(file) != expected:
                raise ValueError(f"Integrity mismatch: {filename}")
        if "pipeline.joblib" not in manifest["files"]:
            raise ValueError("Pipeline checksum missing")
        # joblib is executable serialization. Only registry-created local bundles
        # are accepted; this service has no artifact-upload endpoint.
        bundle = joblib.load(directory / "pipeline.joblib")
        if set(bundle) != {"raw", "inference"}:
            raise ValueError("Unexpected bundle structure")
    except (ValueError, KeyError, OSError, TypeError, ImportError) as error:
        raise FleetRiskError(
            "artifact_incompatible", "Model artifact failed integrity or compatibility checks", 409
        ) from error
    return record, bundle
