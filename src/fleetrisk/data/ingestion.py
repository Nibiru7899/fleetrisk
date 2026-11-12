import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow.parquet as pq
from sqlalchemy import Engine, text

from fleetrisk.common.errors import FleetRiskError
from fleetrisk.common.io import atomic_json, checksum, dumps
from fleetrisk.config import Settings, contained_path
from fleetrisk.data.contracts import get_contract
from fleetrisk.data.db import one
from fleetrisk.data.validation import normalized_scan, validate


def ingest(
    engine: Engine,
    settings: Settings,
    dataset: str,
    relative_path: str,
    provenance: str,
    *,
    replay_of: str | None = None,
    fail_after_batches: int | None = None,
) -> dict[str, Any]:
    contract = get_contract(dataset)
    if provenance not in ("real", "synthetic"):
        raise FleetRiskError("invalid_provenance", "Use real or synthetic provenance")
    if provenance == "synthetic" and not settings.allow_synthetic:
        raise FleetRiskError("synthetic_disabled", "Synthetic mode is explicitly disabled", 409)
    path = contained_path(settings.source_root, relative_path)
    if not path.is_file() or path.stat().st_size > settings.max_source_bytes:
        raise FleetRiskError("invalid_source", "Source missing or exceeds configured byte limit")
    source_hash = checksum(path)
    snapshot_id = hashlib.sha256(
        f"{dataset}:{source_hash}:{contract.version}:{provenance}".encode()
    ).hexdigest()
    run_id = str(uuid4())
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO data_sources(id,description) VALUES (:id,:description) ON CONFLICT DO NOTHING"
            ),
            {"id": dataset, "description": f"{dataset}: separate source population"},
        )
        connection.execute(
            text("""INSERT INTO ingestions(id, source_id, relative_path, source_sha256,
            contract_version, provenance, status, replay_of)
            VALUES (:id,:source,:path,:sha,:version,:provenance,'running',:replay)"""),
            {
                "id": run_id,
                "source": dataset,
                "path": relative_path,
                "sha": source_hash,
                "version": contract.version,
                "provenance": provenance,
                "replay": replay_of,
            },
        )
    quality: dict[str, Any] = {}
    staging: Path | None = None
    try:
        quality = validate(path, dataset)
        quality["provenance"] = provenance
        if not quality["valid"]:
            raise FleetRiskError(
                "data_quality_failed", "Dataset violates contract; inspect quality report"
            )
        destination = settings.processed_root.resolve() / snapshot_id
        destination.mkdir(parents=True, exist_ok=True)
        staging = destination / f"{run_id}.staging.parquet"
        normalized_scan(path, dataset).sink_parquet(
            staging, engine="streaming", maintain_order=True
        )
        if checksum(path) != source_hash:
            raise FleetRiskError("source_changed", "Source changed during ingestion", 409)
        parquet_hash = checksum(staging)
        with engine.begin() as connection:
            # Serialize publication of the same snapshot across processes; released on rollback.
            connection.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": int(snapshot_id[:15], 16)}
            )
            existing = one(
                connection, "SELECT * FROM dataset_snapshots WHERE id=:id", id=snapshot_id
            )
            if existing:
                if (
                    not Path(existing["parquet_path"]).is_file()
                    or checksum(Path(existing["parquet_path"])) != existing["parquet_sha256"]
                ):
                    raise FleetRiskError(
                        "snapshot_corrupt",
                        "Existing committed snapshot failed integrity validation",
                        409,
                    )
                status = "duplicate"
            else:
                final_path = destination / "data.parquet"
                manifest = {
                    "source_sha256": source_hash,
                    "provenance": provenance,
                    "contract_version": contract.version,
                    "dataset": dataset,
                }
                connection.execute(
                    text("""INSERT INTO dataset_snapshots(id,source_id,source_sha256,
                    contract_version,provenance,row_count,parquet_path,parquet_sha256,source_manifest)
                    VALUES (:id,:source,:sha,:version,:provenance,:count,:path,:parquet_sha,CAST(:manifest AS jsonb))"""),
                    {
                        "id": snapshot_id,
                        "source": dataset,
                        "sha": source_hash,
                        "version": contract.version,
                        "provenance": provenance,
                        "count": quality["row_count"],
                        "path": str(final_path),
                        "parquet_sha": parquet_hash,
                        "manifest": dumps(manifest),
                    },
                )
                for batch_number, batch in enumerate(
                    pq.ParquetFile(staging).iter_batches(batch_size=2000), 1
                ):
                    records = batch.to_pylist()
                    payloads = []
                    for record in records:
                        source_key = dumps([record[k] for k in contract.key])
                        if contract.historical:
                            payloads.append(
                                {
                                    "snapshot": snapshot_id,
                                    "key": source_key,
                                    "application": str(record["SK_ID_CURR"]),
                                    "previous": str(record.get("SK_ID_PREV"))
                                    if record.get("SK_ID_PREV") is not None
                                    else None,
                                    "day": record.get(
                                        "DAYS_CREDIT",
                                        record.get(
                                            "DAYS_DECISION", record.get("DAYS_ENTRY_PAYMENT")
                                        ),
                                    ),
                                    "attributes": dumps(record),
                                }
                            )
                        else:
                            attributes = {
                                k: record[k] for k in (*contract.numeric, *contract.categorical)
                            }
                            payloads.append(
                                {
                                    "snapshot": snapshot_id,
                                    "key": source_key,
                                    "target": record.get(contract.target),
                                    "attributes": dumps(attributes),
                                }
                            )
                    if contract.historical:
                        connection.execute(
                            text("""INSERT INTO historical_credit_rows(snapshot_id,source_key,application_key,previous_key,relative_day,attributes)
                            VALUES (:snapshot,:key,:application,:previous,:day,CAST(:attributes AS jsonb))"""),
                            payloads,
                        )
                    else:
                        connection.execute(
                            text("""INSERT INTO applications(snapshot_id,source_key,target,attributes)
                            VALUES (:snapshot,:key,:target,CAST(:attributes AS jsonb))"""),
                            payloads,
                        )
                    if fail_after_batches is not None and batch_number >= fail_after_batches:
                        raise RuntimeError("Injected interrupted-load failure")
                if not contract.historical:
                    amount = "disbursed_amount" if dataset == "ltfs" else "AMT_CREDIT"
                    asset = "(attributes->>'asset_cost')::numeric" if dataset == "ltfs" else "NULL"
                    # Identifiers come only from the fixed contract, never user SQL.
                    connection.execute(
                        text(f"""INSERT INTO loan_observations(application_id,amount,asset_value,currency)
                        SELECT id,(attributes->>'{amount}')::numeric,{asset},:currency
                        FROM applications WHERE snapshot_id=:snapshot"""),
                        {"snapshot": snapshot_id, "currency": "INR" if dataset == "ltfs" else None},
                    )
                staging.replace(final_path)
                atomic_json(
                    destination / "manifest.json",
                    {**manifest, "parquet_sha256": parquet_hash, "quality": quality},
                )
                status = "completed"
            connection.execute(
                text("""UPDATE ingestions SET status=:status,snapshot_id=:snapshot,
                quality=CAST(:quality AS jsonb),finished_at=now() WHERE id=:id"""),
                {
                    "id": run_id,
                    "status": status,
                    "snapshot": snapshot_id,
                    "quality": dumps(quality),
                },
            )
            connection.execute(
                text(
                    "INSERT INTO quality_results(ingestion_id,result) VALUES (:id,CAST(:quality AS jsonb))"
                ),
                {"id": run_id, "quality": dumps(quality)},
            )
            connection.execute(
                text(
                    "INSERT INTO audit_events(event_type,entity_id,metadata) VALUES ('ingestion',:id,CAST(:metadata AS jsonb))"
                ),
                {"id": run_id, "metadata": dumps({"status": status, "snapshot": snapshot_id})},
            )
        return {
            "id": run_id,
            "snapshot_id": snapshot_id,
            "status": status,
            "quality": quality,
            "provenance": provenance,
        }
    except Exception as error:
        code = error.code if isinstance(error, FleetRiskError) else "ingestion_failed"
        with engine.begin() as connection:
            connection.execute(
                text("""UPDATE ingestions SET status='failed',error_code=:code,
                quality=CAST(:quality AS jsonb),finished_at=now() WHERE id=:id"""),
                {"id": run_id, "code": code, "quality": dumps(quality)},
            )
            connection.execute(
                text(
                    "INSERT INTO quality_results(ingestion_id,result) VALUES (:id,CAST(:quality AS jsonb))"
                ),
                {"id": run_id, "quality": dumps(quality)},
            )
        raise
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
