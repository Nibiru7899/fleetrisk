import hashlib
import logging
import math
import re
import secrets
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

import pandas as pd
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import Connection, text
from sqlalchemy.exc import SQLAlchemyError

from apps.api.schemas import (
    ErrorBody,
    IngestionRequest,
    IngestionResponse,
    ModelResponse,
    PortfolioBand,
    PortfolioResponse,
    PredictionResponse,
    ScoreRequest,
)
from fleetrisk.common.errors import FleetRiskError
from fleetrisk.common.io import dumps
from fleetrisk.config import Settings
from fleetrisk.data.db import make_engine, one, ready, rows
from fleetrisk.data.ingestion import ingest
from fleetrisk.features.registry import feature_frame
from fleetrisk.financial.scenarios import ExpectedLossRequest, ExpectedLossResult, expected_loss
from fleetrisk.governance.registry import load_model, model_record

logger = logging.getLogger("fleetrisk.api")
POLICY_VERSION = "descriptive-v1-unvalidated"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = make_engine(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        yield
        engine.dispose()

    application = FastAPI(
        title="FleetRisk OS",
        version="0.1.0",
        lifespan=lifespan,
        responses={422: {"model": ErrorBody}, 503: {"model": ErrorBody}},
    )
    application.state.settings = settings
    application.state.engine = engine

    @application.middleware("http")
    async def correlate(request: Request, call_next: Any) -> Any:
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied if re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", supplied) else str(uuid4())
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            dumps(
                {
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                }
            )
        )
        return response

    @application.exception_handler(FleetRiskError)
    async def domain_error(request: Request, error: FleetRiskError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            content=ErrorBody(
                code=error.code, message=str(error), request_id=request.state.request_id
            ).model_dump(),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorBody(
                code="invalid_request",
                message="Request violates the documented schema",
                request_id=request.state.request_id,
            ).model_dump(),
        )

    @application.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, error: SQLAlchemyError) -> JSONResponse:
        logger.error(dumps({"request_id": request.state.request_id, "error": type(error).__name__}))
        return JSONResponse(
            status_code=503,
            content=ErrorBody(
                code="database_unavailable",
                message="Database operation unavailable",
                request_id=request.state.request_id,
            ).model_dump(),
        )

    def connection() -> Iterator[Connection]:
        with engine.begin() as transaction:
            yield transaction

    def authenticate(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
            raise FleetRiskError("unauthorized", "A valid API key is required", 401)

    database = Annotated[Connection, Depends(connection, scope="function")]
    protected = [Depends(authenticate)]

    @application.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    def readiness() -> dict[str, Any]:
        if not ready(engine):
            raise FleetRiskError("migration_mismatch", "Database migration is incompatible", 503)
        if settings.model_id:
            load_model(engine, settings, settings.model_id)
        return {
            "status": "ready",
            "model_configured": bool(settings.model_id),
            "scoring_requires_explicit_model_id": True,
        }

    @application.post("/api/v1/data/ingestions", status_code=201, dependencies=protected)
    def create_ingestion(payload: IngestionRequest) -> IngestionResponse:
        try:
            result = ingest(
                engine, settings, payload.dataset, payload.relative_path, payload.provenance
            )
        except (ValueError, OSError) as error:
            raise FleetRiskError(
                "invalid_source", "Invalid source path, file, or contract"
            ) from error
        return IngestionResponse.model_validate(result)

    @application.get("/api/v1/data/ingestions/{ingestion_id}", dependencies=protected)
    def get_ingestion(ingestion_id: UUID, db: database) -> IngestionResponse:
        record = one(db, "SELECT * FROM ingestions WHERE id=:id", id=str(ingestion_id))
        if record is None:
            raise FleetRiskError("ingestion_not_found", "Ingestion does not exist", 404)
        return IngestionResponse.model_validate(
            {k: record[k] for k in IngestionResponse.model_fields}
        )

    @application.get("/api/v1/models", dependencies=protected)
    def models(db: database) -> list[ModelResponse]:
        return [
            _model_response(record)
            for record in rows(db, "SELECT * FROM model_versions ORDER BY created_at,id")
        ]

    @application.get("/api/v1/models/{model_id}", dependencies=protected)
    def get_model(model_id: UUID) -> ModelResponse:
        return _model_response(model_record(engine, str(model_id)))

    @application.post("/api/v1/risk/score", dependencies=protected)
    def score(payload: ScoreRequest, request: Request, db: database) -> PredictionResponse:
        record, bundle = load_model(engine, settings, str(payload.model_id))
        manifest = record["manifest"]
        schema = manifest["feature_schema"]
        if (payload.dataset, payload.provenance, payload.feature_schema_version) != (
            manifest["dataset"],
            record["provenance"],
            schema["version"],
        ):
            raise FleetRiskError(
                "schema_mismatch", "Dataset, provenance, or schema version does not match the model"
            )
        if set(payload.features) != set(schema["numeric"] + schema["categorical"]):
            raise FleetRiskError(
                "feature_mismatch", "Supply exactly the feature names in the registered schema"
            )
        for name in schema["numeric"]:
            value = payload.features[name]
            if value is not None and (
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise FleetRiskError("invalid_feature", f"Invalid numerical value for {name}")
        for name in schema["categorical"]:
            value = payload.features[name]
            if value is not None and (not isinstance(value, str) or len(value) > 160):
                raise FleetRiskError("invalid_feature", f"Invalid categorical value for {name}")
        if (
            payload.dataset == "ltfs"
            and payload.features["ltv"] is not None
            and float(payload.features["ltv"]) > 100
        ):
            raise FleetRiskError("invalid_feature", "ltv must be a percentage from 0 to 100")
        features = feature_frame(pd.DataFrame([payload.features]), payload.dataset)
        raw = float(bundle["raw"].predict_proba(features)[0, 1])
        selected = float(bundle["inference"].predict_proba(features)[0, 1])
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in (raw, selected)):
            raise FleetRiskError(
                "invalid_model_output", "Model returned invalid probabilities", 503
            )
        calibrated = selected if manifest["calibration"] != "none" else None
        band = "lower" if selected < 0.10 else "moderate" if selected < 0.25 else "higher"
        prediction_id = str(uuid4())
        feature_hash = hashlib.sha256(dumps(payload.features).encode()).hexdigest()
        record_prediction = one(
            db,
            """INSERT INTO predictions(id,model_id,application_key,dataset_id,provenance,
            raw_probability,calibrated_probability,risk_band,feature_schema_version,feature_sha256,policy_version,request_id)
            VALUES (:id,:model,:application,:dataset,:provenance,:raw,:calibrated,:band,:schema,:hash,:policy,:request)
            RETURNING *""",
            id=prediction_id,
            model=str(payload.model_id),
            application=payload.application_key,
            dataset=payload.dataset,
            provenance=payload.provenance,
            raw=raw,
            calibrated=calibrated,
            band=band,
            schema=schema["version"],
            hash=feature_hash,
            policy=POLICY_VERSION,
            request=request.state.request_id,
        )
        db.execute(
            text(
                "INSERT INTO audit_events(event_type,entity_id,request_id,metadata) VALUES ('prediction',:id,:request,CAST(:metadata AS jsonb))"
            ),
            {
                "id": prediction_id,
                "request": request.state.request_id,
                "metadata": dumps(
                    {"model_id": str(payload.model_id), "provenance": payload.provenance}
                ),
            },
        )
        assert record_prediction is not None
        return _prediction_response(record_prediction, record)

    @application.get("/api/v1/risk/predictions/{prediction_id}", dependencies=protected)
    def get_prediction(prediction_id: UUID, db: database) -> PredictionResponse:
        record = one(db, "SELECT * FROM predictions WHERE id=:id", id=str(prediction_id))
        if record is None:
            raise FleetRiskError("prediction_not_found", "Prediction does not exist", 404)
        return _prediction_response(record, model_record(engine, str(record["model_id"])))

    @application.get("/api/v1/portfolio/summary", dependencies=protected)
    def portfolio(model_id: UUID, db: database) -> PortfolioResponse:
        record = model_record(engine, str(model_id))
        manifest = record["manifest"]
        bands = rows(
            db,
            Path("db/queries/portfolio_summary.sql").read_text(),
            model_id=str(model_id),
            dataset_id=manifest["dataset"],
            provenance=record["provenance"],
        )
        return PortfolioResponse(
            model_id=model_id,
            dataset=manifest["dataset"],
            provenance=record["provenance"],
            target=record["target"],
            horizon=record["horizon"],
            bands=[PortfolioBand.model_validate(band) for band in bands],
        )

    @application.post("/api/v1/financial/expected-loss", dependencies=protected)
    def financial(
        payload: ExpectedLossRequest, request: Request, db: database
    ) -> ExpectedLossResult:
        result = expected_loss(payload)
        db.execute(
            text(
                "INSERT INTO financial_assumptions(id,definition,result,request_id) VALUES (:id,CAST(:definition AS jsonb),CAST(:result AS jsonb),:request)"
            ),
            {
                "id": str(uuid4()),
                "definition": payload.model_dump_json(),
                "result": result.model_dump_json(),
                "request": request.state.request_id,
            },
        )
        return result

    return application


def _model_response(record: dict[str, Any]) -> ModelResponse:
    return ModelResponse.model_validate({k: record[k] for k in ModelResponse.model_fields})


def _prediction_response(record: dict[str, Any], model: dict[str, Any]) -> PredictionResponse:
    return PredictionResponse(
        **record,
        model_version=model["version"],
        target=model["target"],
        horizon=model["horizon"],
        explanation_metadata={
            "status": "not_implemented",
            "interpretation": "Descriptive probability; no causal explanation or lending recommendation",
        },
    )


app = create_app()
