from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, StrictFloat, StrictInt, StrictStr

from fleetrisk.financial.scenarios import StrictModel

Dataset = Literal[
    "ltfs",
    "home_credit_application",
    "home_credit_bureau",
    "home_credit_previous",
    "home_credit_installments",
]
Provenance = Literal["real", "synthetic"]


class ErrorBody(StrictModel):
    code: str
    message: str
    request_id: str


class IngestionRequest(StrictModel):
    dataset: Dataset
    relative_path: str = Field(min_length=1, max_length=500)
    provenance: Provenance


class IngestionResponse(StrictModel):
    id: UUID
    snapshot_id: str | None
    status: str
    quality: dict[str, Any]
    provenance: Provenance


class ModelResponse(StrictModel):
    id: UUID
    version: str
    target: str
    horizon: str
    provenance: Provenance
    validated: bool
    manifest: dict[str, Any]
    created_at: datetime


class ScoreRequest(StrictModel):
    model_id: UUID
    application_key: str = Field(min_length=1, max_length=160)
    dataset: Literal["ltfs", "home_credit_application"]
    provenance: Provenance
    feature_schema_version: str
    features: dict[str, StrictFloat | StrictInt | StrictStr | None]


class PredictionResponse(StrictModel):
    id: UUID
    model_id: UUID
    model_version: str
    application_key: str
    dataset_id: str
    provenance: Provenance
    target: str
    horizon: str
    raw_probability: float
    calibrated_probability: float | None
    risk_band: str
    policy_version: str
    feature_schema_version: str
    feature_sha256: str
    request_id: str
    created_at: datetime
    explanation_metadata: dict[str, str]


class PortfolioBand(StrictModel):
    risk_band: str
    applications: int
    mean_probability: float
    calibrated_count: int


class PortfolioResponse(StrictModel):
    model_id: UUID
    dataset: str
    provenance: Provenance
    target: str
    horizon: str
    bands: list[PortfolioBand]
    interpretation: str = "Latest prediction per application; descriptive unvalidated risk bands"
