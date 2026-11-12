from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Rate = Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
Amount = Annotated[Decimal, Field(ge=0, max_digits=20, decimal_places=4, allow_inf_nan=False)]
Provenance = Literal["observed", "modeled", "assumed"]
CURRENCY_SCALE = {"INR": 2, "USD": 2, "EUR": 2, "GBP": 2, "JPY": 0}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RateInput(StrictModel):
    value: Rate
    provenance: Provenance
    event_definition: str = Field(min_length=1, max_length=500)
    horizon: str = Field(min_length=1, max_length=120)


class ExposureInput(StrictModel):
    amount: Amount
    currency: Literal["INR", "USD", "EUR", "GBP", "JPY"]
    unit: Literal["major"] = "major"
    provenance: Provenance
    event_definition: str = Field(min_length=1, max_length=500)
    horizon: str = Field(min_length=1, max_length=120)
    exposure_limit: Amount | None = None

    @model_validator(mode="after")
    def within_limit(self) -> "ExposureInput":
        if self.exposure_limit is not None and self.amount > self.exposure_limit:
            raise ValueError("EAD exceeds the explicitly declared exposure limit")
        return self


class ExpectedLossRequest(StrictModel):
    pd: RateInput
    ead: ExposureInput
    lgd: RateInput | None = None
    recovery_rate: RateInput | None = None
    scenario_name: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def compatible(self) -> "ExpectedLossRequest":
        if (self.lgd is None) == (self.recovery_rate is None):
            raise ValueError("Supply exactly one of LGD or recovery_rate")
        quantities = [self.pd, self.ead, self.lgd or self.recovery_rate]
        if len({(q.event_definition, q.horizon) for q in quantities if q is not None}) != 1:
            raise ValueError("PD, LGD/recovery, and EAD must share event definition and horizon")
        return self


class ExpectedLossResult(StrictModel):
    scenario_name: str
    expected_loss: Decimal
    unrounded_expected_loss: Decimal
    pd: Decimal
    lgd: Decimal
    ead: Decimal
    currency: str
    unit: Literal["major"] = "major"
    event_definition: str
    horizon: str
    input_provenance: dict[str, str]
    interpretation: str = "Scenario analysis; no empirical recovery or lifetime-PD claim"


def quantize_money(amount: Decimal, currency: str) -> Decimal:
    if currency not in CURRENCY_SCALE:
        raise ValueError("Unsupported currency; no implicit FX conversion")
    with localcontext() as context:
        context.prec = 40
        return amount.quantize(
            Decimal(1).scaleb(-CURRENCY_SCALE[currency]), rounding=ROUND_HALF_EVEN
        )


def expected_loss(request: ExpectedLossRequest) -> ExpectedLossResult:
    with localcontext() as context:
        context.prec = 40
        if request.lgd is not None:
            lgd = request.lgd.value
            lgd_provenance: str = request.lgd.provenance
        else:
            assert request.recovery_rate is not None
            lgd = 1 - request.recovery_rate.value
            lgd_provenance = f"derived_from_{request.recovery_rate.provenance}_recovery"
        unrounded = request.pd.value * lgd * request.ead.amount
        return ExpectedLossResult(
            scenario_name=request.scenario_name,
            expected_loss=quantize_money(unrounded, request.ead.currency),
            unrounded_expected_loss=unrounded,
            pd=request.pd.value,
            lgd=lgd,
            ead=request.ead.amount,
            currency=request.ead.currency,
            event_definition=request.pd.event_definition,
            horizon=request.pd.horizon,
            input_provenance={
                "pd": request.pd.provenance,
                "lgd": lgd_provenance,
                "ead": request.ead.provenance,
            },
        )


def loan_to_value(loan: ExposureInput, collateral: ExposureInput) -> Decimal:
    if loan.currency != collateral.currency or loan.unit != collateral.unit:
        raise ValueError("Currency and unit mismatch")
    if collateral.amount == 0:
        raise ValueError("LTV is undefined for zero collateral value")
    with localcontext() as context:
        context.prec = 40
        return loan.amount / collateral.amount


def collateral_recovery(
    collateral: ExposureInput, haircut: Rate, costs: Amount = Decimal(0)
) -> Decimal:
    if not haircut.is_finite() or not 0 <= haircut <= 1 or not costs.is_finite() or costs < 0:
        raise ValueError("Invalid haircut or recovery costs")
    with localcontext() as context:
        context.prec = 40
        return max(Decimal(0), collateral.amount * (1 - haircut) - costs)


def collateral_lgd(
    ead: ExposureInput, collateral: ExposureInput, haircut: Rate, costs: Amount = Decimal(0)
) -> Decimal:
    if (ead.currency, ead.unit, ead.event_definition, ead.horizon) != (
        collateral.currency,
        collateral.unit,
        collateral.event_definition,
        collateral.horizon,
    ):
        raise ValueError("Collateral recovery scenario is incompatible with exposure")
    if ead.amount == 0:
        raise ValueError("LGD is undefined for zero exposure; supply an explicit scenario LGD")
    with localcontext() as context:
        context.prec = 40
        return 1 - min(ead.amount, collateral_recovery(collateral, haircut, costs)) / ead.amount


def aggregate_scenarios(results: list[ExpectedLossResult]) -> dict[str, str | int]:
    if not results:
        raise ValueError("At least one scenario is required")
    definitions = {(r.currency, r.unit, r.event_definition, r.horizon) for r in results}
    if len(definitions) != 1:
        raise ValueError("Cannot aggregate incompatible currencies, units, events, or horizons")
    with localcontext() as context:
        context.prec = 40
        total = sum((r.unrounded_expected_loss for r in results), Decimal(0))
        return {
            "count": len(results),
            "currency": results[0].currency,
            "expected_loss": str(quantize_money(total, results[0].currency)),
            "event_definition": results[0].event_definition,
            "horizon": results[0].horizon,
        }
