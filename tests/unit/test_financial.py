from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from fleetrisk.financial.scenarios import (
    ExpectedLossRequest,
    aggregate_scenarios,
    collateral_lgd,
    collateral_recovery,
    expected_loss,
    loan_to_value,
    quantize_money,
)


def scenario(pd: str = "0.2", lgd: str = "0.5", ead: str = "1000") -> ExpectedLossRequest:
    definition = {
        "event_definition": "scenario_event",
        "horizon": "12 months",
        "provenance": "assumed",
    }
    return ExpectedLossRequest.model_validate(
        {
            "scenario_name": "boundary",
            "pd": {**definition, "value": pd},
            "lgd": {**definition, "value": lgd},
            "ead": {**definition, "amount": ead, "currency": "INR"},
        }
    )


@pytest.mark.parametrize(
    "pd,lgd,ead,el",
    [
        ("0", "1", "100", "0"),
        ("1", "1", "100", "100"),
        ("1", "0", "100", "0"),
        ("1", "1", "0", "0"),
    ],
)
def test_boundaries(pd: str, lgd: str, ead: str, el: str) -> None:
    assert expected_loss(scenario(pd, lgd, ead)).expected_loss == Decimal(el)


@pytest.mark.parametrize(
    "pd,lgd,ead",
    [
        ("-0.1", "0.5", "100"),
        ("1.1", "0.5", "100"),
        ("0.5", "1.1", "100"),
        ("0.5", "0.5", "-1"),
        ("NaN", "0.5", "100"),
        ("0.5", "Infinity", "100"),
    ],
)
def test_invalid_numbers(pd: str, lgd: str, ead: str) -> None:
    with pytest.raises(ValidationError):
        scenario(pd, lgd, ead)


def test_horizon_and_exposure_limits() -> None:
    data = scenario().model_dump()
    data["ead"]["horizon"] = "lifetime"
    with pytest.raises(ValidationError):
        ExpectedLossRequest.model_validate(data)
    data = scenario().model_dump()
    data["ead"]["exposure_limit"] = "999"
    with pytest.raises(ValidationError):
        ExpectedLossRequest.model_validate(data)


def test_rounding_and_recovery() -> None:
    assert quantize_money(Decimal("1.005"), "INR") == Decimal("1.00")
    assert quantize_money(Decimal("1.015"), "INR") == Decimal("1.02")
    exposure = scenario().ead
    assert collateral_recovery(exposure, Decimal("0.2"), Decimal("10")) == 790
    assert collateral_lgd(exposure, exposure, Decimal("0.2")) == Decimal("0.2")
    assert loan_to_value(exposure, exposure) == 1
    foreign = exposure.model_copy(update={"currency": "USD"})
    with pytest.raises(ValueError):
        loan_to_value(exposure, foreign)
    with pytest.raises(ValueError):
        loan_to_value(exposure, exposure.model_copy(update={"amount": Decimal(0)}))
    with pytest.raises(ValueError):
        collateral_recovery(exposure, Decimal("1.1"))
    data = scenario().model_dump()
    data["recovery_rate"] = data.pop("lgd")
    assert expected_loss(ExpectedLossRequest.model_validate(data)).expected_loss == 100


def test_portfolio_compatibility() -> None:
    result = expected_loss(scenario())
    assert aggregate_scenarios([result, result])["expected_loss"] == "200.00"
    with pytest.raises(ValueError):
        aggregate_scenarios([result, result.model_copy(update={"currency": "USD"})])


@given(pd=st.integers(0, 1000), lgd=st.integers(0, 1000), ead=st.integers(0, 1_000_000))
def test_loss_bounds_and_monotonicity(pd: int, lgd: int, ead: int) -> None:
    request = scenario(str(Decimal(pd) / 1000), str(Decimal(lgd) / 1000), str(ead))
    result = expected_loss(request)
    assert 0 <= result.unrounded_expected_loss <= ead
    higher = expected_loss(scenario("1", str(Decimal(lgd) / 1000), str(ead)))
    assert higher.unrounded_expected_loss >= result.unrounded_expected_loss
