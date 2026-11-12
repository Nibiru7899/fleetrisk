import numpy as np
import pandas as pd
import polars as pl
import pytest

from fleetrisk.evaluation.metrics import bootstrap_intervals, evaluate
from fleetrisk.features.historical import bureau_as_of, installments_as_of
from fleetrisk.features.registry import feature_frame, preprocessing
from fleetrisk.modeling.training import partition


def test_partition_is_disjoint_and_deterministic() -> None:
    y = np.array([0, 0, 0, 1] * 100)
    first, second = partition(y), partition(y)
    assert set(np.concatenate(list(first.values()))) == set(range(len(y)))
    assert sum(map(len, first.values())) == len(y)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_training_only_imputation_and_leakage_exclusion() -> None:
    data = pd.DataFrame(
        {
            "disbursed_amount": [100.0, 200.0],
            "asset_cost": [200.0, 400.0],
            "ltv": [50.0, 50.0],
            "PERFORM_CNS.SCORE": [400.0, 600.0],
            "Employment.Type": ["a", "b"],
            "loan_default": [0, 1],
            "UniqueID": [1, 2],
        }
    )
    features = feature_frame(data, "ltfs")
    assert "loan_default" not in features and "UniqueID" not in features
    transform = preprocessing("ltfs").fit(features)
    statistics = transform.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    future = data.copy()
    future["asset_cost"] = 10**12
    future["loan_default"] = 1
    transform.transform(feature_frame(future, "ltfs"))
    np.testing.assert_array_equal(
        statistics, transform.named_transformers_["numeric"].named_steps["impute"].statistics_
    )


def test_future_rows_do_not_change_historical_features() -> None:
    past = pl.DataFrame({"SK_ID_CURR": [1, 1], "DAYS_CREDIT": [-200, -1]})
    future = pl.DataFrame({"SK_ID_CURR": [1], "DAYS_CREDIT": [1]})
    assert bureau_as_of(past).equals(bureau_as_of(pl.concat([past, future])))
    payments = pl.DataFrame(
        {
            "SK_ID_CURR": [1, 1],
            "DAYS_INSTALMENT": [-20, 5],
            "DAYS_ENTRY_PAYMENT": [-19, 6],
            "AMT_PAYMENT": [100.0, 10000.0],
        }
    )
    first = installments_as_of(payments)
    changed = installments_as_of(
        payments.with_columns(
            pl.when(pl.col("DAYS_ENTRY_PAYMENT") > 0)
            .then(10**9)
            .otherwise(pl.col("AMT_PAYMENT"))
            .alias("AMT_PAYMENT")
        )
    )
    assert first.equals(changed) and first["historical_payment_amount"].item() == 100


def test_metrics_against_known_values() -> None:
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    result = evaluate(y, p)
    assert result["roc_auc"] == 1 and result["average_precision"] == 1
    assert result["brier"] == pytest.approx(0.025)
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert sum(row["count"] for row in result["risk_deciles"]) == 4
    assert bootstrap_intervals(y, p, 20) == bootstrap_intervals(y, p, 20)


@pytest.mark.parametrize("p", [[np.nan, 0.5], [-0.1, 0.5], [1.1, 0.5]])
def test_invalid_probabilities(p: list[float]) -> None:
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1]), np.array(p))
