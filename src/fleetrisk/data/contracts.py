"""Provisional source contracts. Actual-source audit is a separate empirical gate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Contract:
    name: str
    key: tuple[str, ...]
    required: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...] = ()
    target: str | None = None
    historical: bool = False
    version: str = "1"


CONTRACTS = {
    "ltfs": Contract(
        "ltfs",
        ("UniqueID",),
        (
            "UniqueID",
            "disbursed_amount",
            "asset_cost",
            "ltv",
            "PERFORM_CNS.SCORE",
            "Employment.Type",
        ),
        ("disbursed_amount", "asset_cost", "ltv", "PERFORM_CNS.SCORE"),
        ("Employment.Type",),
        "loan_default",
    ),
    "home_credit_application": Contract(
        "home_credit_application",
        ("SK_ID_CURR",),
        ("SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "NAME_CONTRACT_TYPE"),
        ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        ("NAME_CONTRACT_TYPE",),
        "TARGET",
    ),
    "home_credit_bureau": Contract(
        "home_credit_bureau",
        ("SK_ID_BUREAU",),
        ("SK_ID_BUREAU", "SK_ID_CURR", "DAYS_CREDIT", "AMT_CREDIT_SUM"),
        ("DAYS_CREDIT", "AMT_CREDIT_SUM"),
        historical=True,
    ),
    "home_credit_previous": Contract(
        "home_credit_previous",
        ("SK_ID_PREV",),
        ("SK_ID_PREV", "SK_ID_CURR", "DAYS_DECISION", "AMT_CREDIT"),
        ("DAYS_DECISION", "AMT_CREDIT"),
        historical=True,
    ),
    "home_credit_installments": Contract(
        "home_credit_installments",
        ("SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_ENTRY_PAYMENT"),
        (
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ),
        ("DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"),
        historical=True,
    ),
}


def get_contract(dataset: str) -> Contract:
    if dataset not in CONTRACTS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return CONTRACTS[dataset]
