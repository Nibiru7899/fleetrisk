from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fleetrisk.data.contracts import get_contract

FEATURE_VERSION = "origination-core-v1"
LEAKAGE_COLUMNS = {
    "loan_default",
    "TARGET",
    "UniqueID",
    "SK_ID_CURR",
    "Date.of.Birth",
    "DisbursalDate",
}


def feature_schema(dataset: str) -> dict[str, Any]:
    contract = get_contract(dataset)
    if contract.historical:
        raise ValueError("Historical tables are not an independent supervised population")
    return {
        "version": FEATURE_VERSION,
        "dataset": dataset,
        "numeric": list(contract.numeric),
        "categorical": list(contract.categorical),
        "features": [
            {
                "name": name,
                "source_columns": [name],
                "transformation": "identity-v1",
                "description": f"Source {name}; verify origination availability against dictionary",
                "availability": "provisional_origination_snapshot",
                "missing": "training_fitted_imputation",
            }
            for name in (*contract.numeric, *contract.categorical)
        ],
        "excluded": sorted(LEAKAGE_COLUMNS),
    }


def feature_frame(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    schema = feature_schema(dataset)
    names = schema["numeric"] + schema["categorical"]
    selected = frame.loc[:, names].copy()
    for column in schema["numeric"]:
        selected[column] = pd.to_numeric(selected[column], errors="raise").astype(float)
    for column in schema["categorical"]:
        selected[column] = selected[column].map(
            lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else np.nan
        )
    return selected


def preprocessing(dataset: str) -> ColumnTransformer:
    schema = feature_schema(dataset)
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    category = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, schema["numeric"]), ("categorical", category, schema["categorical"])]
    )
