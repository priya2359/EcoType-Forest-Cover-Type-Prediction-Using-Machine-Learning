# filename: src/preprocessing.py
# purpose:  OHE encoding, standard scaling, schema validation
# version:  2.0

import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def separate_features_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def fit_ohe(
    X_col: pd.Series, col_name: str, handle_unknown: str = "ignore"
) -> OneHotEncoder:
    encoder = OneHotEncoder(
        handle_unknown=handle_unknown,
        sparse_output=False,
    )
    encoder.fit(X_col.values.reshape(-1, 1))
    return encoder


def transform_ohe(
    encoder: OneHotEncoder, X_col: pd.Series, col_name: str
) -> pd.DataFrame:
    # Always use get_feature_names_out — never hardcode column names
    col_names = encoder.get_feature_names_out([col_name])
    transformed = encoder.transform(X_col.values.reshape(-1, 1))
    return pd.DataFrame(transformed, columns=col_names, index=X_col.index)


def fit_scaler(X_quant_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_quant_df)
    return scaler


def transform_scaler(
    scaler: StandardScaler,
    X_quant_df: pd.DataFrame,
    quant_cols: list[str],
) -> pd.DataFrame:
    scaled = scaler.transform(X_quant_df[quant_cols])
    return pd.DataFrame(scaled, columns=quant_cols, index=X_quant_df.index)


def validate_schema(df: pd.DataFrame, expected_cols: list[str]) -> None:
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed — missing columns: {missing}")


def save_encoder(encoder: OneHotEncoder, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoder, path)
    logger.info("Saved encoder: %s", path)


def save_scaler(scaler: StandardScaler, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Saved scaler: %s", path)
