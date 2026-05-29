# filename: src/feature_selector.py
# purpose:  Feature importance-based selection and artifact persistence
# version:  2.0

import logging
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def select_by_importance(
    X_df: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.0005,
    n_estimators: int = 100,
    random_state: int = 42,
) -> list[str]:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_df, y)
    importances = pd.Series(rf.feature_importances_, index=X_df.columns)
    selected = importances[importances >= threshold].index.tolist()
    dropped = importances[importances < threshold].index.tolist()
    logger.info("Selected %d features (dropped %d below %s)", len(selected), len(dropped), threshold)
    return selected, importances


def save_selected_features(features: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(features))
    logger.info("Saved selected features: %s (%d features)", path, len(features))


def save_feature_columns(X_selected_df: pd.DataFrame, path: str) -> None:
    """Save EXACT column order for inference alignment."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(X_selected_df.columns.tolist()))
    logger.info("Saved feature columns: %s (%d columns)", path, len(X_selected_df.columns))


def save_quant_feature_columns(quant_cols: list[str], path: str) -> None:
    """Save quantitative column names for scaler at inference."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(quant_cols))
    logger.info("Saved quant feature columns: %s (%d columns)", path, len(quant_cols))


def load_feature_columns(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def apply_selection(X_df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    missing = set(feature_list) - set(X_df.columns)
    if missing:
        raise ValueError(f"Cannot apply selection — missing columns: {missing}")
    return X_df[feature_list]
