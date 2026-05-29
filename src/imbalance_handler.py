# filename: src/imbalance_handler.py
# purpose:  Class distribution analysis and SMOTE for minority classes
# version:  2.0

import logging
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

logger = logging.getLogger(__name__)


def get_class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts().sort_index()
    total = len(y)
    return {
        int(cls): {"count": int(cnt), "pct": round(cnt / total * 100, 2)}
        for cls, cnt in counts.items()
    }


def get_class_weights(y: pd.Series) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def apply_smote(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: dict,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE with an explicit per-class target dict.

    sampling_strategy must be a dict {class_label: target_count} for only the
    classes that need oversampling. Classes not listed are left untouched.
    Never pass 'auto' or 'minority' — those oversample to match the majority.
    """
    if not sampling_strategy:
        logger.info("Empty sampling_strategy — skipping SMOTE.")
        return X_train_df, y_train

    logger.info("Applying SMOTE: %s", sampling_strategy)
    col_names = X_train_df.columns.tolist()

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    X_res, y_res = smote.fit_resample(X_train_df.values, y_train.values)

    # Preserve column names — SMOTE returns numpy arrays
    X_res_df = pd.DataFrame(X_res, columns=col_names)
    y_res_series = pd.Series(y_res, name=y_train.name)

    logger.info("After SMOTE: %s rows (was %s)", f"{len(X_res_df):,}", f"{len(X_train_df):,}")
    return X_res_df, y_res_series


def save_resampled(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    x_path: str,
    y_path: str,
) -> None:
    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    X_df.to_csv(x_path, index=False)
    y_series.to_csv(y_path, index=False)
    logger.info("Saved resampled X: %s", x_path)
    logger.info("Saved resampled y: %s", y_path)
