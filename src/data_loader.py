# filename: src/data_loader.py
# purpose:  Load raw CSV, validate schema, split data, save processed CSVs
# version:  2.0

import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area", "Soil_Type", "Cover_Type",
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    logger.info("Loaded: %s rows x %s cols", f"{df.shape[0]:,}", df.shape[1])
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_col = df[target_col] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    logger.info("Train: %s rows | Test: %s rows", f"{len(train_df):,}", f"{len(test_df):,}")
    return train_df, test_df


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    size_kb = os.path.getsize(path) / 1024
    logger.info("Saved: %s (%.0f KB, %s rows)", path, size_kb, f"{len(df):,}")
