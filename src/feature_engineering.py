# filename: src/feature_engineering.py
# purpose:  Add 7 engineered features: Aspect sin/cos + 5 domain arithmetic features
# version:  2.0

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Circular decomposition of Aspect — preserves compass-bearing distance correctly.
    # Raw Aspect is dropped: 1° and 359° are 2° apart but |359-1|=358 as integers.
    # Guard makes the function idempotent: safe to call on already-processed data.
    if "Aspect" in df.columns:
        aspect_rad = np.deg2rad(df["Aspect"])
        df["Aspect_sin"] = np.sin(aspect_rad)
        df["Aspect_cos"] = np.cos(aspect_rad)
        df = df.drop(columns=["Aspect"])

    df["Hydro_Distance_Combined"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    df["Hydro_Elev_interaction"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    df["Hillshade_mean"] = (
        df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
    ) / 3

    df["Elevation_Slope_interaction"] = df["Elevation"] * df["Slope"]

    df["Distance_Road_Fire_Ratio"] = df["Horizontal_Distance_To_Roadways"] / (
        df["Horizontal_Distance_To_Fire_Points"] + 1
    )

    return df


ENGINEERED_COLS = [
    "Aspect_sin",
    "Aspect_cos",
    "Hydro_Distance_Combined",
    "Hydro_Elev_interaction",
    "Hillshade_mean",
    "Elevation_Slope_interaction",
    "Distance_Road_Fire_Ratio",
]
