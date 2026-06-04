# filename: tests/test_feature_engineering.py
# purpose:  Unit tests for feature engineering — idempotency, determinism, edge cases

import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import engineer_features, ENGINEERED_COLS

BASE_ROW = {
    "Elevation": 2596, "Aspect": 90, "Slope": 10,
    "Horizontal_Distance_To_Hydrology": 300,
    "Vertical_Distance_To_Hydrology": 50,
    "Horizontal_Distance_To_Roadways": 500,
    "Hillshade_9am": 200, "Hillshade_Noon": 230, "Hillshade_3pm": 150,
    "Horizontal_Distance_To_Fire_Points": 6000,
    "Wilderness_Area": 1, "Soil_Type": 10,
}


def test_aspect_dropped_and_decomposed():
    df = engineer_features(pd.DataFrame([BASE_ROW]))
    assert "Aspect" not in df.columns
    assert "Aspect_sin" in df.columns and "Aspect_cos" in df.columns
    # Aspect=90deg -> sin(deg2rad(90)) = 1.0, cos(deg2rad(90)) = 0.0
    assert abs(df["Aspect_sin"].iloc[0] - 1.0) < 1e-6
    assert abs(df["Aspect_cos"].iloc[0] - 0.0) < 1e-6


def test_idempotent_checks_values():
    """Second call must not re-apply transforms — checks VALUES not just column names."""
    df1 = engineer_features(pd.DataFrame([BASE_ROW]))
    df2 = engineer_features(df1.copy())
    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True),
        df2.reset_index(drop=True),
        check_exact=False, rtol=1e-5,
    )


def test_deterministic():
    """Same input always produces same output (training-serving consistency)."""
    df1 = engineer_features(pd.DataFrame([BASE_ROW]))
    df2 = engineer_features(pd.DataFrame([BASE_ROW]))
    pd.testing.assert_frame_equal(df1, df2)


def test_no_infinity_values():
    """+1 guard in Distance_Road_Fire_Ratio prevents infinity."""
    row = {**BASE_ROW, "Horizontal_Distance_To_Fire_Points": 0}
    df = engineer_features(pd.DataFrame([row]))
    numeric_df = df.select_dtypes(include=[np.number])
    inf_cols = numeric_df.columns[np.isinf(numeric_df.values).any(axis=0)].tolist()
    assert not inf_cols, f"Infinity values in: {inf_cols}"


def test_road_fire_ratio_zero_denominator():
    """Ratio is finite and positive when Fire_Points == 0."""
    row = {**BASE_ROW, "Horizontal_Distance_To_Fire_Points": 0}
    df = engineer_features(pd.DataFrame([row]))
    ratio = df["Distance_Road_Fire_Ratio"].iloc[0]
    assert np.isfinite(ratio), f"Expected finite ratio, got {ratio}"
    assert ratio > 0, "Expected positive ratio"


def test_hydro_distance_combined():
    df = engineer_features(pd.DataFrame([BASE_ROW]))
    expected = np.sqrt(300**2 + 50**2)
    assert abs(df["Hydro_Distance_Combined"].iloc[0] - expected) < 1e-6


def test_negative_vertical_hydrology():
    """Vertical distance can be negative — Euclidean formula still valid."""
    row = {**BASE_ROW, "Vertical_Distance_To_Hydrology": -50}
    df = engineer_features(pd.DataFrame([row]))
    expected = np.sqrt(300**2 + 50**2)   # same as positive — v is squared
    assert abs(df["Hydro_Distance_Combined"].iloc[0] - expected) < 1e-6
    assert df.isnull().sum().sum() == 0


def test_all_engineered_cols_present():
    df = engineer_features(pd.DataFrame([BASE_ROW]))
    for col in ENGINEERED_COLS:
        assert col in df.columns, f"Missing engineered column: {col}"


def test_no_nan_output():
    df = engineer_features(pd.DataFrame([BASE_ROW]))
    assert df.isnull().sum().sum() == 0
