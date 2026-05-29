# filename: tests/test_preprocessing.py
# purpose:  Unit tests for OHE, scaler, and schema validation
# version:  1.0

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    fit_ohe, transform_ohe, fit_scaler, transform_scaler, validate_schema
)


def test_ohe_wilderness_column_count():
    col = pd.Series([1, 2, 3, 4, 1, 2], name="Wilderness_Area")
    enc = fit_ohe(col, "Wilderness_Area")
    result = transform_ohe(enc, col, "Wilderness_Area")
    assert result.shape[1] == 4, f"Expected 4 cols, got {result.shape[1]}"


def test_ohe_uses_get_feature_names_out():
    col = pd.Series([1, 2, 3, 4], name="Wilderness_Area")
    enc = fit_ohe(col, "Wilderness_Area")
    result = transform_ohe(enc, col, "Wilderness_Area")
    # Names must come from get_feature_names_out, not hardcoded
    assert all("Wilderness_Area" in c for c in result.columns)


def test_ohe_handle_unknown_ignores_unseen():
    train_col = pd.Series([1, 2, 3], name="Soil_Type")
    enc = fit_ohe(train_col, "Soil_Type", handle_unknown="ignore")
    # Soil_Type=99 was not in training — should not crash
    test_col = pd.Series([99], name="Soil_Type")
    result = transform_ohe(enc, test_col, "Soil_Type")
    assert result.shape[0] == 1


def test_scaler_output_shape():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    scaler = fit_scaler(X)
    result = transform_scaler(scaler, X, ["a", "b"])
    assert result.shape == X.shape


def test_validate_schema_passes():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    validate_schema(df, ["a", "b"])  # should not raise


def test_validate_schema_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(df, ["a", "b"])
