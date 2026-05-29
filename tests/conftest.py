# filename: tests/conftest.py
# purpose:  Shared pytest fixtures — mock artifacts so API starts without PKLs in CI
# version:  1.0

from src.xgb_wrapper import XGBWrapper  # noqa — must precede any api.main import

import pytest
from unittest.mock import MagicMock
import numpy as np


@pytest.fixture
def mock_artifacts():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array(
        [[0.1, 0.7, 0.05, 0.05, 0.03, 0.04, 0.03]]
    )
    mock_model.predict.return_value = np.array([2])
    mock_model.classes_ = np.array([1, 2, 3, 4, 5, 6, 7])

    mock_wld_ohe = MagicMock()
    mock_wld_ohe.get_feature_names_out.return_value = [
        "Wilderness_Area_1", "Wilderness_Area_2",
        "Wilderness_Area_3", "Wilderness_Area_4",
    ]
    mock_wld_ohe.transform.return_value = np.zeros((1, 4))

    mock_soil_ohe = MagicMock()
    mock_soil_ohe.get_feature_names_out.return_value = [
        "Soil_Type_1", "Soil_Type_2", "Soil_Type_3", "Soil_Type_4",
    ]
    mock_soil_ohe.transform.return_value = np.zeros((1, 4))

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 2))

    # "Aspect" is dropped by engineer_features (replaced by Aspect_sin/Aspect_cos)
    feature_cols = ["Elevation", "Slope"]
    quant_cols = ["Elevation", "Slope"]

    return (mock_model, mock_wld_ohe, mock_soil_ohe, mock_scaler, feature_cols, quant_cols)
