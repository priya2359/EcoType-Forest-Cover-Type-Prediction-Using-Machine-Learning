# filename: tests/test_api.py
# purpose:  FastAPI endpoint tests using TestClient + mock_artifacts from conftest.py
# version:  1.0

# Golden Rule 4: XGBWrapper before any joblib.load or api.main import
from src.xgb_wrapper import XGBWrapper  # noqa

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "Elevation": 2596,
    "Aspect": 51,
    "Slope": 3,
    "Horizontal_Distance_To_Hydrology": 258,
    "Vertical_Distance_To_Hydrology": 0,
    "Horizontal_Distance_To_Roadways": 510,
    "Hillshade_9am": 221,
    "Hillshade_Noon": 232,
    "Hillshade_3pm": 148,
    "Horizontal_Distance_To_Fire_Points": 6279,
    "Wilderness_Area": 1,
    "Soil_Type": 29,
}


def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_valid_input(mock_artifacts):
    with patch.object(app.state, "artifacts", mock_artifacts, create=True), \
         patch.object(app.state, "class_map", {1: "Spruce/Fir", 2: "Lodgepole Pine",
                                                3: "Ponderosa Pine", 4: "Cottonwood/Willow",
                                                5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"},
                      create=True):
        resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert "cover_type_id" in data
    assert "cover_type_name" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert 1 <= data["cover_type_id"] <= 7
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_missing_field_returns_422():
    payload = VALID_PAYLOAD.copy()
    del payload["Elevation"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_out_of_range_returns_422():
    payload = {**VALID_PAYLOAD, "Soil_Type": 41}  # max is 40
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_wilderness_out_of_range_returns_422():
    payload = {**VALID_PAYLOAD, "Wilderness_Area": 5}  # max is 4
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_503_when_no_model():
    with patch.object(app.state, "artifacts", None, create=True):
        resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"].lower()
