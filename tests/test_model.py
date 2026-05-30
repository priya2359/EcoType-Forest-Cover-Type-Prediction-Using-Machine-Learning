# filename: tests/test_model.py
# purpose:  Model loading and prediction tests; skipped in CI when PKLs absent
# version:  2.0

from src.xgb_wrapper import XGBWrapper  # noqa — module-level, must precede any joblib.load

import os
import numpy as np
import pytest
import joblib
from pathlib import Path

MODEL_PATH = "models/best_model/ecotype_best_model.pkl"

# Skip model-loading tests in CI — PKLs not in repo (stored on HuggingFace)
# Run locally where PKLs exist after training
requires_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model PKL not available in CI — run locally after training",
)


@pytest.fixture
def feature_count():
    """Load number of features from artifact file. Handles trailing newlines correctly."""
    path = Path(__file__).parent.parent / "artifacts" / "feature_columns.txt"
    assert path.exists(), f"Feature columns not found: {path}"
    with open(path) as f:
        cols = [line.strip() for line in f if line.strip()]
    assert len(cols) > 0, "feature_columns.txt is empty"
    return len(cols)


@requires_model
def test_best_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


@requires_model
def test_predict_range(feature_count):
    model = joblib.load(MODEL_PATH)
    X_dummy = np.zeros((5, feature_count))
    preds = model.predict(X_dummy)
    assert len(preds) == 5
    assert all(1 <= int(p) <= 7 for p in preds), f"Out-of-range predictions: {preds}"


@requires_model
def test_probabilities_sum_to_one(feature_count):
    model = joblib.load(MODEL_PATH)
    X_dummy = np.zeros((5, feature_count))
    proba = model.predict_proba(X_dummy)
    assert proba.shape == (5, 7), f"Expected shape (5, 7), got {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities don't sum to 1"


def test_xgb_wrapper_sklearn_api():
    """XGBWrapper behaves like a sklearn estimator — runs in CI without PKLs."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=10, n_classes=3,
                               n_informative=5, random_state=42)
    y = y + 1  # shift to 1-3 like Cover_Type

    wrapper = XGBWrapper(n_estimators=10, random_state=42, eval_metric="mlogloss")
    wrapper.fit(X, y)
    preds = wrapper.predict(X)
    assert all(p in [1, 2, 3] for p in preds), "Predictions must be in label range"

    proba = wrapper.predict_proba(X)
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
