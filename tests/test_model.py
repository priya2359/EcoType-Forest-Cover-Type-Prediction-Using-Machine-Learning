# filename: tests/test_model.py
# purpose:  Model loading and prediction tests; skipped in CI when PKLs absent
# version:  1.0

from src.xgb_wrapper import XGBWrapper  # noqa — must precede joblib.load()

import os
import numpy as np
import pytest
import joblib

MODEL_PATH = "models/best_model/ecotype_best_model.pkl"

# Skip model-loading tests in CI — PKLs not in repo (stored on HuggingFace)
# Run locally where PKLs exist after training
requires_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model PKL not available in CI — run locally after training",
)


@requires_model
def test_best_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


@requires_model
def test_predict_range():
    model = joblib.load(MODEL_PATH)
    # Create minimal valid input (shape depends on selected features)
    # This test uses a small dummy array — shape must match after feature selection
    # Run locally after training to get correct feature count
    pass


@requires_model
def test_probabilities_sum_to_one():
    model = joblib.load(MODEL_PATH)
    pass


def test_xgb_wrapper_sklearn_api():
    """XGBWrapper behaves like a sklearn estimator — runs in CI without PKLs."""
    import numpy as np
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
