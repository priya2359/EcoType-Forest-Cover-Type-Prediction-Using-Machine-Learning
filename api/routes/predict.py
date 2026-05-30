# filename: api/routes/predict.py
# purpose:  POST /predict endpoint — preprocesses 12-field input, returns cover type prediction
# version:  2.0

import os
import json
import logging
import numpy as np
from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException

from api.schemas.input_schema import CoverTypeInput, CoverTypePrediction
from api.limiter import limiter, RATE_LIMIT   # single instance — no circular import
from src.predictor import preprocess_input, predict

router      = APIRouter()
pred_logger = logging.getLogger("ecotype.predictions")

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
MODEL_VERSION        = os.environ.get("MODEL_VERSION", "unknown")


@router.post("/predict", response_model=CoverTypePrediction)
@limiter.limit(RATE_LIMIT)
def predict_cover_type(request: Request, input_data: CoverTypeInput):
    artifacts = request.app.state.artifacts
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Runtime assertion 1: verify artifact count BEFORE unpacking
    assert len(artifacts) == 6, (
        f"Expected 6 artifacts (model, wld_ohe, soil_ohe, scaler, cols, quant), "
        f"got {len(artifacts)}. Check load_artifacts() return signature."
    )

    # Named unpacking — explicit, not fragile index access
    model, wld_ohe, soil_ohe, scaler, feature_cols, quant_cols = artifacts

    class_map = request.app.state.class_map
    data_dict = input_data.model_dump()

    X = preprocess_input(data_dict, artifacts)
    class_id, proba = predict(model, X)   # predict() already returns (int, ndarray)

    # Runtime assertions 2 & 3: verify predict() return types immediately after call
    assert isinstance(class_id, (int, np.integer)), (
        f"predict() must return int class_id, got {type(class_id)}"
    )
    assert isinstance(proba, (np.ndarray, list)), (
        f"predict() must return array-like proba, got {type(proba)}"
    )

    proba = np.array(proba).flatten()   # ensure 1D shape (7,)

    probabilities = {
        class_map.get(i + 1, f"Class {i+1}"): round(float(p), 4)
        for i, p in enumerate(proba)
    }
    max_confidence = float(np.max(proba))
    low_conf       = max_confidence < CONFIDENCE_THRESHOLD

    pred_logger.info(json.dumps({
        "event":           "prediction",
        "input_features":  data_dict,
        "cover_type_id":   int(class_id),
        "cover_type_name": class_map.get(int(class_id), f"Class {class_id}"),
        "confidence":      round(max_confidence, 4),
        "low_confidence":  low_conf,
        "model_version":   MODEL_VERSION,
        "ts":              datetime.now(timezone.utc).isoformat(),
    }))

    return CoverTypePrediction(
        cover_type_id=int(class_id),
        cover_type_name=class_map.get(int(class_id), f"Class {class_id}"),
        confidence=max_confidence,
        probabilities=probabilities,
        low_confidence=low_conf,
        warning="Confidence below threshold — treat with caution" if low_conf else None,
        model_version=MODEL_VERSION,
    )
