# filename: api/routes/predict.py
# purpose:  POST /predict endpoint — preprocesses 12-field input, returns cover type prediction
# version:  2.0

import os
import asyncio
import json as _json
import logging
import uuid
import numpy as np
from datetime import datetime, timezone
from functools import partial
from fastapi import APIRouter, Request, HTTPException

from api.schemas.input_schema import CoverTypeInput, CoverTypePrediction
from api.limiter import limiter, RATE_LIMIT
from src.predictor import preprocess_input, predict

router      = APIRouter()
logger      = logging.getLogger(__name__)
pred_logger = logging.getLogger("ecotype.predictions")

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
MODEL_VERSION        = os.environ.get("MODEL_VERSION", "unknown")


@router.post("/predict", response_model=CoverTypePrediction)
@limiter.limit(RATE_LIMIT)
async def predict_cover_type(request: Request, input_data: CoverTypeInput):
    artifacts = request.app.state.artifacts
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(artifacts) != 6:
        logger.critical(
            "Artifact count mismatch: expected 6, got %d. request_id=%s",
            len(artifacts),
            request.headers.get("X-Request-ID", "unknown"),
        )
        raise HTTPException(status_code=503, detail="Model artifacts corrupted — contact admin")

    # Named unpacking — explicit, not fragile index access
    model, wld_ohe, soil_ohe, scaler, feature_cols, quant_cols = artifacts

    class_map = request.app.state.class_map
    data_dict = input_data.model_dump()

    # Run CPU-bound sklearn work in thread pool — keeps event loop free
    loop = asyncio.get_running_loop()
    X = await loop.run_in_executor(None, partial(preprocess_input, data_dict, artifacts))
    class_id, proba = await loop.run_in_executor(None, partial(predict, model, X))

    proba = np.array(proba).flatten()   # ensure 1D shape (7,)

    probabilities = {
        class_map.get(i + 1, f"Class {i+1}"): round(float(p), 4)
        for i, p in enumerate(proba)
    }
    max_confidence = float(np.max(proba))
    low_conf       = max_confidence < CONFIDENCE_THRESHOLD

    request_id = getattr(request.state, 'request_id',
                         request.headers.get("X-Request-ID", "unknown"))

    pred_logger.info(_json.dumps({
        "event":           "prediction",
        "request_id":      request_id,
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
