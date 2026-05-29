# filename: api/routes/predict.py
# purpose:  POST /predict endpoint — preprocesses 12-field input, returns cover type prediction
# version:  1.0

import json
import numpy as np
from fastapi import APIRouter, Request, HTTPException

from api.schemas.input_schema import CoverTypeInput, CoverTypePrediction
from src.predictor import preprocess_input, predict

router = APIRouter()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)   # BEFORE np.integer — np.bool_ subclasses np.integer in numpy < 2.0
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


@router.post("/predict", response_model=CoverTypePrediction)
def predict_cover_type(input_data: CoverTypeInput, request: Request):
    artifacts = request.app.state.artifacts
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    class_map = request.app.state.class_map
    data_dict = input_data.model_dump()

    X = preprocess_input(data_dict, artifacts)
    class_id, proba = predict(artifacts[0], X)

    probabilities = {
        class_map.get(i + 1, f"Class {i+1}"): float(p)
        for i, p in enumerate(proba)
    }

    return CoverTypePrediction(
        cover_type_id=int(class_id),
        cover_type_name=class_map.get(int(class_id), f"Class {class_id}"),
        confidence=float(max(proba)),
        probabilities=probabilities,
    )
