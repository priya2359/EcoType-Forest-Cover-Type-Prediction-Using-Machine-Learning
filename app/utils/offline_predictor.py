# filename: app/utils/offline_predictor.py
# purpose:  Fallback predictor when FastAPI is unavailable; returns None gracefully
# version:  1.0

import os
import yaml


def _load_class_map() -> dict:
    try:
        with open("configs/feature_config.yaml") as f:
            config = yaml.safe_load(f)
        return {int(k): v for k, v in config["class_map"].items()}
    except Exception:
        return {1:"Spruce/Fir", 2:"Lodgepole Pine", 3:"Ponderosa Pine",
                4:"Cottonwood/Willow", 5:"Aspen", 6:"Douglas-fir", 7:"Krummholz"}


def predict_directly(input_data: dict) -> dict | None:
    """
    Works in local/Docker deployment where model PKLs exist on disk.
    Returns None gracefully on Streamlit Cloud (no PKLs available there).
    """
    try:
        from src.predictor import load_artifacts, preprocess_input, predict
        artifacts = load_artifacts()
        X = preprocess_input(input_data, artifacts)
        class_id, proba = predict(artifacts[0], X)
        cover_type_map = _load_class_map()
        return {
            "cover_type_id": int(class_id),
            "cover_type_name": cover_type_map.get(class_id, f"Class {class_id}"),
            "confidence": float(max(proba)),
            "probabilities": {
                cover_type_map.get(i + 1, f"Class {i+1}"): float(p)
                for i, p in enumerate(proba)
            },
        }
    except FileNotFoundError:
        return None
    except Exception:
        return None
