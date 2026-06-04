# filename: app/utils/offline_predictor.py
# purpose:  Fallback predictor when FastAPI is unavailable; returns None gracefully
# version:  2.0

import logging
import functools
import numpy as np
from pathlib import Path
import yaml

logger = logging.getLogger("ecotype.offline_predictor")


@functools.lru_cache(maxsize=1)
def _load_class_map() -> dict:
    config_path = Path(__file__).parent.parent.parent / "configs" / "feature_config.yaml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return {int(k): v for k, v in config["class_map"].items()}
    except Exception:
        return {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
                4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}


@functools.lru_cache(maxsize=1)
def _get_cached_artifacts():
    """Load and cache model artifacts once. Do NOT mutate returned objects."""
    from src.predictor import load_artifacts
    return load_artifacts()


def predict_directly(input_data: dict) -> dict | None:
    """
    Works in local/Docker deployment where model PKLs exist on disk.
    Returns None gracefully on Streamlit Cloud (no PKLs available there).
    """
    try:
        artifacts = _get_cached_artifacts()
        model, wld_ohe, soil_ohe, scaler, feature_cols, quant_cols = artifacts
        from src.predictor import preprocess_input, predict
        X = preprocess_input(input_data, artifacts)
        class_id, proba = predict(model, X)
        cover_type_map = _load_class_map()
        return {
            "cover_type_id": int(class_id),
            "cover_type_name": cover_type_map.get(class_id, f"Class {class_id}"),
            "confidence": float(np.max(proba)),
            "probabilities": {
                cover_type_map.get(i + 1, f"Class {i+1}"): float(p)
                for i, p in enumerate(proba)
            },
        }
    except FileNotFoundError:
        _get_cached_artifacts.cache_clear()
        return None
    except Exception as exc:
        _get_cached_artifacts.cache_clear()   # prevent caching failure state
        logger.warning(
            "offline_predictor failed: %s: %s", type(exc).__name__, exc,
            exc_info=True,
        )
        return None
