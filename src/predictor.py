# filename: src/predictor.py
# purpose:  Inference pipeline — exact order must match training
# version:  1.0

import os
import joblib
import numpy as np
import pandas as pd
import pathlib

# CRITICAL: import XGBWrapper BEFORE any joblib.load()
# Pickle needs the class importable at deserialization time
from src.xgb_wrapper import XGBWrapper  # noqa

from src.feature_engineering import engineer_features
from src.preprocessing import transform_ohe, transform_scaler
from src.feature_selector import load_feature_columns

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model/ecotype_best_model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scalers/standard_scaler.pkl")
WILDERNESS_OHE_PATH = os.environ.get("WILDERNESS_OHE_PATH", "models/encoders/wilderness_ohe.pkl")
SOIL_OHE_PATH = os.environ.get("SOIL_OHE_PATH", "models/encoders/soil_ohe.pkl")
FEATURE_COLUMNS_PATH = os.environ.get("FEATURE_COLUMNS_PATH", "artifacts/feature_columns.txt")
QUANT_FEATURE_COLUMNS_PATH = os.environ.get("QUANT_FEATURE_COLUMNS_PATH", "artifacts/quant_feature_columns.txt")


def _download_from_hf() -> None:
    from huggingface_hub import hf_hub_download
    repo_id = os.environ["HF_REPO_ID"]
    token = os.environ.get("HF_TOKEN")
    files = [MODEL_PATH, SCALER_PATH, WILDERNESS_OHE_PATH, SOIL_OHE_PATH,
             FEATURE_COLUMNS_PATH, QUANT_FEATURE_COLUMNS_PATH]
    for f in files:
        pathlib.Path(f).parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename=f, local_dir=".",
                        token=token, force_download=False)


def load_artifacts() -> tuple:
    if os.environ.get("MODEL_SOURCE") == "huggingface":
        _download_from_hf()
    model = joblib.load(MODEL_PATH)
    wilderness_ohe = joblib.load(WILDERNESS_OHE_PATH)
    soil_ohe = joblib.load(SOIL_OHE_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = load_feature_columns(FEATURE_COLUMNS_PATH)
    quant_cols = load_feature_columns(QUANT_FEATURE_COLUMNS_PATH)
    return model, wilderness_ohe, soil_ohe, scaler, feature_cols, quant_cols


def preprocess_input(data: dict, artifacts: tuple) -> np.ndarray:
    model, wilderness_ohe, soil_ohe, scaler, feature_cols, quant_cols = artifacts

    # Step 1: arithmetic feature engineering
    df = pd.DataFrame([data])
    df = engineer_features(df)

    # Step 2: OHE (uses get_feature_names_out internally)
    wild_enc = transform_ohe(wilderness_ohe, df["Wilderness_Area"], "Wilderness_Area")
    soil_enc = transform_ohe(soil_ohe, df["Soil_Type"], "Soil_Type")
    df = df.drop(columns=["Wilderness_Area", "Soil_Type"])
    df = pd.concat([df.reset_index(drop=True), wild_enc.reset_index(drop=True),
                    soil_enc.reset_index(drop=True)], axis=1)

    # Step 3: scale quantitative cols only
    present_quant = [c for c in quant_cols if c in df.columns]
    scaled = scaler.transform(df[present_quant])
    df[present_quant] = scaled

    # Step 4: align to exact training column order (loud error if mismatch)
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns at inference: {missing}")
    df = df.reindex(columns=feature_cols, fill_value=0)

    return df.values


def predict(model, X: np.ndarray) -> tuple[int, np.ndarray]:
    proba = model.predict_proba(X)[0]
    class_id = int(model.predict(X)[0])
    return class_id, proba
