# filename: api/main.py
# purpose:  FastAPI entry point with lifespan model loading and HuggingFace download
# version:  1.0

import os
import pathlib
import logging
import yaml
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# CRITICAL: import XGBWrapper BEFORE any joblib.load()
from src.xgb_wrapper import XGBWrapper  # noqa

from src.predictor import load_artifacts
from api.routes import health, predict
from api.middleware.logging_middleware import LoggingMiddleware

logging.basicConfig(level=logging.INFO)
PORT = int(os.environ.get("PORT", 8000))


def _load_class_map() -> dict:
    with open("configs/feature_config.yaml") as f:
        config = yaml.safe_load(f)
    return {int(k): v for k, v in config["class_map"].items()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.artifacts = load_artifacts()
        app.state.class_map = _load_class_map()
        logging.info("Model artifacts loaded successfully.")
    except Exception as e:
        logging.warning(f"Failed to load artifacts: {e}. /predict will return 503.")
        app.state.artifacts = None
        app.state.class_map = {}
    yield
    app.state.artifacts = None


app = FastAPI(
    title="EcoType API",
    version="1.0.0",
    description="Forest Cover Type Prediction — 12-field input, 7-class output",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

app.include_router(health.router)
app.include_router(predict.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=PORT, reload=True)
