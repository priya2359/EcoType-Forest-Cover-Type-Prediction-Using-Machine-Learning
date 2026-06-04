# filename: api/main.py
# purpose:  FastAPI entry point with lifespan model loading and HuggingFace download
# version:  2.0

import os
import logging
import json as _json
import traceback as _traceback
import yaml
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# CRITICAL: import XGBWrapper BEFORE any joblib.load()
from src.xgb_wrapper import XGBWrapper  # noqa

from src.predictor import load_artifacts, FeatureEngineeringError
from api.limiter import limiter
from api.routes import health, predict
from api.middleware.logging_middleware import LoggingMiddleware


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if record.exc_info:
            log_dict["exception"] = _traceback.format_exception(*record.exc_info)
        return _json.dumps(log_dict)


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    handlers=[_handler],
    force=True,
)
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
        logging.error("STARTUP FAILED — model not loaded: %s: %s", type(e).__name__, e)
        logging.error(
            "Check: HF_REPO_ID=%s | HF_TOKEN set=%s | MODEL_SOURCE=%s",
            os.environ.get("HF_REPO_ID", "NOT SET"),
            bool(os.environ.get("HF_TOKEN")),
            os.environ.get("MODEL_SOURCE", "local"),
        )
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


@app.exception_handler(FeatureEngineeringError)
async def fe_error_handler(request: Request, exc: FeatureEngineeringError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})

# Rate limiting — single instance shared with routes
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — explicit origins for prod; set ALLOWED_ORIGINS env var on Render once Streamlit URL known
_raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]
if "*" in ALLOWED_ORIGINS:
    logging.warning("SECURITY: CORS allow_origins=['*'] — set ALLOWED_ORIGINS in production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials="*" not in ALLOWED_ORIGINS,  # credentials=True illegal with wildcard
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)
app.add_middleware(LoggingMiddleware)

app.include_router(health.router)
app.include_router(predict.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=PORT, reload=True)
