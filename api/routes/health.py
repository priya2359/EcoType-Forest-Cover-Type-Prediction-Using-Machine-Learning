# filename: api/routes/health.py
# purpose:  GET /health endpoint — always returns 200; never raises
# version:  1.0

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health_check(request: Request):
    state = request.app.state
    model_loaded = hasattr(state, "artifacts") and state.artifacts is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "version": "1.0.0",
    }
