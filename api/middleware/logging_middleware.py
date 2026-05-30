# filename: api/middleware/logging_middleware.py
# purpose:  Log request method, path, status code, and latency per request (structured JSON)
# version:  2.0

import json
import uuid
import time
import logging
from datetime import datetime, timezone
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("ecotype.api")

# Skip logging for health/static paths — prevents log spam from Render health checks
SKIP_LOG_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in SKIP_LOG_PATHS:
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        try:
            response    = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.error(json.dumps({
                "request_id": request_id,
                "method":     request.method,
                "path":       request.url.path,
                "status_code": 500,
                "latency_ms": latency_ms,
                "error":      str(exc),
                "error_type": type(exc).__name__,
                "ts":         datetime.now(timezone.utc).isoformat(),
            }))
            raise  # re-raise so FastAPI error handlers still work

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(json.dumps({
            "request_id":  request_id,
            "method":      request.method,
            "path":        request.url.path,
            "status_code": status_code,
            "latency_ms":  latency_ms,
            "ts":          datetime.now(timezone.utc).isoformat(),
        }))
        response.headers["X-Request-ID"] = request_id
        return response
