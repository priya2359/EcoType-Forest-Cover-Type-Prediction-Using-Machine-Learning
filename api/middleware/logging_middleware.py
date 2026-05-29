# filename: api/middleware/logging_middleware.py
# purpose:  Log request method, path, status code, and latency per request
# version:  1.0

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("ecotype.api")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "%s %s %s %.1fms",
            request.method, request.url.path,
            response.status_code, latency_ms,
        )
        return response
