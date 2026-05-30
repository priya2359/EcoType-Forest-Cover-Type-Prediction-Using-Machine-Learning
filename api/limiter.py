# filename: api/limiter.py
# purpose:  Single Limiter instance — imported by api/main.py and api/routes/predict.py.
#           Never instantiate Limiter in both places; that creates two independent counters.
import os
from slowapi import Limiter
from slowapi.util import get_remote_address

# NOTE: Default storage is in-memory per-process (no Redis).
# On Render free tier (single worker), rate limits are effectively global.
# Multi-worker deployments need Redis: Limiter(..., storage_uri="redis://...")
RATE_LIMIT = os.environ.get("RATE_LIMIT", "60/minute")

limiter = Limiter(key_func=get_remote_address)
