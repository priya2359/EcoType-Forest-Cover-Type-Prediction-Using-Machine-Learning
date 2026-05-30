"""Verify rate limiting works correctly.

Run: python scripts/test_rate_limit.py
Requires: uvicorn api.main:app running on localhost:8000
"""
import os
import sys
import requests

API_URL = "http://localhost:8000/predict"
PAYLOAD = {
    "Elevation": 2596, "Aspect": 51, "Slope": 3,
    "Horizontal_Distance_To_Hydrology": 258, "Vertical_Distance_To_Hydrology": 0,
    "Horizontal_Distance_To_Roadways": 510, "Hillshade_9am": 221,
    "Hillshade_Noon": 232, "Hillshade_3pm": 148,
    "Horizontal_Distance_To_Fire_Points": 6279,
    "Wilderness_Area": 1, "Soil_Type": 29,
}

try:
    EXPECTED_LIMIT = int(os.environ.get("RATE_LIMIT", "60/minute").split("/")[0])
except (ValueError, IndexError):
    print("ERROR: RATE_LIMIT must be in format 'N/minute'")
    sys.exit(1)

print(f"Testing rate limit: expected 429 at request {EXPECTED_LIMIT + 1}")

results = []
for i in range(EXPECTED_LIMIT + 5):
    resp = requests.post(API_URL, json=PAYLOAD)
    results.append(resp.status_code)
    print(f"Request {i + 1:3d}: HTTP {resp.status_code}")

first_429 = next((i + 1 for i, s in enumerate(results) if s == 429), None)

if first_429 is None:
    print(f"\nFAIL: No 429 received in {len(results)} requests")
    sys.exit(1)
elif first_429 <= EXPECTED_LIMIT:
    print(f"\nFAIL: Rate limit triggered too early at request {first_429} (expected {EXPECTED_LIMIT + 1})")
    sys.exit(1)
else:
    print(f"\nPASS: Rate limit correctly triggered at request {first_429}")
