"""Verify calibration curves PNG is valid after running 07_final_evaluation.py.

Run: python scripts/verify_calibration.py
"""
from PIL import Image
from pathlib import Path
import sys

cal_path = Path("reports/figures/model/calibration_curves.png")

if not cal_path.exists():
    print(f"FAIL: Calibration curves not found at {cal_path}")
    sys.exit(1)

try:
    img = Image.open(cal_path)
except Exception as e:
    print(f"FAIL: Cannot open image: {e}")
    sys.exit(1)

if img.size[0] < 500 or img.size[1] < 500:
    print(f"FAIL: Image too small: {img.size}. Expected at least 500x500px.")
    sys.exit(1)

if img.mode not in ("RGB", "RGBA", "L"):
    print(f"WARNING: Unusual image mode: {img.mode}")

print(f"PASS: {img.size[0]}x{img.size[1]}px, mode={img.mode}")
