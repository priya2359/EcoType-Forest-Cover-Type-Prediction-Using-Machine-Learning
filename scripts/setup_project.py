# filename: scripts/setup_project.py
# purpose:  Create all project directories and verify dataset presence
# version:  1.0

import os
from pathlib import Path

DIRS = [
    "data/raw",
    "data/processed",
    "data/interim",
    "notebooks",
    "src",
    "models/encoders",
    "models/scalers",
    "models/trained",
    "models/best_model",
    "artifacts",
    "reports/figures/eda",
    "reports/figures/model",
    "reports/figures/features",
    "app/pages",
    "app/components",
    "app/utils",
    "api/routes",
    "api/schemas",
    "api/middleware",
    "configs",
    "tests",
    "docs",
    "scripts",
    "docker",
    ".streamlit",
    ".github/workflows",
]

def main():
    print("EcoType Project Setup")
    print("=" * 50)

    project_root = Path(__file__).parent.parent
    created, existed = [], []

    for d in DIRS:
        path = project_root / d
        if path.exists():
            existed.append(d)
        else:
            path.mkdir(parents=True, exist_ok=True)
            created.append(d)
            print(f"  Created: {d}/")

    print(f"\n{len(created)} directories created, {len(existed)} already existed.")

    # Dataset check
    dataset = project_root / "data" / "raw" / "forest_cover.csv"
    if dataset.exists():
        size_mb = dataset.stat().st_size / (1024 * 1024)
        print(f"\n  Dataset found: data/raw/forest_cover.csv ({size_mb:.1f} MB)")
    else:
        print("\n  WARNING: data/raw/forest_cover.csv not found.")
        print("           Copy your GUVI dataset there before running notebooks.")

    print("\nSetup complete. Next: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
