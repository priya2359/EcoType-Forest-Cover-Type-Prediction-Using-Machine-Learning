# filename: app/pages/2_EDA_Dashboard.py
# purpose:  EDA Dashboard -- saved figures + dataset statistics

import os
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="EDA Dashboard -- EcoType", layout="wide")
st.title("Exploratory Data Analysis")
st.markdown("Dataset: **145,891 rows × 13 columns** — GUVI Forest Cover subset.")

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures" / "eda"


@st.cache_data
def load_image(path: str):
    with open(path, "rb") as f:
        return f.read()


# ── Dataset overview ─────────────────────────────────────────────────────────
st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows", "145,891")
col2.metric("Features (post-OHE)", "59")
col3.metric("Target Classes", "7")
col4.metric("Duplicates / Nulls", "0 / 0")

# ── Class distribution ────────────────────────────────────────────────────────
st.subheader("Class Distribution (Train Set)")
st.markdown("""
| Class | Cover Type | Train Rows | Note |
|---|---|---|---|
| 1 | Spruce/Fir | 24,888 | — |
| 2 | **Lodgepole Pine** | **82,457** | Dominant class |
| 3 | Ponderosa Pine | 1,728 | SMOTE applied |
| 4 | Cottonwood/Willow | 1,728 | SMOTE applied |
| 5 | Aspen | 2,455 | Above threshold — no SMOTE |
| 6 | Douglas-fir | 1,728 | SMOTE applied |
| 7 | Krummholz | 1,728 | SMOTE applied |

**Class imbalance ratio:** 47.7× (max/min). SMOTE applied to classes 3, 4, 6, 7 → target 2,455 each.
""")

# ── Soil type sparsity ────────────────────────────────────────────────────────
st.subheader("Soil Type Sparsity")
st.info("30 of 39 Soil_Type OHE columns have < 1% of training rows. "
        "Most were dropped in feature selection (nb07 importance threshold 0.0005). "
        "Soil_Type_25 had only 1 row — Soil_Type_28 had 6.")

# ── EDA figures ───────────────────────────────────────────────────────────────
figure_meta = [
    ("01_class_distribution.png",    "Class Distribution"),
    ("02_quant_distributions.png",   "Quantitative Feature Distributions"),
    ("03_correlation_heatmap.png",   "Feature Correlation Heatmap"),
    ("04_features_by_class.png",     "Features by Cover Type (Box Plots)"),
    ("05_wilderness_by_class.png",   "Wilderness Area vs Cover Type"),
    ("06_soil_sparsity.png",         "Soil Type OHE Column Sparsity"),
    ("07_engineered_features.png",   "Engineered Features by Cover Type"),
    ("08_smote_distribution.png",    "Class Distribution Before/After SMOTE"),
    ("09_feature_importance.png",    "Feature Importance (Random Forest)"),
]

available = [(fname, title) for fname, title in figure_meta
             if (FIGURES_DIR / fname).exists()]

if not available:
    st.warning(f"No EDA figures found in {FIGURES_DIR}. Run notebooks/05_eda.py first.")
else:
    for fname, title in available:
        st.subheader(title)
        st.image(load_image(str(FIGURES_DIR / fname)), use_column_width=True)
