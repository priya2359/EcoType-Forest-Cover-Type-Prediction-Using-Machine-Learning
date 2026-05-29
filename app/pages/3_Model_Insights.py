# filename: app/pages/3_Model_Insights.py
# purpose:  Model comparison, confusion matrices, Optuna trial stats

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Model Insights -- EcoType", layout="wide")
st.title("Model Insights")

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures" / "model"
OPTUNA_DIR  = Path(__file__).resolve().parent.parent.parent / "reports" / "optuna"

# Discrete palette (shared rules: never sequential colorscale for category charts)
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#795548", "#607D8B"]


@st.cache_data
def load_image(path: str):
    with open(path, "rb") as f:
        return f.read()


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ── Model comparison table ────────────────────────────────────────────────────
st.subheader("All Models -- Test Set Performance")

baseline_data = {
    "Model":        ["Logistic Regression", "Decision Tree", "KNN",
                     "Random Forest", "Extra Trees", "XGBoost"],
    "Macro F1":     [0.6391, 0.8665, 0.8162, 0.9149, 0.9062, 0.9043],
    "Accuracy":     [0.6705, 0.9323, 0.9048, 0.9582, 0.9481, 0.9424],
    "Weighted F1":  [0.7039, 0.9331, 0.9033, 0.9577, 0.9472, 0.9414],
    "Role":         ["Baseline", "Interpretable", "GUVI required",
                     "Core ensemble", "Bonus ensemble", "Primary candidate"],
}
df_base = pd.DataFrame(baseline_data).sort_values("Macro F1", ascending=False).reset_index(drop=True)
st.dataframe(df_base.style.highlight_max(subset=["Macro F1", "Accuracy", "Weighted F1"],
                                          color="#c8e6c9"), use_container_width=True)

# Macro F1 bar chart
fig = go.Figure(go.Bar(
    x=df_base["Model"],
    y=df_base["Macro F1"],
    marker_color=PALETTE[:len(df_base)],
    text=[f"{v:.4f}" for v in df_base["Macro F1"]],
    textposition="outside",
))
fig.update_layout(
    title="Macro F1 Comparison (nb08 baseline, FAST_MODE=False)",
    yaxis_title="Macro F1",
    yaxis_range=[0.5, 1.0],
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# ── Optuna tuning results ─────────────────────────────────────────────────────
st.subheader("Hyperparameter Tuning (Optuna TPE)")
rf_csv = OPTUNA_DIR / "random_forest_trials.csv"
et_csv = OPTUNA_DIR / "extra_trees_trials.csv"

if rf_csv.exists() and et_csv.exists():
    rf_trials = load_csv(str(rf_csv))
    et_trials = load_csv(str(et_csv))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest Trials**")
        st.metric("Best CV macro F1", f"{rf_trials['value'].max():.4f}")
        st.metric("Mean CV macro F1", f"{rf_trials['value'].mean():.4f}")
        st.metric("Trials run", len(rf_trials))
        st.dataframe(rf_trials[["number", "value"]].rename(
            columns={"number": "Trial", "value": "CV macro F1"}
        ).sort_values("CV macro F1", ascending=False).head(5), use_container_width=True)

    with col2:
        st.markdown("**Extra Trees Trials**")
        st.metric("Best CV macro F1", f"{et_trials['value'].max():.4f}")
        st.metric("Mean CV macro F1", f"{et_trials['value'].mean():.4f}")
        st.metric("Trials run", len(et_trials))
        st.dataframe(et_trials[["number", "value"]].rename(
            columns={"number": "Trial", "value": "CV macro F1"}
        ).sort_values("CV macro F1", ascending=False).head(5), use_container_width=True)
else:
    st.info("Optuna trial CSVs not found. Run notebooks/06_model_building.py first.")

# ── Confusion matrices ────────────────────────────────────────────────────────
st.subheader("Confusion Matrices")

cm_files = sorted(FIGURES_DIR.glob("cm_*.png")) if FIGURES_DIR.exists() else []
if cm_files:
    cm_cols = st.columns(2)
    for i, cm_path in enumerate(cm_files):
        label = cm_path.stem.replace("cm_", "").replace("_", " ").title()
        with cm_cols[i % 2]:
            st.markdown(f"**{label}**")
            st.image(load_image(str(cm_path)), use_column_width=True)
else:
    st.info("No confusion matrix images found. Run notebooks/08_model_building.py first.")
