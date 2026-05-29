# filename: app/streamlit_app.py
# purpose:  EcoType Streamlit entry point with sidebar API config
# version:  1.0

import os
import streamlit as st

# pages/ routing is automatic — no st.navigation() used (requires Streamlit >= 1.36.0)
st.set_page_config(
    layout="wide",
    page_title="EcoType — Forest Cover Classifier",
    page_icon="🌲",
    initial_sidebar_state="auto",
)


def _get_api_url() -> str:
    try:
        return st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))
    except Exception:
        return os.environ.get("API_URL", "http://localhost:8000")


if "api_url" not in st.session_state:
    st.session_state["api_url"] = _get_api_url()

with st.sidebar:
    st.markdown("## ⚙️ API Configuration")
    st.session_state["api_url"] = st.text_input(
        "API URL",
        value=st.session_state["api_url"],
        help="Base URL of the FastAPI backend",
    )
    st.caption("⚠️ First request on Render free tier may take 30–60s (cold start).")
    st.divider()
    st.caption("EcoType · Forest Cover Prediction")

st.title("🌲 EcoType — Forest Cover Type Prediction")
st.markdown(
    "**Classify forest cover types from cartographic data.** "
    "Input land features to predict which of 7 forest types is present."
)

col1, col2, col3 = st.columns(3)
col1.metric("Models Trained", "6", "LR · DT · KNN · RF · ET · XGB")
col2.metric("Dataset", "145,891 rows", "GUVI subset")
col3.metric("Target Classes", "7", "Forest cover types")

st.divider()
st.markdown(
    "**Navigate using the sidebar:**\n"
    "- 🎯 **Prediction** — classify a land area\n"
    "- 📊 **EDA Dashboard** — explore the dataset\n"
    "- 🔍 **Model Insights** — compare model performance"
)
