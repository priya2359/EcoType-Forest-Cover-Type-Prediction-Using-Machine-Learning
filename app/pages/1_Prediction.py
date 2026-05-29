# filename: app/pages/1_Prediction.py
# purpose:  Prediction page -- 12-field form, POST to API, display result

import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Prediction -- EcoType", layout="wide")
st.title("Classify Forest Cover Type")
st.markdown("Enter cartographic measurements to predict the forest cover type.")

# Discrete palette -- 7 classes (shared rules: never sequential colorscale)
CLASS_COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#F44336", "#00BCD4", "#795548",
]

api_url = st.session_state.get("api_url", "http://localhost:8000")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.subheader("Quantitative Features")
    col1, col2 = st.columns(2)

    with col1:
        elevation = st.number_input("Elevation (m)", min_value=1859, max_value=3858, value=2596,
                                     help="Elevation in meters (1859–3858)")
        aspect    = st.number_input("Aspect (degrees)", min_value=0, max_value=360, value=51,
                                     help="Compass bearing of slope (0–360)")
        slope     = st.number_input("Slope (degrees)", min_value=0, max_value=66, value=3,
                                     help="Steepness of slope (0–66)")
        h_hydro   = st.number_input("Horizontal Distance to Hydrology (m)", min_value=0, value=258,
                                     help="Horizontal distance to nearest water (≥ 0)")
        v_hydro   = st.number_input("Vertical Distance to Hydrology (m)", value=0,
                                     help="Vertical distance to nearest water (can be negative)")
        h_road    = st.number_input("Horizontal Distance to Roadways (m)", min_value=0, value=510,
                                     help="Horizontal distance to nearest road (≥ 0)")

    with col2:
        hs_9am    = st.number_input("Hillshade 9am (0–255)", min_value=0, max_value=255, value=221,
                                     help="Hillshade index at 9am")
        hs_noon   = st.number_input("Hillshade Noon (0–255)", min_value=0, max_value=255, value=232,
                                     help="Hillshade index at noon")
        hs_3pm    = st.number_input("Hillshade 3pm (0–255)", min_value=0, max_value=255, value=148,
                                     help="Hillshade index at 3pm")
        h_fire    = st.number_input("Horizontal Distance to Fire Points (m)", min_value=0, value=6279,
                                     help="Horizontal distance to nearest wildfire ignition point (≥ 0)")

        st.subheader("Categorical Features")
        wilderness = st.selectbox("Wilderness Area", options=[1, 2, 3, 4],
                                   help="Wilderness designation (1–4)")
        soil_type  = st.selectbox("Soil Type", options=list(range(1, 41)),
                                   help="Soil type code (1–40)")

    submitted = st.form_submit_button("Predict Cover Type", type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "Elevation":                          elevation,
        "Aspect":                             aspect,
        "Slope":                              slope,
        "Horizontal_Distance_To_Hydrology":   h_hydro,
        "Vertical_Distance_To_Hydrology":     v_hydro,
        "Horizontal_Distance_To_Roadways":    h_road,
        "Hillshade_9am":                      hs_9am,
        "Hillshade_Noon":                     hs_noon,
        "Hillshade_3pm":                      hs_3pm,
        "Horizontal_Distance_To_Fire_Points": h_fire,
        "Wilderness_Area":                    wilderness,
        "Soil_Type":                          soil_type,
    }

    with st.spinner("Predicting..."):
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)

            if resp.status_code == 200:
                result = resp.json()
                st.success(f"**{result['cover_type_name']}**")

                col_res, col_chart = st.columns([1, 2])
                with col_res:
                    st.metric("Cover Type", result["cover_type_name"])
                    st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    st.metric("Class ID", result["cover_type_id"])

                with col_chart:
                    probs = result["probabilities"]
                    labels = list(probs.keys())
                    values = [v * 100 for v in probs.values()]
                    colors = CLASS_COLORS[:len(labels)]

                    fig = go.Figure(go.Bar(
                        x=labels, y=values,
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in values],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Class Probabilities (%)",
                        yaxis_title="Probability (%)",
                        yaxis_range=[0, max(values) * 1.2],
                        showlegend=False,
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif resp.status_code == 503:
                st.warning("Model is still loading. Please wait a moment and try again.")
            elif resp.status_code == 422:
                st.error(f"Invalid input: {resp.json().get('detail', 'Validation error')}")
            else:
                st.error(f"API error {resp.status_code}: {resp.text[:200]}")

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach API at {api_url}. Is it running?")
        except requests.exceptions.Timeout:
            st.warning("Request timed out. The API may be starting up (Render cold start: 30–60s). Please retry.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
