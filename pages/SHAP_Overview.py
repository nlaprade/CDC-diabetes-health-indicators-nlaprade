import streamlit as st
import os

from utils.paths import IMAGES_DIR, MODEL_DIR
from utils.model_utils import load_all_models

# --- Streamlit Setup ---
st.set_page_config(
    page_title="CDC Prediabetes Classification Dashboard",
    page_icon=os.path.join(IMAGES_DIR, "icon.png"),
    layout="wide"
)

# --- Page Title ---
st.title("SHAP Overview")

# --- SHAP Interpretability Section ---
st.subheader("❓What are SHAP Values❓")

# --- Load Models ---
model_paths = {
    "XGBoost": os.path.join(MODEL_DIR, "xgboost_prediabetes_model.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "randomforest_prediabetes_model.pkl"),
    "Extra Trees": os.path.join(MODEL_DIR, "extratrees_prediabetes_model.pkl"),
    "HistGradientBoosting": os.path.join(MODEL_DIR, "histgb_prediabetes_model.pkl"),
    "Gradient Boosting": os.path.join(MODEL_DIR, "gradientboosting_prediabetes_model.pkl")
}
models = load_all_models(model_paths)

# --- Initialize session state ---
if "current_model" not in st.session_state:
    st.session_state.current_model = list(models.keys())[0]

if "temp_model" not in st.session_state:
    st.session_state.temp_model = st.session_state.current_model

if "model_switch_triggered" not in st.session_state:
    st.session_state.model_switch_triggered = False

# --- Callback to track selection change ---
def on_model_change():
    st.session_state.model_switch_triggered = True

# --- Sidebar Content ---
with st.sidebar:
    st.subheader("Model Selection")

    # Model selector driven by temp_model
    st.selectbox(
        "Choose Model",
        list(models.keys()),
        index=list(models.keys()).index(st.session_state.temp_model),
        key="model_selector",
        on_change=on_model_change
    )

    # Update temp_model if user changed selection
    if st.session_state.model_switch_triggered:
        st.session_state.temp_model = st.session_state.model_selector

    # Show confirm/cancel buttons only if temp_model differs from current_model
    if st.session_state.temp_model != st.session_state.current_model:
        st.warning("⚠️ Switching models may take time on cloud-hosted dashboards.")
        confirm_switch = st.button("✅ Confirm Model Switch")
        cancel_switch = st.button("⛔ Cancel Model Change")

        if confirm_switch:
            st.session_state.current_model = st.session_state.temp_model
            st.session_state.model_switch_triggered = False
            st.toast(f"✅ Switched to {st.session_state.current_model}")
            st.rerun()

        elif cancel_switch:
            st.session_state.temp_model = st.session_state.current_model
            st.session_state.model_switch_triggered = False
            st.toast("⛔ Model switch cancelled")
            st.rerun()

st.markdown("""
**SHAP** (**SH**apley **A**dditive ex**P**lanations) is a powerful method for interpreting machine learning models.  
It assigns each feature a contribution value showing how much that feature pushed the prediction up or down.

🔍 **Why use SHAP?**
- Understand *why* a model made a specific prediction.
- Reveal which features are most influential for each individual prediction.
- Support trust, transparency, and auditability in ML systems — especially important for clinicians and decision-makers.

**How does it work?**
SHAP is based on cooperative game theory. Imagine each feature as a player in a game, and the prediction as the payout.  
SHAP calculates how much each feature contributes to the final prediction by comparing all possible combinations of features.

**In this dashboard**, SHAP values show how your input features (like `BMI`, `Income`, `Age`, etc.) influence the predicted **risk of prediabetes**.
""")
st.markdown("[Learn more about SHAP](https://shap.readthedocs.io/en/latest/)")

st.markdown("---")

# --- Visual Aid ---
st.image(os.path.join(IMAGES_DIR, "shap_visio.jpg"), caption="SHAP values reflect how each feature contributes across all possible combinations")

# --- Footer ---
st.markdown("---")
st.caption("Built by Nicholas Laprade — [LinkedIn](https://www.linkedin.com/in/nicholas-laprade) • [GitHub](https://github.com/nlaprade)")