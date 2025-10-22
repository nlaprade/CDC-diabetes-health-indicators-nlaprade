import streamlit as st
import os

from utils.paths import IMAGES_DIR

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

st.markdown("---")

# --- Visual Aid ---
st.image(os.path.join(IMAGES_DIR, "shap_visio.jpg"), caption="SHAP values reflect how each feature contributes across all possible combinations")

st.markdown("---")

# --- Optional Sidebar Link ---
st.sidebar.markdown("[Learn more about SHAP](https://shap.readthedocs.io/en/latest/)")

# --- Footer ---
st.markdown("---")
st.caption("Built by Nicholas Laprade — [LinkedIn](https://www.linkedin.com/in/nicholas-laprade) • [GitHub](https://github.com/nlaprade)")