import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils.paths import DATA_PATH, MODEL_DIR, IMAGES_DIR
from utils.data_utils import load_data, preprocessing
from utils.model_utils import load_all_models

shap.initjs()

# --- Page Setup ---
st.set_page_config(page_title="Feature Importance Summary", page_icon=os.path.join(IMAGES_DIR, "icon.png"), layout="wide")
st.title("Feature Importance Summary")
st.caption("Explore how features influence model predictions.")

# --- Load Data ---
df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test, min_max = preprocessing(df)

# --- Load Models ---
model_paths = {
    "XGBoost": os.path.join(MODEL_DIR, "xgboost_prediabetes_model.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "randomforest_prediabetes_model.pkl"),
    "Extra Trees": os.path.join(MODEL_DIR, "extratrees_prediabetes_model.pkl"),
    "HistGradientBoosting": os.path.join(MODEL_DIR, "histgb_prediabetes_model.pkl"),
    "Gradient Boosting": os.path.join(MODEL_DIR, "gradientboosting_prediabetes_model.pkl")
}
models = load_all_models(model_paths)

# --- Model Selection ---
st.sidebar.title("Model Selection")

if "current_model" not in st.session_state:
    st.session_state.current_model = list(models.keys())[0]

selected_model = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys()),
    index=list(models.keys()).index(st.session_state.current_model)
)

if selected_model != st.session_state.current_model:
    st.sidebar.warning("⚠️ Switching models may take time on cloud-hosted dashboards.")
    if st.sidebar.button("✅ Confirm Model Switch"):
        st.session_state.current_model = selected_model
        st.rerun()

# --- Finalize Model ---
model = models[st.session_state.current_model]

# --- Caching ---
@st.cache_data
def get_sample(X_test, model_name):
    return X_test.sample(n=500, random_state=42)

@st.cache_resource
def get_explainer(model_name, _model, X_train):  # model_name ensures cache refresh
    return shap.Explainer(_model, X_train)

@st.cache_data
def get_shap_values(model_name, _explainer, X_sample):  # model_name ensures cache refresh
    return _explainer(X_sample)

# --- SHAP Computation ---
X_sample = get_sample(X_test, st.session_state.current_model)

explainer = get_explainer(st.session_state.current_model, model, X_train)
shap_values = get_shap_values(st.session_state.current_model, explainer, X_sample)

# --- Tabs Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 SHAP Summary",
    "📈 Dependence Plot",
    "🕹️ Decision Plot",
    "📥 Download SHAP Values"
])

# --- Tab 1: SHAP Summary ---
with tab1:
    st.markdown(
    "### 🔎 SHAP Summary Plot",
    help="""**Summary Plot**
- **Color** → Feature value (**red** = high, **blue** = low, **purple** = mid)  
- **Position** → SHAP value (**left** = negative, **right** = positive)  
- **Density** → Importance across samples  

**Feature Importance Plot**
- Ranks features by average absolute SHAP value  
- **Longer bars** = higher influence  
- Helps identify top drivers of prediction
""")

    # --- Initialize session state ---
    if "shap_plot_type" not in st.session_state:
        st.session_state.shap_plot_type = "dot"

    # --- Toggle button logic ---
    def toggle_shap_plot():
        st.session_state.shap_plot_type = (
            "bar" if st.session_state.shap_plot_type == "dot" else "dot")

    # --- Button with callback ---
    st.button(f"🔄 Switch to {'Bar' if st.session_state.shap_plot_type == 'dot' else 'Dot'} Plot", on_click=toggle_shap_plot)

    # --- Display SHAP plot ---
    st.markdown(f"Showing SHAP summary as **{st.session_state.shap_plot_type}** plot.")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    shap.summary_plot(shap_values, X_sample, plot_type=st.session_state.shap_plot_type, show=False, plot_size=None)
    st.pyplot(plt)


# --- Tab 2: Dependence Plot ---
with tab2:
    st.markdown("### 📈 SHAP Dependence Plot",
                help="""**Dependence Plot**  
- **X-axis** → Feature value  
- **Y-axis** → SHAP value (impact on prediction)  
- **Color** → Interaction with another feature  
- Reveals nonlinear effects and feature interactions
""")
    importance_order = pd.DataFrame(shap_values.values, columns=X_sample.columns).abs().mean().sort_values(ascending=False).index.tolist()
    col1, col2 = st.columns(2)
    with col1:
        feature = st.selectbox("Feature to plot", X_sample.columns, index=0)
    with col2:
        color_feature = st.selectbox("Color by feature", X_sample.columns, index=1)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    shap.dependence_plot(feature, shap_values.values, X_sample, interaction_index=color_feature, ax=ax, show=False)
    st.pyplot(plt)

# --- Tab 3: Decision Plot ---
with tab3:
    st.markdown("### 🕹️ Decision Plot", help="""**Decision Plot**  
- Shows how each feature contributes to a prediction  
- **Left to right** = cumulative impact  
- **Lines** = individual samples  
- Great for understanding thresholds and tipping points""")
    shap.decision_plot(explainer.expected_value, shap_values.values, X_sample, feature_names=list(X_sample.columns), show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 5.5)
    st.pyplot(fig)

# --- Tab 4: Download SHAP Values ---
with tab4:
    st.markdown("### 📥 Download SHAP Values")
    shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
    st.dataframe(shap_df.head(10))

    st.download_button(
        label="Download full SHAP values as CSV",
        data=shap_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{st.session_state.current_model}_shap_values.csv",
        mime="text/csv"
    )

# --- Footer ---
st.markdown("---")
st.caption("Built by Nicholas Laprade — [LinkedIn](https://www.linkedin.com/in/nicholas-laprade) • [GitHub](https://github.com/nlaprade)")