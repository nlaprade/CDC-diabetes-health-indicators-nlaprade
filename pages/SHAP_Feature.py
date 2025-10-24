import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

from utils.paths import DATA_PATH, MODEL_DIR, IMAGES_DIR
from utils.data_utils import load_data, preprocessing
from utils.model_utils import load_all_models

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

shap.initjs()

# --- Page Setup ---
st.set_page_config(
    page_title="Feature Importance Summary",
    page_icon=os.path.join(IMAGES_DIR, "icon.png"),
    layout="wide"
)
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

# --- Load Thresholds ---
threshold_path = os.path.join(MODEL_DIR, "thresholds.pkl")

if os.path.exists(threshold_path):
    with open(threshold_path, "rb") as f:
        thresholds = pickle.load(f)
else:
    thresholds = {name: 0.5 for name in models}

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
    st.subheader("Model Configuration")

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
            st.session_state.model_switch_triggered = True  # triggers recompute
            st.toast(f"✅ Switched to {st.session_state.current_model}")
            st.rerun()

        elif cancel_switch:
            st.session_state.temp_model = st.session_state.current_model
            st.session_state.model_switch_triggered = False
            st.toast("⛔ Model switch cancelled")
            st.rerun()

# --- Finalize Model ---
model = models[st.session_state.current_model]

# --- Caching ---
@st.cache_data
def get_sample(X_test, model_name, n):
    return X_test.sample(n=n, random_state=42)

@st.cache_resource
def get_explainer(model_name, _model, X_train):
    return shap.Explainer(_model, X_train)

@st.cache_data
def get_shap_values(model_name, _explainer, X_sample):
    return _explainer(X_sample)

# --- Sample Size Slider ---
if "sample_size" not in st.session_state:
    st.session_state.sample_size = 50  # default value

if "prev_sample_size" not in st.session_state:
    st.session_state.prev_sample_size = st.session_state.sample_size

# --- Recompute Sample if Needed ---
sample_size_changed = st.session_state.sample_size != st.session_state.prev_sample_size

if (
    "X_sample" not in st.session_state
    or st.session_state.model_switch_triggered
    or sample_size_changed
):
    st.session_state.X_sample = get_sample(
        X_test,
        st.session_state.current_model,
        st.session_state.sample_size
    )
    st.session_state.model_switch_triggered = False
    st.session_state.prev_sample_size = st.session_state.sample_size

X_sample = st.session_state.X_sample

# --- SHAP Computation ---
explainer = get_explainer(st.session_state.current_model, model, X_train)
shap_values = get_shap_values(st.session_state.current_model, explainer, X_sample)

# --- SHAP Preprocessing (shared across tabs) ---
if isinstance(X_sample, np.ndarray):
    X_sample = pd.DataFrame(X_sample, columns=X_train.columns)
else:
    X_sample = pd.DataFrame(X_sample)

X_sample.columns = [str(col) for col in X_sample.columns]

if isinstance(shap_values, list):
    shap_array = shap_values[0].values if hasattr(shap_values[0], "values") else shap_values[0]
elif hasattr(shap_values, "values") and shap_values.values.ndim == 3:
    shap_array = shap_values.values[:, :, 0]
else:
    shap_array = shap_values.values if hasattr(shap_values, "values") else shap_values

if shap_array.ndim == 3:
    shap_array = np.array([np.diag(sample) for sample in shap_array])

if shap_array.shape[1] != X_sample.shape[1]:
    st.warning(
        f"⚠️ Mismatch detected: SHAP has {shap_array.shape[1]} features, "
        f"but X_sample has {X_sample.shape[1]}. Adjusting automatically."
    )
    min_cols = min(shap_array.shape[1], X_sample.shape[1])
    shap_array = shap_array[:, :min_cols]
    X_sample = X_sample.iloc[:, :min_cols]

importance_order = (
    pd.DataFrame(shap_array, columns=X_sample.columns)
    .abs()
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

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
"""
    )

    if "shap_plot_type" not in st.session_state:
        st.session_state.shap_plot_type = "dot"

    def toggle_shap_plot():
        st.session_state.shap_plot_type = (
            "bar" if st.session_state.shap_plot_type == "dot" else "dot"
        )

    st.button(
        f"Switch to {'Bar' if st.session_state.shap_plot_type == 'dot' else 'Dot'} Plot",
        on_click=toggle_shap_plot
    )

    col1, col2 = st.columns([1, 3])  # col1 is narrower
    with col1:
        st.session_state.sample_size = st.slider(
            "Select number of samples",
            min_value=10,
            max_value=500,
            value=st.session_state.sample_size,
            step=10,
            key="sample_size_slider"
        )

    st.markdown(f"Showing SHAP summary as **{st.session_state.shap_plot_type}** plot.")
    col1, col2 = st.columns(2)
    with col1:
        plt.clf()
        shap.summary_plot(
            shap_array,
            X_sample.values,
            feature_names=np.array(X_sample.columns),
            plot_type=st.session_state.shap_plot_type,
            show=False)
        fig = plt.gcf()
        fig.set_size_inches(12, 5.5)
        st.pyplot(fig)

# --- Tab 2: Dependence Plot ---
with tab2:
    st.markdown("### 📈 SHAP Dependence Plot",
                help="""**Dependence Plot**  
- **X-axis** → Feature value  
- **Y-axis** → SHAP value (impact on prediction)  
- **Color** → Interaction with another feature  
- Reveals nonlinear effects and feature interactions
""")

    # Create columns for controls and plot side by side
    col1, col2 = st.columns([1, 3])  # Adjust ratio if needed

    with col1:
        feature = st.selectbox("Feature to plot", importance_order, index=0)
        color_feature = st.selectbox("Color by feature", X_sample.columns, index=1)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.dependence_plot(
            feature,
            shap_array,
            X_sample,
            interaction_index=color_feature,
            ax=ax,
            show=False
        )
        st.pyplot(fig)

# --- Tab 3: Decision Plot ---
with tab3:
    st.markdown("### 🕹️ Decision Plot", help="""**Decision Plot**  
- Shows how each feature contributes to a prediction  
- **Left to right** = cumulative impact  
- **Lines** = individual samples  
- Great for understanding thresholds and tipping points""")

    # Use class 0 expected value if multi-class
    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
        expected_val = explainer.expected_value[0]
    else:
        expected_val = explainer.expected_value
    
    col1, col2 = st.columns(2)
    
    with col1:
        plt.clf()
        shap.decision_plot(
            expected_val,
            shap_array,
            X_sample,
            feature_names=list(X_sample.columns),
            show=False)
        fig = plt.gcf()
        fig.set_size_inches(12, 10)
        st.pyplot(fig)

# --- Tab 4: Download SHAP Values ---
with tab4:
    col1, col2 = st.columns(2)
    st.markdown("### 📥 Download SHAP Values")
    with col1:
        shap_df = pd.DataFrame(shap_array, columns=X_sample.columns)
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