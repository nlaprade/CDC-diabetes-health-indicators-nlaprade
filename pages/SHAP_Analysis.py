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

def on_sample_size_change():
    st.session_state.prev_sample_size = -1
    st.rerun()

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

# --- SHAP Preprocessing ---
X_sample = pd.DataFrame(X_sample, columns=X_train.columns)
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

# --- Tabs Layout with Persistence ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🔎 SHAP Overview"

def set_active_tab(tab_name: str):
    st.session_state.active_tab = tab_name
    st.rerun()

tabs = {
    "🔎 SHAP Overview": "tab1",
    "📊 SHAP Summary": "tab2",
    "📈 Dependence Plot": "tab3",
    "🕹️ Decision Plot": "tab4",
    "📥 Download SHAP Values": "tab5"
}

tab1, tab2, tab3, tab4, tab5 = st.tabs(list(tabs.keys()))

# --- Tab 1: SHAP Overview ---
with tab1:
    st.session_state.active_tab = "🔎 SHAP Overview"

    st.markdown("## What is SHAP?")
    st.markdown("""
SHAP stands for Shapley Additive Explanations.  
It is a method for interpreting machine learning models by assigning each feature a contribution value.  
This value shows how much the feature pushed the prediction higher or lower.
""")

    st.markdown("### Why Use SHAP?")
    st.markdown("""
- Understand why a model made a specific prediction  
- Identify which features are most influential for each individual prediction  
- Build trust and transparency in machine learning systems  
- Support auditability for clinical and decision-making workflows
""")

    st.markdown("### How It Works")
    st.markdown("""
SHAP is based on cooperative game theory.  
Each feature is treated like a player in a game, and the prediction is the payout.  
SHAP calculates how much each feature contributes to the final prediction by comparing all possible combinations of features.

In this dashboard, SHAP values show how your input features such as `BMI`, `Income`, and `Age` influence the predicted risk of prediabetes.
""")

    st.markdown("[Learn more about SHAP](https://shap.readthedocs.io/en/latest/)")
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(os.path.join(IMAGES_DIR, "shap_visio.jpg"), caption="SHAP values reflect how each feature contributes across all possible combinations")
    with col2:
        st.markdown("""
### Visual Explanation: SHAP as Cooperative Game

This diagram illustrates the core idea behind SHAP values using a simplified game-theory analogy.

- Each figure (A, B, C) represents a feature  
- The colored squares show the total "payout" when features are combined  
- SHAP calculates each feature's contribution by comparing all possible combinations

For example, the full combination of A, B, and C yields a total value of 10.  
By analyzing how this value changes when features are added or removed, SHAP assigns fair contribution scores to each one.

This visual helps explain why SHAP is grounded in cooperative game theory and how it ensures consistent, additive explanations.
""")




# --- Tab 2: SHAP Summary ---
with tab2:
        st.session_state.active_tab = "📊 SHAP Summary"

        st.markdown(
            "### 🔎 SHAP Summary Plot")
        # --- Initialize session state variables ---
        if "shap_plot_type" not in st.session_state:
            st.session_state.shap_plot_type = "dot"

        if "sample_size" not in st.session_state:
            st.session_state.sample_size = 50

        # --- Toggle between Dot and Bar plots ---
        def toggle_shap_plot():
            st.session_state.shap_plot_type = (
                "bar" if st.session_state.shap_plot_type == "dot" else "dot"
            )

        st.button(
            f"Switch to {'Bar' if st.session_state.shap_plot_type == 'dot' else 'Dot'} Plot",
            on_click=toggle_shap_plot,
        )

        # --- Layout ---
        col1, col2 = st.columns([1, 3])

        with col1:
            sample_size = st.slider(
                "Select number of samples",
                min_value=10,
                max_value=500,
                value=st.session_state.sample_size,
                step=10,
                key="sample_size"
            )

            st.markdown(
                f"Showing SHAP summary as **{st.session_state.shap_plot_type}** plot "
                f"with **{sample_size}** samples."
            )

            # --- Sample Subset ---
            X_subset = X_sample.iloc[:sample_size]
            shap_subset = shap_array[:sample_size]

        # --- SHAP Plot ---
        
        subcol1, subcol2 =st.columns([2, 1])
        with subcol1:
            plt.clf()
            shap.summary_plot(
                shap_subset,
                X_subset.values,
                feature_names=np.array(X_subset.columns),
                plot_type=st.session_state.shap_plot_type,
                show=False)
            fig = plt.gcf()
            fig.set_size_inches(12, 7.3)
            st.pyplot(fig)
        
        with subcol2:
            with st.container(border=True):
                st.markdown("#### Sample Size")
                st.markdown("""
Use the slider on the left to control how many samples are included in the SHAP summary plot.

- A smaller sample size loads faster and highlights individual variation  
- A larger sample size reveals more stable global patterns  
- The plot updates automatically when you adjust the slider
""")
            with st.container(border=True):
                st.markdown("#### Summary Plot Interpretation")
                st.markdown("""
This plot shows how each feature influences model predictions across the selected samples.

- **Color** represents feature value  
  - Red means high  
  - Blue means low  
  - Purple means mid-range  
- **Position** reflects SHAP value  
  - Left means the feature pushed the prediction lower  
  - Right means it pushed the prediction higher  
- **Density** shows how frequently a feature influences predictions  

Use this plot to explore global patterns, spot outliers, and understand how feature values shape predictions.
""")

# --- Tab 3: Dependence Plot ---
with tab3:
    st.markdown("### 📈 SHAP Dependence Plot")
    with st.expander("Dependence Plot Overview"):
        st.markdown("""
**Dependence Plot**  
- **X-axis** → Feature value  
- **Y-axis** → SHAP value (impact on prediction)  
- **Color** → Interaction with another feature  
- Reveals nonlinear effects and feature interactions
""")
    col1, col2 = st.columns(2)

    with col1:
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            feature_left = st.selectbox("Feature to plot", importance_order, index=0, key="feature_left")
        with subcol2:
            color_feature_left = st.selectbox("Color by feature", X_sample.columns, index=1, key="color_feature_left")

        fig_left, ax_left = plt.subplots(figsize=(10, 5))
        shap.dependence_plot(
            feature_left,
            shap_array,
            X_sample,
            interaction_index=color_feature_left,
            ax=ax_left,
            show=False
        )
        st.pyplot(fig_left, width='content')
    
    with col2:
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            feature_right = st.selectbox("Feature to plot", importance_order, index=0, key="feature_right")
        with subcol2:
            color_feature_right = st.selectbox("Color by feature", X_sample.columns, index=1, key="color_feature_right")
        
        fig_right, ax_right = plt.subplots(figsize=(10, 5))
        shap.dependence_plot(
            feature_right,
            shap_array,
            X_sample,
            interaction_index=color_feature_right,
            ax=ax_right,
            show=False
        )
        st.pyplot(fig_right, width='content')

with tab4:
    st.markdown("### 🕹️ Decision Plot")

    # Use class 0 expected value if multi-class
    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
        expected_val = explainer.expected_value[0]
    else:
        expected_val = explainer.expected_value

    col1, col2 = st.columns([2, 1])  # Wider plot column

    with col1:
        plt.clf()
        shap.decision_plot(
            expected_val,
            shap_array,
            X_sample,
            feature_names=list(X_sample.columns),
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(12, 10)
        st.pyplot(fig, use_container_width=False)

    with col2:
        with st.container(border=True):
            st.markdown("""
### How to Read This Plot
- **Cumulative impact**: Each line shows how features push the prediction from the base value.
- **Left to right**: Features are ordered by importance; early features have stronger influence.
- **Lines**: Each line represents a sample, useful for spotting outliers or tipping points.
- **Flat segments**: Feature had little impact for that sample.
- **Sharp jumps**: Feature caused a strong shift in prediction.""")

        with st.container(border=True):
            st.markdown("""
### Why It Matters
- Helps identify **which features dominate early** in the decision path.
- Reveals **threshold effects**, where predictions flip from low to high risk.
- Supports **clinical interpretability** by showing how predictions evolve step-by-step.

""")


# --- Tab 5: Download SHAP Values ---
with tab5:
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