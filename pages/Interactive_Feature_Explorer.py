import streamlit as st
import os
import plotly.express as px

from utils.paths import IMAGES_DIR, DATA_PATH, MODEL_DIR
from utils.plot_utils import plot_xy_selector, get_corr_matrix
from utils.data_utils import load_data
from utils.model_utils import load_all_models
from utils.metrics_utils import render_model_change

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# --- Page Setup ---
st.set_page_config(
    page_title="Dataset Feature Explorer",
    page_icon=os.path.join(IMAGES_DIR, "icon.png"),
    layout="wide"
)

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

def on_model_change():
    st.session_state.model_switch_triggered = True

render_model_change(models)

# --- Load Data ---
df = load_data(DATA_PATH)

# --- Session State Initialization ---
if "plots_active" not in st.session_state:
    st.session_state.plots_active = {}
if "plot_uid" not in st.session_state:
    st.session_state.plot_uid = 0

# --- Title and Caption ---
st.title("Feature Explorer")

st.info("**Class labels:** 0 = **Healthy** · 1 = **Prediabetes** · 2 = **Diabetes**")

tab1, tab2, tab3, tab4 = st.tabs(["🔥 Feature Correlation Heatmap", "📊  Feature Comparison","🔎 Column Feature Explorer", "🌞 Sunburst Interactive Plot"])

with tab1:
# --- Heatmap Section ---
    st.caption("Explore relationships between numerical features using a color-coded correlation matrix.")
    corr = get_corr_matrix(df)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="temps",
        aspect="auto",
        labels=dict(x="Features", y="Features", color="Correlation"))

    fig.update_layout(
        xaxis_tickangle=45,
        margin=dict(t=40, l=0, r=0, b=0),
        height=800)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.caption("Select features to compare across classes. Add up to 3 plots and explore relationships interactively.")
    # --- Initialize Session State ---
    if "plot_uid" not in st.session_state:
        st.session_state.plot_uid = 0
    if "plots_active" not in st.session_state:
        st.session_state.plots_active = {}

    # --- Handle Deletion First ---
    to_delete = None
    for plot_key in list(st.session_state.plots_active.keys()):
        if st.session_state.get(f"delete_{plot_key}"):
            to_delete = plot_key

    if to_delete:
        del st.session_state.plots_active[to_delete]

    # --- Recalculate plot count after deletion
    current_plot_count = len(st.session_state.plots_active)

    # --- Add Plot Button (triggers rerun for immediate rendering)
    if current_plot_count < 3:
        if st.button("➕ Add Plot", key="add_plot"):
            st.toast("Adding plot please wait...")
            uid = st.session_state.plot_uid
            plot_key = f"plot_{uid}"
            st.session_state.plots_active[plot_key] = {
                "uid": uid,
                "x_col": None,
                "y_col": None
            }
            st.session_state.plot_uid += 1
            st.rerun()

    # --- Render Plots ---
    plot_keys = list(st.session_state.plots_active.keys())
    if plot_keys:
        cols = st.columns(len(plot_keys)) if len(plot_keys) > 1 else [st.container()]
        for plot_key, col in zip(plot_keys, cols):
            plot_info = st.session_state.plots_active[plot_key]
            uid = plot_info["uid"]
            with col:
                with st.container():
                    st.checkbox("❌ Delete", key=f"delete_{plot_key}")
                    plot_xy_selector(df, plot_id=plot_key, uid=uid)

with tab3:
    st.caption("Visualize how each feature is distributed across diabetes classes. Automatically adapts to categorical or numerical data.")
    # --- Feature Explorer Section ---
    selected_col = st.selectbox("Choose a feature to explore:", df.columns, index=4)

    if df[selected_col].nunique() < 20:
        plot_data = df.groupby([selected_col, 'Diabetes_012']).size().unstack(fill_value=0)
        st.bar_chart(plot_data)
    else:
        fig = px.histogram(df, x=selected_col, color='Diabetes_012', barmode='overlay')
        st.plotly_chart(fig, use_container_width=True, key=f"hist_{selected_col}")
    st.info("☝️ Click on class labels in the legend to show or hide them in the plot.")

with tab4:
    st.caption("Sunburst plots visualize hierarchical data, expanding outward from root to leaves.")

    col1, col2, col3 = st.columns([1.5, 2.5, 0.5])  # Wider center column for the plot

    with col1:

            st.markdown("### Plot Structure")
            st.markdown("""
| Layer         | Feature        |
|---------------|----------------|
| Root          | Age            |
| Second Layer  | BMI            |
| Outer Layer   | Diabetes_012   |
""")


            st.markdown("### Column Definitions")

            st.markdown("**Age Labels**")
            st.markdown("""
| Code | Age Range |
|------|------------|
| 1    | 18 - 24      |
| 2    | 25 - 29      |
| 3    | 30 - 34      |
| 4    | 35 - 39      |
| 5    | 40 - 44      |
| 6    | 45 - 49      |
| 7    | 50 - 54      |
| 8    | 55 - 59      |
| 9    | 60 - 64      |
| 10   | 65 - 69      |
| 11   | 70 - 74      |
| 12   | 75 - 79      |
| 13   | 80+          |
""")

    with col2:
        fig = px.sunburst(df, path=["Age", "BMI", "Diabetes_012"])
        fig.update_layout(width=1000, height=850)
        plot_config = {"displayModeBar": False, "responsive": True}
        st.plotly_chart(fig, use_container_width=False, config=plot_config)

# --- Footer ---
st.markdown("---")
st.caption("Built by Nicholas Laprade — [LinkedIn](https://www.linkedin.com/in/nicholas-laprade) • [GitHub](https://github.com/nlaprade)")