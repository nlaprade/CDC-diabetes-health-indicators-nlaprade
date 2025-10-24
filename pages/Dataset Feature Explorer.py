import streamlit as st
import os
import matplotlib.pylab as plt
import seaborn as sns

from utils.paths import IMAGES_DIR, DATA_PATH
from utils.plot_utils import generate_plot, plot_xy_selector, get_corr_matrix
from utils.data_utils import load_data

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# --- Page Setup ---
st.set_page_config(
    page_title="Dataset Feature Explorer",
    page_icon=os.path.join(IMAGES_DIR, "icon.png"),
    layout="wide"
)

# --- Load Data ---
df = load_data(DATA_PATH)

# --- Session State Initialization ---
if "plots_active" not in st.session_state:
    st.session_state.plots_active = {}
if "plot_uid" not in st.session_state:
    st.session_state.plot_uid = 0

# --- Title and Caption ---
st.title("🧮 Feature Explorer")
st.caption("Explore feature relationships with a correlation heatmap and class-separated scatter plots.")

# --- Heatmap Section ---
corr = get_corr_matrix(df)
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        cbar_kws={"shrink": 0.7},
        annot_kws={"size": 8}
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout(pad=2.0)
    st.pyplot(fig)

# --- Plot Section ---
st.markdown("### 📊 Select and Compare Features")

# --- Add Plot Button ---
if len(st.session_state.plots_active) < 3:
    if st.button("➕ Add Plot"):
        uid = st.session_state.plot_uid
        plot_key = f"plot_{uid}"
        st.session_state.plots_active[plot_key] = {
            "uid": uid,
            "x_col": None,
            "y_col": None
        }
        st.session_state.plot_uid += 1

# --- Render Plots ---
plot_keys = list(st.session_state.plots_active.keys())
plot_count = len(plot_keys)
to_delete = None

if plot_count == 1:
    plot_key = plot_keys[0]
    plot_info = st.session_state.plots_active[plot_key]
    uid = plot_info["uid"]

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("❌ Delete Plot", key=f"delete_{plot_key}"):
            to_delete = plot_key
        plot_xy_selector(df, plot_id=plot_key, uid=uid)


elif plot_count > 1:
    cols = st.columns(plot_count)
    for plot_key, col in zip(plot_keys, cols):
        plot_info = st.session_state.plots_active[plot_key]
        uid = plot_info["uid"]
        with col:
            if st.button("❌ Delete Plot", key=f"delete_{plot_key}"):
                to_delete = plot_key
            plot_xy_selector(df, plot_id=plot_key, uid=uid)

# --- Deferred Deletion ---
if to_delete:
    del st.session_state.plots_active[to_delete]
    st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("Built by Nicholas Laprade — [LinkedIn](https://www.linkedin.com/in/nicholas-laprade) • [GitHub](https://github.com/nlaprade)")