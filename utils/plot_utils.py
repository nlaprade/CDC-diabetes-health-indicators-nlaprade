import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px

def generate_plot(df, x_col, y_col, hue_col="Diabetes_012", figsize=(3.5, 2.5), title=None):
    fig, ax = plt.subplots(figsize=figsize)

    filtered_df = df[df[hue_col].isin([0, 1])]

    colors = {0: "#1f77b4", 1: "#ff7f0e"}
    markers = {0: "o", 1: "s"}
    sizes = {0: 10, 1: 30}
    alphas = {0: 0.6, 1: 0.8}

    for cls in [0, 1]:
        subset = filtered_df[filtered_df[hue_col] == cls]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            alpha=alphas[cls],
            s=sizes[cls],
            color=colors[cls],
            marker=markers[cls],
            edgecolors='black',
            linewidths=0.5,
            label=f"{hue_col} = {cls}"
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} vs {x_col}", fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.8)
    st.pyplot(fig)

def plot_xy_selector(df, plot_id, uid):
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X-axis", df.columns, key=f"x_{uid}")
    with col2:
        y_col = st.selectbox("Y-axis", df.columns, key=f"y_{uid}")

    if x_col and y_col:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=df.columns[-1],  # or your target column
            title=f"{y_col} vs {x_col}",
            labels={x_col: x_col, y_col: y_col},
            height=400
        )
        fig.update_layout(
            font=dict(size=14),
            title_font=dict(size=18),
            margin=dict(t=40, l=0, r=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{uid}")

@st.cache_data
def get_corr_matrix(df):
    return df.select_dtypes(include="number").corr()