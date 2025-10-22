import os
import pickle
import time
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@st.cache_resource
def load_all_models(paths):
    loaded = {}
    for name, path in paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
        else:
            st.warning(f"⚠️ Model file not found: {name} → {path}")
    return loaded

def evaluate_models(models, X_test, y_test, thresholds):
    results = []
    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        thresh = thresholds.get(name, 0.5)
        y_pred = (y_probs >= thresh).astype(int)
        results.append({
            "Model": name,
            "Accuracy": f"{accuracy_score(y_test, y_pred):.2f}",
            "Precision": f"{precision_score(y_test, y_pred):.2f}",
            "Recall": f"{recall_score(y_test, y_pred):.2f}",
            "F1 Score": f"{f1_score(y_test, y_pred):.2f}"
        })
    return pd.DataFrame(results).set_index("Model").round(2)

def handle_model_selection(models: dict, results_df, best_model_name: str):
    if "last_selected_model" not in st.session_state:
        st.session_state.last_selected_model = best_model_name

    selected_model_name = st.selectbox(
        "🔀 Select model for SHAP analysis",
        results_df.index.tolist(),
        index=results_df.index.tolist().index(st.session_state.last_selected_model)
    )
    selected_model = models[selected_model_name]

    if selected_model_name != st.session_state.last_selected_model:
        with st.spinner("🔄 Switching model and recalculating SHAP values..."):
            progress_bar = st.progress(0)
            for percent_complete in range(0, 101, 10):
                time.sleep(0.1)
                progress_bar.progress(percent_complete)
            progress_bar.empty()
        st.toast(f"✅ Successfully switched to **{selected_model_name}** and recalculated SHAP values! Please wait for the modules to be available.")
        time.sleep(5)
        st.session_state.last_selected_model = selected_model_name

    return selected_model_name, selected_model
