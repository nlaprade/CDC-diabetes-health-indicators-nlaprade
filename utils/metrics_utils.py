import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def compute_confusion_metrics(model, X_test, y_test, threshold=0.5):
    # Predict
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        try:
            y_pred_score = model.decision_function(X_test)
            y_pred = (y_pred_score >= threshold).astype(int)
        except AttributeError:
            y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    return {
        "y_pred": y_pred,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "breakdown": {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    }

def plot_confusion_matrix(cm, figsize=(3, 3), title=""):
    """
    Returns a matplotlib figure of the confusion matrix.
    """
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def render_threshold_slider(thresholds):
    current_model = st.session_state.current_model

    # Only initialize slider value before widget is rendered
    if "threshold_slider" not in st.session_state:
        st.session_state.threshold_slider = st.session_state.thresholds.get(current_model, 0.5)

    # Render slider using session state
    threshold = st.slider(
        label="Threshold",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="threshold_slider",
        label_visibility="collapsed")

    # Toast on change
    if threshold != st.session_state.get("prev_threshold", threshold):
        if not st.session_state.get("just_reset_thresholds", False):
            st.toast(f"✅ Threshold changed to {threshold:.2f}")
        st.session_state.prev_threshold = threshold

    # Clear reset flag
    if st.session_state.get("just_reset_thresholds", False):
        del st.session_state["just_reset_thresholds"]

    # Sync threshold to model
    st.session_state.thresholds[current_model] = threshold

    # Reset button
    if st.button("🔁 Reset Thresholds"):
        # Update model threshold
        st.session_state.thresholds = thresholds.copy()
        st.session_state.just_reset_thresholds = True

        # Remove slider key so it reinitializes on rerun
        if "threshold_slider" in st.session_state:
            del st.session_state["threshold_slider"]

        st.toast("🔁 Thresholds reset to optimal value")
        st.rerun()

def on_model_change():
    st.session_state.model_switch_triggered = True

def render_model_change(models):
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