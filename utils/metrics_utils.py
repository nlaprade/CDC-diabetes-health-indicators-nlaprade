import matplotlib.pyplot as plt
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
