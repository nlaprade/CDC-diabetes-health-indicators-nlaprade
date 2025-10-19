"""
Author: Nicholas Laprade
Date: 2025-10-19
Topic: CDC Diabetes Health Indicators - Dashboard
Dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # or whatever number of cores you want

import shap
import matplotlib.pyplot as plt

from imblearn.combine import SMOTETomek
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Streamlit Setup ---
st.set_page_config(
    page_title="CDC Prediabetes Classification Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.title("CDC Prediabetes Classification Dashboard")
st.markdown("A modular dashboard for SHAP interpretability and model benchmarking.")

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_012_health_indicators_BRFSS2015.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- Model Paths ---
model_paths = {
    "XGBoost": os.path.join(BASE_DIR, "models", "xgboost_prediabetes_model.pkl"),
    "Random Forest": os.path.join(BASE_DIR, "models", "randomforest_prediabetes_model.pkl"),
    "Extra Trees": os.path.join(BASE_DIR, "models", "extratrees_prediabetes_model.pkl"),
    "HistGradientBoosting": os.path.join(BASE_DIR, "models", "histgb_prediabetes_model.pkl"),
    "Gradient Boosting": os.path.join(BASE_DIR, "models", "gradientboosting_prediabetes_model.pkl"),
    "AdaBoost": os.path.join(BASE_DIR, "models", "adaboost_prediabetes_model.pkl")
}

# --- Load Models ---
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

models = load_all_models(model_paths)

# --- Load Thresholds ---
threshold_path = os.path.join(MODEL_DIR, "thresholds.pkl")
if os.path.exists(threshold_path):
    with open(threshold_path, "rb") as f:
        thresholds = pickle.load(f)
else:
    st.warning("⚠️ Thresholds file not found. Defaulting to 0.5 for all models.")
    thresholds = {name: 0.5 for name in models}

# --- Load Data ---
@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df

df = load_data(DATA_PATH)

# --- Preprocessing ---
def preprocessing(df):
    df["BMI_Outlier"] = (np.abs(df["BMI"]) > 3).astype(int)
    df["LowActivity_HighBMI"] = ((df["PhysActivity"] == 0) & (df["BMI"] > 30)).astype(int)
    df["LogBMI"] = np.log1p(df["BMI"])
    df["Income_Age"] = df["Income"] / (df["Age"] + 1)
    df["DistressCombo"] = (df["MentHlth"] + df["PhysHlth"]) * (df["GenHlth"] >= 4)
    df["SocioEconBurden"] = ((df["Income"] <= 3).astype(int) + (df["Education"] <= 2).astype(int) + (df["NoDocbcCost"] == 1).astype(int))
    df["LowEdu"] = (df["Education"] <= 2).astype(int)
    df["BMI_GenHlth"] = df["BMI"] * df["GenHlth"]
    df["CardioRisk"] = df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"]

    df_filtered = df[df["Diabetes_012"].isin([0.0, 1.0])]
    df_majority = df_filtered[df_filtered["Diabetes_012"] == 0.0]
    df_minority = df_filtered[df_filtered["Diabetes_012"] == 1.0]

    df_majority_downsampled = resample(df_majority, replace=False, n_samples=2 * len(df_minority), random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)
    df_balanced.drop(["Fruits", "Veggies"], axis=1, inplace=True)

    X = df_balanced.drop("Diabetes_012", axis=1)
    y = df_balanced["Diabetes_012"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train_bal, y_train_bal = SMOTETomek(random_state=42).fit_resample(X_train, y_train)

    return X_train_bal, X_test, y_train_bal, y_test

X_train_bal, X_test, y_train_bal, y_test = preprocessing(df)

# --- Evaluate Models ---
def evaluate_models(models, X_test, y_test):
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

results_df = evaluate_models(models, X_test, y_test)

# --- Model Comparison Section ---
st.subheader("📊 Model Performance Comparison")
st.dataframe(results_df, width='stretch')

best_model_name = results_df["F1 Score"].idxmax()
st.success(f"🏆 Best Model: **{best_model_name}**")
best_model = models[best_model_name]

selected_model_name = st.selectbox(
    "🔀 Select model for SHAP analysis",
    results_df.index.tolist(),
    index=results_df.index.tolist().index(best_model_name)
)
selected_model = models[selected_model_name]