"""
Author: Nicholas Laprade
Date: 2025-10-17
Topic: CDC Diabetes Health Indicators - Preprocessing & Data Exploration
Dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
"""

import os
import pickle
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_recall_curve, f1_score
)
from xgboost import XGBClassifier

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_012_health_indicators_BRFSS2015.csv")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# -------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------------------------
# Exploratory Data Analysis
# -------------------------------------------------------------------
# Correlation Heatmap
plt.figure(figsize=(20, 14))
sns.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "heatmap.png"))
plt.close()

# BMI Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["BMI"], bins=30, kde=True, color="skyblue")
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "bmi_dist.png"))
plt.close()

# -------------------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------------------
df["BMI_Outlier"] = (np.abs(df["BMI"]) > 3).astype(int)
df["LowActivity_HighBMI"] = ((df["PhysActivity"] == 0) & (df["BMI"] > 30)).astype(int)
df["LogBMI"] = np.log1p(df["BMI"])
df["Income_Age"] = df["Income"] / (df["Age"] + 1)
df["DistressCombo"] = (df["MentHlth"] + df["PhysHlth"]) * (df["GenHlth"] >= 4)
df["SocioEconBurden"] = (
    (df["Income"] <= 3).astype(int)
    + (df["Education"] <= 2).astype(int)
    + (df["NoDocbcCost"] == 1).astype(int)
)
df["LowEdu"] = (df["Education"] <= 2).astype(int)
df["BMI_GenHlth"] = df["BMI"] * df["GenHlth"]
df["CardioRisk"] = df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"]

# -------------------------------------------------------------------
# Standardization
# -------------------------------------------------------------------
numeric_cols = [
    "BMI", "MentHlth", "PhysHlth", "GenHlth",
    "Age", "DistressCombo", "BMI_GenHlth",
    "Income_Age", "LogBMI"
]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------------------------------------------------------------------
# Binary Classification: Healthy vs Prediabetes
# -------------------------------------------------------------------
df_filtered = df[df["Diabetes_012"].isin([0.0, 1.0])]
df_majority = df_filtered[df_filtered["Diabetes_012"] == 0.0]
df_minority = df_filtered[df_filtered["Diabetes_012"] == 1.0]

# Downsample majority (3:1 ratio)
df_majority_downsampled = resample(
    df_majority, replace=False, n_samples=2 * len(df_minority), random_state=42
)
df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)

# Drop less relevant features
df_balanced.drop(["Fruits", "Veggies"], axis=1, inplace=True)

X = df_balanced.drop("Diabetes_012", axis=1)
y = df_balanced["Diabetes_012"]

print("Class distribution after undersampling:")
print(y.value_counts())

# -------------------------------------------------------------------
# Train-Test Split & Resampling
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train_bal, y_train_bal = SMOTETomek(random_state=42).fit_resample(X_train, y_train)

print("Class distribution after SMOTETomek:")
print(pd.Series(y_train_bal).value_counts())

# -------------------------------------------------------------------
# Model Training
# -------------------------------------------------------------------
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.005,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)
xgb_model.fit(X_train_bal, y_train_bal)

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
y_probs = xgb_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_thresh = thresholds[f1_scores.argmax()]

y_pred_thresh = (y_probs >= best_thresh).astype(int)
print("\nClassification Report:\n", classification_report(y_test, y_pred_thresh, zero_division=0))

# -------------------------------------------------------------------
# Risk Tiering
# -------------------------------------------------------------------
risk_labels = pd.Series(
    pd.cut(
        y_probs,
        bins=[0, 0.15, 0.35, 0.6, 1.0],
        labels=["Low", "Moderate", "High", "Very High"]
    ),
    index=X_test.index
)
X_test_risk = X_test.copy()
X_test_risk["PredictedRisk"] = y_probs
X_test_risk["RiskTier"] = risk_labels

# Risk Distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_probs, bins=50, kde=True, color="seagreen")
plt.title("Prediabetes Risk Distribution")
plt.xlabel("Predicted Risk Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "risk_dist.png"))
plt.close()

print("\n--- Risk Tier Distribution ---")
print(X_test_risk["RiskTier"].value_counts().sort_index())

# -------------------------------------------------------------------
# SHAP Analysis
# -------------------------------------------------------------------
explainer = shap.TreeExplainer(xgb_model, model_output="raw", feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test, check_additivity=False)

# Top 5% High-Risk Individuals
top_5 = X_test_risk.nlargest(int(0.05 * len(X_test_risk)), "PredictedRisk")
shap_values_top = explainer.shap_values(top_5[X_test.columns], check_additivity=False)

shap.summary_plot(shap_values_top, top_5[X_test.columns])
shap.summary_plot(shap_values_top, top_5[X_test.columns], plot_type="bar")

# SHAP Decision Plot (Example)
shap.decision_plot(
    explainer.expected_value,
    shap_values_top[1],
    top_5[X_test.columns].iloc[1],
    feature_order="importance"
)

# SHAP by Risk Tier (Low vs Very High)
for tier in ["Low", "Very High"]:
    tier_sample = X_test_risk[X_test_risk["RiskTier"] == tier][X_test.columns]
    if not tier_sample.empty:
        shap_values_tier = explainer.shap_values(tier_sample, check_additivity=False)
        shap.summary_plot(shap_values_tier, tier_sample, plot_type="bar")

# -------------------------------------------------------------------
# SHAP-Based Risk Promotion
# -------------------------------------------------------------------
borderline_mask = (y_probs >= 0.30) & (y_probs < 0.35)
borderline_indices = X_test.index[borderline_mask]

key_features = ["DistressCombo", "BMI_GenHlth", "CardioRisk"]
key_indices = [X_test.columns.get_loc(f) for f in key_features]
key_shap = shap_values[borderline_mask][:, key_indices]
promote_mask = key_shap.sum(axis=1) > 0.05
promoted_indices = borderline_indices[promote_mask]

risk_labels.loc[promoted_indices] = "High"
print(f"Promoted {len(promoted_indices)} borderline cases to 'High' risk tier based on SHAP impact.")

# -------------------------------------------------------------------
# Income-Based SHAP Comparison
# -------------------------------------------------------------------
low_income_mask = X_test["Income"] <= 5
high_income_mask = X_test["Income"] == 8
shap_low, shap_high = shap_values[low_income_mask], shap_values[high_income_mask]

shap.summary_plot(shap_low, X_test[low_income_mask], plot_type="bar")
shap.summary_plot(shap_high, X_test[high_income_mask], plot_type="bar")

feature_names = X_test.columns
low_df = pd.DataFrame({
    "Feature": feature_names,
    "MeanAbsSHAP_LowIncome": np.abs(shap_low).mean(axis=0)
}).sort_values("MeanAbsSHAP_LowIncome", ascending=False)

high_df = pd.DataFrame({
    "Feature": feature_names,
    "MeanAbsSHAP_HighIncome": np.abs(shap_high).mean(axis=0)
}).sort_values("MeanAbsSHAP_HighIncome", ascending=False)

print("\nTop SHAP Features for Low-Income Group:")
print(low_df.head(10))

print("\nTop SHAP Features for High-Income Group:")
print(high_df.head(10))

merged = low_df.merge(high_df, on="Feature")
merged["SHAP_Skew"] = merged["MeanAbsSHAP_LowIncome"] - merged["MeanAbsSHAP_HighIncome"]
print("\n--- SHAP Feature Skew by Income ---")
print(merged.sort_values("SHAP_Skew", ascending=False).head(10))

# --- Save Model ---
model_path = os.path.join(MODELS_DIR, "xgb_prediabetes_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(xgb_model, f)