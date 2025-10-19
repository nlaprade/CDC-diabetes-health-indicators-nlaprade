"""
Author: Nicholas Laprade
Date: 2025-10-17
Topic: CDC Diabetes Health Indicators - Preprocessing/Data Exploration
Dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
"""

import os
import shap
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "diabetes_012_health_indicators_BRFSS2015.csv")
graphs_dir = os.path.join(BASE_DIR, "graphs")
models_dir = os.path.join(BASE_DIR, "models")

# --- Read CSV ---
df = pd.read_csv(data_path)

# --- Correlation Heatmap ---
corr_data = df.corr(numeric_only=True)
plt.figure(figsize=(20, 14))
sns.heatmap(corr_data, cmap="YlGnBu", annot=True)
plt.savefig(os.path.join(graphs_dir, "heatmap.png"))
plt.close()

# --- BMI Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(df["BMI"], bins=30, kde=True, color="skyblue")
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "bmi_dist.png"))
plt.close()

# --- Feature Engineering ---
df["BMI_Outlier"] = (np.abs(df["BMI"]) > 3).astype(int)
df["LowActivity_HighBMI"] = ((df["PhysActivity"] == 0) & (df["BMI"] > 30)).astype(int)
df["LogBMI"] = np.log1p(df["BMI"])
df["Income_Age"] = df["Income"] / (df["Age"] + 1)
df["DistressCombo"] = (df["MentHlth"] + df["PhysHlth"]) * (df["GenHlth"] >= 4)
df["SocioEconBurden"] = (df["Income"] <= 3).astype(int) + (df["Education"] <= 2).astype(int) + (df["NoDocbcCost"] == 1).astype(int)
df["LowEdu"] = (df["Education"] <= 2).astype(int)
df["BMI_GenHlth"] = df["BMI"] * df["GenHlth"]
df["CardioRisk"] = df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"]



# --- Standardize non-binary numeric features ---
numeric_cols = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income", "DistressCombo"]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# --- Binary Classification: Healthy vs Prediabetes ---
df_filtered = df[df["Diabetes_012"].isin([0.0, 1.0])]
df_majority = df_filtered[df_filtered["Diabetes_012"] == 0.0]
df_minority = df_filtered[df_filtered["Diabetes_012"] == 1.0]

# 3:1 Undersampling
df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=4 * len(df_minority),
    random_state=42
)

df_balanced = pd.concat([df_majority_downsampled, df_minority])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced.drop("Sex", axis=1, inplace=True)
X = df_balanced.drop("Diabetes_012", axis=1)
y = df_balanced["Diabetes_012"]

print("Class distribution after undersampling:")
print(y.value_counts())

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- SMOTETomek Resampling ---
X_train_bal, y_train_bal = SMOTETomek(random_state=42).fit_resample(X_train, y_train)

print("Class distribution after SMOTETomek:")
print(pd.Series(y_train_bal).value_counts())

# --- Train XGBoost Model ---
xgb_model = XGBClassifier(
    n_estimators=5000,
    learning_rate=0.001,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train_bal, y_train_bal)

# --- Predict Probabilities ---
y_probs = xgb_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_recall_curve, f1_score

precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)  # avoid div by zero
best_thresh = thresholds[f1_scores.argmax()]
#print("Optimal threshold for max F1:", best_thresh)

# --- Threshold Tuning ---
threshold = best_thresh
y_pred_thresh = (y_probs >= threshold).astype(int)

print(f"\n--- Threshold @ {threshold} ---")
print("Accuracy:", accuracy_score(y_test, y_pred_thresh))
print("\nClassification Report:\n", classification_report(y_test, y_pred_thresh, zero_division=0))

# --- Risk Tiering ---
risk_labels = pd.cut(
    y_probs,
    bins=[0, 0.2, 0.4, 0.7, 1.0],
    labels=["Low", "Moderate", "High", "Very High"]
)

X_test_risk = X_test.copy()
X_test_risk["PredictedRisk"] = y_probs
X_test_risk["RiskTier"] = risk_labels

#print("Low-risk individuals (<0.2):", (y_probs < 0.2).sum())

# --- Visualize Risk Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(y_probs, bins=50, kde=True, color="seagreen")
plt.title("Prediabetes Risk Distribution")
plt.xlabel("Predicted Risk Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "risk_dist.png"))
plt.close()

# --- SHAP Analysis ---
explainer = shap.TreeExplainer(xgb_model)

top_5_percent = X_test_risk.sort_values("PredictedRisk", ascending=False).head(int(0.05 * len(X_test_risk)))
top_sample = top_5_percent.drop(columns=["PredictedRisk", "RiskTier"])
shap_values_top = explainer.shap_values(top_sample, check_additivity=False)

shap.summary_plot(shap_values_top, top_sample)
shap.summary_plot(shap_values_top, top_sample, plot_type="bar")

i = 1  # Index of high-risk individual
shap.decision_plot(
    explainer.expected_value,
    shap_values_top[i],
    top_sample.iloc[i],
    feature_order='importance'
)

#print("\n--- Feature Stats for High-Risk Individual ---")
#print(top_sample.iloc[i])

# --- SHAP by Risk Tier ---
tiers = ["Low", "Very High"]
for tier in tiers:
    tier_sample = X_test_risk[X_test_risk["RiskTier"] == tier].drop(columns=["PredictedRisk", "RiskTier"])
    if len(tier_sample) > 0:
        shap_values_tier = explainer.shap_values(tier_sample, check_additivity=False)
        #print(f"\n--- SHAP Summary for {tier} Risk Tier ---")
        shap.summary_plot(shap_values_tier, tier_sample, plot_type="bar")

# --- Risk Tier Summary ---
tier_counts = X_test_risk["RiskTier"].value_counts().sort_index()
#print("\n--- Risk Tier Distribution ---")
print(tier_counts)

"""
fn = (y_test == 1) & (y_pred_thresh == 0)
fp = (y_test == 0) & (y_pred_thresh == 1)

shap_fn = explainer.shap_values(X_test[fn], check_additivity=False)
shap_fp = explainer.shap_values(X_test[fp], check_additivity=False)

flip_score = np.abs(np.mean(np.abs(shap_fn), axis=0) - np.mean(np.abs(shap_fp), axis=0))
top_flippers = np.argsort(flip_score)[-10:]

feature_names = X_test.columns
flipper_names = feature_names[top_flippers]
print("Top flip features:", flipper_names.tolist())

for idx in top_flippers:
    plt.figure(figsize=(8, 4))
    plt.hist(shap_fn[:, idx], bins=30, alpha=0.6, label="False Negatives", color="red")
    plt.hist(shap_fp[:, idx], bins=30, alpha=0.6, label="False Positives", color="blue")
    plt.title(f"SHAP Contrast: {feature_names[idx]}")
    plt.xlabel("SHAP Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
"""

# --- Save Model ---
model_path = os.path.join(models_dir, "xgb_prediabetes_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(xgb_model, f)
