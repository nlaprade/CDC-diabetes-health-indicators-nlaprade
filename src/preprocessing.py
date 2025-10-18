"""
Author: Nicholas Laprade
Date: 2025-10-17
Topic: CDC Diabetes Health Indicators - Preprocessing/Data Exploration
Dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
"""

import os

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "diabetes_012_health_indicators_BRFSS2015.csv")
graphs_dir = os.path.join(BASE_DIR, "graphs")

# --- Read CSV ---
df = pd.read_csv(data_path)

#print(df.head())
#print(df.info())

"""
RangeIndex: 253680 entries, 0 to 253679
Data columns (total 22 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   Diabetes_012          253680 non-null  float64 | 0=healthy 1=prediabetes 2=diabetes
 1   HighBP                253680 non-null  float64 | 0=normal 1=high 
 2   HighChol              253680 non-null  float64 | 0=normal 1=high
 3   CholCheck             253680 non-null  float64 | 0=no check 1=yes check
 4   BMI                   253680 non-null  float64 | body mass index
 5   Smoker                253680 non-null  float64 | at least 100 cigarettes total life; 0=no 1=yes
 6   Stroke                253680 non-null  float64 | 0=no 1=yes
 7   HeartDiseaseorAttack  253680 non-null  float64 | 0=no 1=yes
 8   PhysActivity          253680 non-null  float64 | physical activity in past 30 days; 0=no 1=yes
 9   Fruits                253680 non-null  float64 | eat fruit one or more times per day; 0=no 1=yes
 10  Veggies               253680 non-null  float64 | eat veggie one or more times per day; 0=no 1=yes
 11  HvyAlcoholConsump     253680 non-null  float64 | more than 14 drinks (men) 7 drinks (women) per week; 0=no 1=yes
 12  AnyHealthcare         253680 non-null  float64 | 0=no 1=yes
 13  NoDocbcCost           253680 non-null  float64 | needed to see doctor past 12 months; 0=no 1=yes
 14  GenHlth               253680 non-null  float64 | general health; 1=excellent 2=very good 3=good 4=fair 5=poor
 15  MentHlth              253680 non-null  float64 | how many days during the past 30 days was your mental health not good
 16  PhysHlth              253680 non-null  float64 | how many days during the past 30 days was your physical health not good
 17  DiffWalk              253680 non-null  float64 | 0=no 1=yes
 18  Sex                   253680 non-null  float64 | 0=female 1=yes
 19  Age                   253680 non-null  float64 | 1=18-24 2=25-29 3=30-34 4=35-39 5=40-44 6=45-49 7=50-54 8=55-59 9=60-64 10=65-69 11=70-74 12=75-79 13=80+ 14=NAN
 20  Education             253680 non-null  float64 | 1=never attended 2=grade1-8 3=grade9-11 4=12/GED 5=college1-3 6=college4+
 21  Income                253680 non-null  float64 | 1=<10,000 5=<35,000 8=75,000+
dtypes: float64(22)
"""

corr_data = df.corr(numeric_only=True)

plt.figure(figsize=(20, 14))
sns.heatmap(corr_data, cmap="YlGnBu", annot=True)

heatmap_path = os.path.join(graphs_dir, "heatmap.png")
plt.savefig(heatmap_path)
plt.close()
#plt.show()

#correlations = df.corr(numeric_only=True)["Diabetes_012"]
#print(correlations)

"""
Diabetes_012            1.000000 | Skip
HighBP                  0.271596 | High Pos!
HighChol                0.209085 | High Pos!
CholCheck               0.067546 | Low Pos
BMI                     0.224379 | High Pos!
Smoker                  0.062914 | Low Pos
Stroke                  0.107179 | Low Pos
HeartDiseaseorAttack    0.180272 | High Pos!
PhysActivity           -0.121947 | High Neg!
Fruits                 -0.042192 | Low Neg
Veggies                -0.058972 | Low Neg
HvyAlcoholConsump      -0.057882 | Low Neg
AnyHealthcare           0.015410 | Low Pos
NoDocbcCost             0.035436 | Low Pos
GenHlth                 0.302587 | High Pos!
MentHlth                0.073507 | Low Pos
PhysHlth                0.176287 | High Pos!
DiffWalk                0.224239 | High Pos!
Sex                     0.031040 | Low Pos
Age                     0.185026 | High Pos!
Education              -0.130517 | High Neg!
Income                 -0.171483 | High Neg!
Name: Diabetes_012, dtype: float64

Thoughs on correlations:
- Focusing on both high pos and high neg correlations.
- See if I can combine similar low correlation columns (fruits/veggies, smoker/stroke, MentlHlth/HvyAlcoholConsump)
"""

"""
for col in df.columns:
    unique_vals = df[col].unique()
    print(f"{col}: {unique_vals}")
"""

plt.figure(figsize=(10, 6))
sns.histplot(df["BMI"], bins=30, kde=True, color="skyblue")
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.tight_layout()
bmi_dist_path = os.path.join(graphs_dir, "bmi_dist.png")
plt.savefig(bmi_dist_path)
plt.close()
#plt.show()

# --- Feature Engineering ---
df["HealthyDiet"] = df["Fruits"] + df["Veggies"]
df["LifeStyleRisk"] = df["Smoker"] + df["Stroke"]
df["MentalALchFlag"] = ((df["MentHlth"] > 7) | (df["HvyAlcoholConsump"] == 1)).astype(int)

df["HighBP_HDA"] = df["HighBP"] + df["HeartDiseaseorAttack"]
df["BMI_Age"] = df["BMI"] * df["Age"]
df["GenHlth_Phys"] = df["GenHlth"] * df["PhysActivity"]

# --- Splitting Data ---
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# --- Convert to binary: combine prediabetes (1.0) with diabetes (2.0) ---
y_binary = y.replace({1.0: 1.0, 2.0: 1.0})

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)

# --- Apply SMOTE ---
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# --- Train XGBoost ---
neg_count = (y_binary == 0).sum()
pos_count = (y_binary == 1).sum()
scale_pos_weight = neg_count / pos_count


xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=12,
    scale_pos_weight=scale_pos_weight,  # Try different values (e.g., 3–10)
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train_bal, y_train_bal)

# --- Predict & Evaluate ---
y_pred = xgb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Apply custom threshold
y_probs = xgb_model.predict_proba(X_test)[:, 1]
y_pred_thresh = (y_probs >= 0.65).astype(int)

# Re-evaluate
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred_thresh))
print("\nClassification Report:\n", classification_report(y_test, y_pred_thresh, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_thresh)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Threshold = 0.65)")
plt.show()

