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
st.caption("Built for clinical insight, model benchmarking, and transparent feature attribution.")

with st.expander("📂 Dataset Information"):
    st.markdown("""
    - The **Diabetes Health Indicators Dataset** is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), originally compiled by the CDC through the Behavioral Risk Factor Surveillance System (BRFSS).

    - It contains survey-based health statistics and lifestyle factors from over 250,000 U.S. adults, including metrics like BMI, physical activity, mental health, and socioeconomic indicators.  
    
    - This dashboard focuses on binary classification between **healthy individuals (class 0)** and those with **prediabetes (class 1)**, using engineered features to explore risk stratification and model interpretability.
    """)

    with st.expander("🧪 Engineered Features"):
        st.markdown("""
        - **BMI_Outlier**: Flags extreme BMI values beyond ±3 standard deviations.
        - **LowActivity_HighBMI**: Identifies individuals with no physical activity and BMI > 30.
        - **LogBMI**: Applies log transformation to BMI for normalization.
        - **Income_Age**: Ratio of income to age, capturing economic maturity.
        - **DistressCombo**: Combines mental and physical health burden, weighted by poor general health.
        - **SocioEconBurden**: Composite score of low income, low education, and cost-related care avoidance.
        - **LowEdu**: Flags individuals with education level ≤ 2.
        - **BMI_GenHlth**: Interaction term between BMI and general health rating.
        - **CardioRisk**: Sum of cardiovascular risk indicators (high BP, high cholesterol, heart disease).
        """)

with st.expander("📘 Dataset Column Description"):
    st.markdown("""
**Dtypes summary**: `float64(22)`

| #  | Column               | Dtype    | Description                                                 |
|----|----------------------|----------|-------------------------------------------------------------|
| 0  | Diabetes_012         | float64  | Indicator for diabetes (0 = No, 1 = Prediabetes, 2 = Yes)   |
| 1  | HighBP               | float64  | High blood pressure (1 = Yes, 0 = No)                       |
| 2  | HighChol             | float64  | High cholesterol (1 = Yes, 0 = No)                          |
| 3  | CholCheck            | float64  | Had cholesterol checked in past 5 years (1 = Yes, 0 = No)   |
| 4  | BMI                  | float64  | Body Mass Index (weight in kg / height in m²)               |
| 5  | Smoker               | float64  | Current smoker (1 = Yes, 0 = No)                            |
| 6  | Stroke               | float64  | History of stroke (1 = Yes, 0 = No)                         |
| 7  | HeartDiseaseorAttack | float64  | History of heart disease or heart attack (1 = Yes, 0 = No)  |
| 8  | PhysActivity         | float64  | Physically active (1 = Yes, 0 = No)                         |
| 9  | Fruits               | float64  | Consumes fruits at least once per day (1 = Yes, 0 = No)     |
| 10 | Veggies              | float64  | Consumes vegetables at least once per day (1 = Yes, 0 = No) |
| 11 | HvyAlcoholConsump    | float64  | Heavy alcohol consumption (1 = Yes, 0 = No)                 |
| 12 | AnyHealthCare        | float64  | Has any form of health care coverage (1 = Yes, 0 = No)      |
| 13 | NoDocbcCost          | float64  | Could not see a doctor due to cost (1 = Yes, 0 = No)        |
| 14 | GenHlth              | float64  | General health rating (1 = Excellent → 5 = Poor)            |
| 15 | MentHlth             | float64  | Days mental health was not good in past 30 days             |
| 16 | PhysHlth             | float64  | Days physical health was not good in past 30 days           |
| 17 | DiffWalk             | float64  | Difficulty walking or climbing stairs (1 = Yes, 0 = No)     |
| 18 | Sex                  | float64  | Gender (0 = Male, 1 = Female)                               |
| 19 | Age                  | float64  | Age in years                                                |
| 20 | Education            | float64  | Education level (1 = None → 6 = College graduate)           |
| 21 | Income               | float64  | Annual income (1 = <$10k → 8 = ≥$75k)                       |
""")

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

original_cols = [
        "Age", "BMI", "MentHlth", "PhysHlth", "GenHlth",
        "Education", "Income"
    ]

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
    
    min_max_dict = {col: (X_train_bal[col].min(), X_train_bal[col].max()) for col in original_cols}
    return X_train_bal, X_test, y_train_bal, y_test, min_max_dict

X_train_bal, X_test, y_train_bal, y_test, min_max = preprocessing(df)

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

# --- BMI Risk Classifier Section ---
yes_no_map = {"Yes": 1.0, "No": 0.0}
gender_map = {"Male": 1.0, "Female": 0.0}

def binary_input(label, help_text=""):
    choice = st.selectbox(label, ["Yes", "No"], index=1, help=help_text)
    return yes_no_map[choice]

def gender_input(label, help_text=""):
    choice = st.selectbox(label, ["Male", "Female"], index=0, help=help_text)
    return gender_map[choice]

with st.expander("🧮 Predict Risk from User Input"):
    with st.form("risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Demographics**")
            age_options = list(range(int(min_max["Age"][0]), int(min_max["Age"][1]) + 1))
            age = st.selectbox("What is your age bracket (code)", age_options, index=4)

            with st.expander("Age Bracket Reference"):
                st.markdown("""
                | Code | Age Range     |
                |------|---------------|
                | 1    | 18-24         |
                | 2    | 25-29         |
                | 3    | 30-34         |
                | 4    | 35-39         |
                | 5    | 40-44         |
                | 6    | 45-49         |
                | 7    | 50-54         |
                | 8    | 55-59         |
                | 9    | 60-64         |
                | 10   | 65-69         |
                | 11   | 70-74         |
                | 12   | 75-79         |
                | 13   | 80+           |
                """)
            sex = gender_input("What is your sex?")
            education = st.selectbox("What is your highest level of education?", list(np.arange(1.0, 7.0)))

            with st.expander("Education Bracket Reference"):
                st.markdown("""
                | Code | Education Level             |
                |------|-----------------------------|
                | 1    | Never attended/kindergarten |
                | 2    | Grades 1 through 8          |
                | 3    | Grades 9 through 11         |
                | 4    | Grade 12 or GED             |
                | 5    | College 1 year to 3 years   |
                | 6    | College 4 years or more     |
                """)
            income = st.selectbox("What is your annual income range?", list(np.arange(1.0, 9.0)))
            
            with st.expander("Income Bracket Reference"):
                st.markdown("""
                | Code | Income Range Bracket |
                |------|----------------------|
                | 1    | < $10,000            |
                | 2    | $10,000 - <$15,000   |
                | 3    | $15,000 - <$20,000   |
                | 4    | $20,000 - <$25,000   |
                | 5    | $25,000 - <$35,000   |
                | 6    | $35,000 - <$50,000   |
                | 7    | $50,000 - <$75,000   |
                | 8    | $75,000 or more      |
                """)
            st.markdown("**Health History**")
            high_bp = binary_input("Have you ever been diagnosed with high blood pressure?")
            high_chol = binary_input("Have you ever been diagnosed with high cholesterol?")
            chol_check = binary_input("Have you had your cholesterol checked in the past 5 years?")
            stroke = binary_input("Have you ever had a stroke?")
            heart_disease = binary_input("Have you ever been diagnosed with heart disease or had a heart attack?")

        with col2:
            st.markdown("**Physical & Mental Health**")
            gen_health = st.selectbox("How would you rate your general health? (1 = Excellent → 5 = Poor)", [1.0, 2.0, 3.0, 4.0, 5.0], index=2)
            ment_health = st.slider("In the past 30 days, how many days was your mental health not good?", min_value=0, max_value=30, value=5, step=1)
            phys_health = st.slider("PIn the past 30 days, how many days was your physical health not good?", min_value=0, max_value=30, value=5, step=1)
            diff_walk = binary_input("Do you have difficulty walking or climbing stairs?", "Difficulty walking or climbing stairs")

            st.markdown("**Lifestyle & Behaviour**")
            smoker = binary_input("Have you smoked at least 100 cigarettes in your life?")
            phys_activity = binary_input("Do you engage in regular physical activity?")
            fruits = binary_input("Do you consume fruits at least once per day?")
            veggies = binary_input("Do you consume vegetables at least once per day?")
            alcohol = binary_input("Do you consume alcohol heavily?")

            st.markdown("**Access to Care**")
            any_healthcare = binary_input("Do you have any form of health insurance or coverage?")
            no_doc_cost = binary_input("Have you ever avoided seeing a doctor due to cost?")

            st.markdown("**Body Metrics**")
            bmi = st.slider("What is your Body Mass Index (BMI)?", min_value=int(min_max["BMI"][0]), max_value=int(min_max["BMI"][1]), value=25, step=1)

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_df = pd.DataFrame([{
            "HighBP": high_bp,
            "HighChol": high_chol,
            "CholCheck": chol_check,
            "BMI": bmi,
            "Smoker": smoker,
            "Stroke": stroke,
            "HeartDiseaseorAttack": heart_disease,
            "PhysActivity": phys_activity,
            "Fruits": fruits,
            "Veggies": veggies,
            "HvyAlcoholConsump": alcohol,
            "AnyHealthcare": any_healthcare,
            "NoDocbcCost": no_doc_cost,
            "GenHlth": gen_health,
            "MentHlth": ment_health,
            "PhysHlth": phys_health,
            "DiffWalk": diff_walk,
            "Sex": sex,
            "Age": age,
            "Education": education,
            "Income": income
        }])

        # Apply engineered features
        input_df["BMI_Outlier"] = (np.abs(input_df["BMI"]) > 3).astype(int)
        input_df["LowActivity_HighBMI"] = ((input_df["PhysActivity"] == 0) & (input_df["BMI"] > 30)).astype(int)
        input_df["LogBMI"] = np.log1p(input_df["BMI"])
        input_df["Income_Age"] = input_df["Income"] / (input_df["Age"] + 1)
        input_df["DistressCombo"] = (input_df["MentHlth"] + input_df["PhysHlth"]) * (input_df["GenHlth"] >= 4)
        input_df["SocioEconBurden"] = ((input_df["Income"] <= 3).astype(int) + (input_df["Education"] <= 2).astype(int) + (input_df["NoDocbcCost"] == 1).astype(int))
        input_df["LowEdu"] = (input_df["Education"] <= 2).astype(int)
        input_df["BMI_GenHlth"] = input_df["BMI"] * input_df["GenHlth"]
        input_df["CardioRisk"] = input_df["HighBP"] + input_df["HighChol"] + input_df["HeartDiseaseorAttack"]

        input_df.drop(["Fruits", "Veggies"], axis=1, inplace=True)

        # Predict
        model = models[selected_model_name]
        threshold = thresholds.get(selected_model_name, 0.5)
        risk_score = model.predict_proba(input_df)[0][1]
        risk_label = pd.cut([risk_score], bins=[0, 0.15, 0.35, 0.6, 1.0], labels=["Low", "Moderate", "High", "Very High"])[0]

        st.metric(label="Predicted Risk Score", value=f"{risk_score:.3f}")
        st.success(f"🩺 Risk Tier: **{risk_label}**")

# --- Shap Explain Section ---
st.subheader("📊 SHAP Interpretability")
with st.expander("❗ What are SHAP Values?"):
    st.markdown("""
    **SHAP** (**SH**apley **A**dditive ex**P**lanations) is a powerful method for interpreting machine learning models. It assigns each feature a contribution value showing how much that feature pushed the prediction up or down.

    🔍 **Why use SHAP?**
    - It helps you understand *why* a model made a specific prediction.
    - It reveals which features are most influential for each individual prediction.
    - It supports trust and transparency in ML systems — especially important for stakeholders and decision-makers.

    🧠 **How does it work?**
    SHAP is based on game theory. Imagine each feature as a player in a game, and the prediction as the payout. SHAP calculates how much each feature contributes to the final prediction by comparing all possible combinations of features.

    📊 **In this dashboard**, SHAP values show how your input features (like BMI, Income, Age, etc.) influence the predicted price — positively or negatively.

    """)
