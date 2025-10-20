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
import time

os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # or whatever number of cores you want

import shap
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

shap.initjs()

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
    "Gradient Boosting": os.path.join(BASE_DIR, "models", "gradientboosting_prediabetes_model.pkl")
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
    df["BMI_Outlier"] = (df["BMI"] > 50).astype(int)
    df["LowActivity_HighBMI"] = ((df["PhysActivity"] == 0) & (df["BMI"] > 30)).astype(int)
    df["LogBMI"] = np.log1p(df["BMI"])
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

    min_max_dict = {col: (X_train[col].min(), X_train[col].max()) for col in original_cols}
    return X_train, X_test, y_train, y_test, min_max_dict

X_train, X_test, y_train, y_test, min_max = preprocessing(df)

# --- Evaluate Models ---
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

results_df = evaluate_models(models, X_test, y_test, thresholds)

# --- Model Comparison Section ---
st.subheader("📊 Model Performance Comparison")
st.dataframe(results_df, use_container_width=True)

with st.expander("🧠 Why These Models?"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        These models were chosen for their strong performance on structured health data and compatibility with SHAP for interpretability:
           
        - **XGBoost**  
        A high-performance gradient boosting model known for its accuracy and speed. Widely used in clinical ML tasks due to its robustness and SHAP support.

        - **Random Forest**  
        An ensemble of decision trees that reduces overfitting and handles feature interactions well. Offers reliable performance and intuitive feature importance.

        - **Extra Trees**  
        Similar to Random Forest but uses more randomness during tree construction. Often faster and can improve generalization.

        - **HistGradientBoosting**  
        A fast, scalable boosting model optimized for large datasets. Supports missing values natively and integrates well with SHAP.

        - **Gradient Boosting**  
        A classic boosting method that builds trees sequentially to correct errors. Included for comparison with more modern variants.

        These models balance `predictive power`, `clinical transparency`, and `interpretability`, making them ideal for risk prediction in prediabetes.
        """)
    
    with col2:
        st.markdown("""
        ### 📐 Metrics Explanation

        - **Accuracy**  
        Out of all predictions made, how many were correct.

        - **Precision**  
        Out of all the positive predictions made, how many were actually true.

        - **Recall**  
        Out of all the true positive cases, how many were correctly identified.

        - **F1 Score**  
        Combines `precision` and `recall` into a single metric.  
        Useful when there's a **trade-off** between identifying positives and avoiding false alarms.

        - **Formula:**  
        **F1 Score** = 2 × (`Precision` × `Recall`) / (`Precision` + `Recall`)
        """)

# Find the best model
best_model_name = results_df["F1 Score"].idxmax()
st.success(f"🏆 Best Model: **{best_model_name}**")
best_model = models[best_model_name]

# Initialize session state if not set
if "last_selected_model" not in st.session_state:
    st.session_state.last_selected_model = best_model_name

# Model selection
selected_model_name = st.selectbox(
    "🔀 Select model for SHAP analysis",
    results_df.index.tolist(),
    index=results_df.index.tolist().index(st.session_state.last_selected_model)
)
selected_model = models[selected_model_name]

# Handle model switching
if selected_model_name != st.session_state.last_selected_model:
    with st.spinner("🔄 Switching model and recalculating SHAP values..."):
        progress_bar = st.progress(0)
        for percent_complete in range(0, 101, 10):
            time.sleep(0.1)  # Simulate work
            progress_bar.progress(percent_complete)
        progress_bar.empty()
    st.toast(f"✅ Successfully switched to **{selected_model_name}** and recalculated SHAP values!")
    time.sleep(5)
    st.session_state.last_selected_model = selected_model_name

# --- BMI Risk Classifier Section ---
yes_no_map = {"Yes": 1.0, "No": 0.0}
gender_map = {"Male": 1.0, "Female": 0.0}
education_map = {"Never Attended/Kindergaten": 1, "Grades 1-8": 2, "Grades 11-12": 3, "Grade 12/GED": 4, "College 1-3 Years": 5, 
                 "College 4 Years or More": 6}
income_map = {"< $10,000": 1, "$10,000 - < $15,000": 2, "$15,000 - < $20,000": 3, "$20,000 - < $25,000": 4, "$25,000 - < $35,000": 5, 
              "$35,000 - < $50,000": 6, "$50,000 - < $75,000": 7, "$75,000 or More": 8}

age_map = {"18 - 24": 1, "25 - 29": 2, "30 - 34": 3, "35 - 39": 4, "40 - 44": 5, "45 - 49": 6, "50 - 54": 7, "55 - 59": 8, "60 - 64": 9, 
           "65 - 69": 10, "70 - 74": 11, "75 - 79": 12, "80+": 13}

def binary_input(label, help_text=""):
    choice = st.selectbox(label, ["Yes", "No"], index=1, help=help_text)
    return yes_no_map[choice]

def gender_input(label, help_text=""):
    choice = st.selectbox(label, ["Male", "Female"], index=0, help=help_text)
    return gender_map[choice]

def education_input(label, help_text=""):
    choice = st.selectbox(label, ["Never Attended/Kindergaten", "Grades 1-8", "Grades 11-12", "Grade 12/GED", "College 1-3 Years", 
                                  "College 4 Years or More"], index=0, help=help_text)
    return education_map[choice]

def income_input(label, help_text=""):
    choice = st.selectbox(label, ["< $10,000", "$10,000 - < $15,000", "$15,000 - < $20,000", "$20,000 - < $25,000", "$25,000 - < $35,000", 
                                  "$35,000 - < $50,000", "$50,000 - < $75,000", "$75,000 or More"], index=0, help=help_text)
    return income_map[choice]

def age_input(label, help_text=""):
    choice = st.selectbox(label, ["18 - 24", "25 - 29", "30 - 34", "35 - 39", "40 - 44", "45 - 49", "50 - 54", "55 - 59", "60 - 64", "65 - 69",
                                  "70 - 74", "75 - 79", "80+"], index=0, help=help_text)
    return age_map[choice]

def compute_single_shap(model, input_df):
    model_name = type(model).__name__
    tree_models = [
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
        "XGBClassifier"
    ]

    if model_name in tree_models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        return shap_values[1][0] if isinstance(shap_values, list) else shap_values[0], input_df.columns.tolist()

    explainer = shap.Explainer(model, input_df)
    shap_values = explainer(input_df)
    return shap_values.values[0], input_df.columns.tolist()


with st.expander("🧮 Predict Risk from User Input"):
    with st.form("risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### **Demographics**")
            age = age_input("What is your age bracket?", help_text="Diabetes risk increases with age.")
            sex = gender_input("What is your sex?", help_text="Sex differences may influence risk profiles and care access.")
            education = education_input("What is your highest level of education?",
                                        help_text="Lower education may correlate with reduced health literacy and access.")
            income = income_input("What is your annual income range?", help_text="Lower income is linked to higher diabetes risk and care barriers.")

            st.markdown("##### **Health History**")
            high_bp = binary_input("Have you ever been diagnosed with high blood pressure?", help_text="This factor is linked to increased diabetes risk.")
            high_chol = binary_input("Have you ever been diagnosed with high cholesterol?", help_text="Often correlates with metabolic syndrome.")
            chol_check = binary_input("Have you had your cholesterol checked in the past 5 years?", help_text="May indicate engagement in preventive care.")
            stroke = binary_input("Have you ever had a stroke?", help_text="History of stroke may reflect underlying vascular or metabolic issues.")
            heart_disease = binary_input("Have you ever been diagnosed with heart disease or had a heart attack?",
                                         help_text="Often co-occurs with diabetes and signals elevated cardiovascular risk.")

            st.markdown("##### **Physical & Mental Health**")
            gen_health = st.slider("How would you rate your general health? (1 = Excellent → 5 = Poor)", min_value=1, max_value=5, value=1, step=1,
                                   help="Self-rated health often reflects underlying chronic conditions.")
            ment_health = st.slider("In the past 30 days, how many days was your mental health not good?", min_value=0, max_value=30, value=0, step=1,
                                    help="Mental distress can influence lifestyle and self-care behaviors.")
            phys_health = st.slider("PIn the past 30 days, how many days was your physical health not good?", min_value=0, max_value=30, value=0, step=1,
                                    help="Physical limitations may reduce activity and increase metabolic risk.")
            diff_walk = binary_input("Do you have difficulty walking or climbing stairs?",
                                     help_text="Mobility issues often correlate with obesity and cardiovascular burden.")
        with col2:
            st.markdown("##### **Lifestyle & Behaviour**")
            smoker = binary_input("Have you smoked at least 100 cigarettes in your life?",
                                  help_text="Smoking history is associated with increased risk of chronic disease.")
            phys_activity = binary_input("Do you engage in regular physical activity?",
                                         help_text="Physical inactivity is a known contributor to insulin resistance.")
            fruits = binary_input("Do you consume fruits at least once per day?",
                                  help_text="Daily fruit intake supports metabolic health and may reduce diabetes risk.")
            veggies = binary_input("Do you consume vegetables at least once per day?", help_text="Vegetable consumption is protective against chronic disease.")
            alcohol = binary_input("Do you consume alcohol heavily?", help_text="Heavy alcohol use can impair glucose regulation and liver function.")

            st.markdown("##### **Access to Care**")
            any_healthcare = binary_input("Do you have any form of health insurance or coverage?",
                                          help_text="Access to care influences early detection and management of diabetes.")
            no_doc_cost = binary_input("Have you ever avoided seeing a doctor due to cost?", help_text="May indicate barriers to preventive care.")

            st.markdown("##### **Body Metrics**")
            bmi = st.slider("What is your Body Mass Index (BMI)?", min_value=int(min_max["BMI"][0]), max_value=int(min_max["BMI"][1]), value=25, step=1, 
                            help="Higher BMI is a strong predictor of metabolic and cardiovascular risk.")

            submitted = st.form_submit_button("##### **Predict Risk**")

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
        input_df["BMI_Outlier"] = (input_df["BMI"] > 50).astype(int)
        input_df["LowActivity_HighBMI"] = ((input_df["PhysActivity"] == 0) & (input_df["BMI"] > 30)).astype(int)
        input_df["LogBMI"] = np.log1p(input_df["BMI"])
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
        st.subheader("📈 Estimated Risk of Prediabetes")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Risk Score",
                value=f"{risk_score:.3f}",
                help="""
                    **Risk Tiering:**
                    - 🟢 **Low**: 0.00 - 0.15  
                    - 🟡 **Moderate**: 0.16 - 0.35  
                    - 🟠 **High**: 0.36 - 0.60  
                    - 🔴 **Very High**: 0.61 - 1.00  

                    This score is not a diagnosis, it's a model-based estimate to explore how lifestyle, health history, 
                    and access to care relate to prediabetes risk.
                    """)

            st.markdown("##### **Risk Tier:**")
            if risk_label == "Low":
                st.markdown("""
                    **🟢 Low Risk**
                    - This profile suggests a low estimated risk of prediabetes.
                    - Maintain healthy habits:
                        - Regular physical activity  
                        -  Balanced nutrition  
                        - Preventive care engagement
                    """)
            elif risk_label == "Moderate":
                st.markdown("""
                    **🟡 Moderate Risk**
                    - This profile indicates a moderate estimated risk of prediabetes.
                    - Consider reviewing lifestyle factors:
                        - Physical activity  
                        - Dietary habits  
                        - Access to preventive care
                    - Small changes can have a meaningful impact.
                    """)
            elif risk_label == "High":
                st.markdown("""
                    **🟠 High Risk**
                    - This profile reflects a high estimated risk of prediabetes.
                    - Multiple factors may be contributing:
                        - Elevated BMI  
                        - Blood pressure concerns  
                        - Limited physical activity
                    - Exploring these areas with a healthcare provider could be beneficial.
                    """)
            else:
                st.markdown("""
                    **🔴 Very High Risk**
                    - This profile suggests a very high estimated risk of prediabetes.
                    - Several risk factors may be interacting.
                    - It may be helpful to:
                        - Engage with a clinician  
                        - Discuss personalized prevention strategies  
                        - Explore early intervention options
                    """)

        with col2:    
            shap_contribs, feature_names = compute_single_shap(models[selected_model_name], input_df)
            top_indices = np.argsort(np.abs(shap_contribs))[::-1][:5]

            st.markdown("##### 🔍 Top Contributors to Your Risk Score")
            st.markdown("ℹ️ _Positive contributions raise the predicted risk; negative ones lower it._")

            feature_aliases = {"BMI_Outlier": "BMI > 50", "LowActivity_HighBMI": "Low Activity + High BMI",
                            "LogBMI": "Log-transformed BMI", "DistressCombo": "Mental + Physical Distress (if poor health)",
                            "SocioEconBurden": "Low Income + Low Education + Cost Barrier", "LowEdu": "Low Education",
                            "BMI_GenHlth": "BMI × General Health", "CardioRisk": "High BP + Cholesterol + Heart Disease"}

            for idx in top_indices:
                feature = feature_names[idx]
                shap_val = shap_contribs[idx]
                input_val = input_df.iloc[0][feature]
                direction = "increased" if shap_val > 0 else "decreased"
                emoji = "🔺" if shap_val > 0 else "🔻"

                display_name = feature_aliases.get(feature, feature)
                st.markdown(f"""
                    - {emoji} **{display_name}** contributed **{shap_val:+.3f}** to your risk → This {direction} risk due to a value of **{input_val}**
                    """)




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

    📊 **In this dashboard**, SHAP values show how your input features (like `BMI`, `Income`, `Age`, etc.) influence the predicted price — positively or negatively.

    """)

# --- SHAP Interpretability Section ---
with st.expander("📈 SHAP Summary & Feature Importance"):
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)

    def compute_shap_values(model, X_sample):
        """Robust SHAP computation across tree and non-tree models."""
        model_name = type(model).__name__

        tree_models = [
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
            "XGBClassifier"
        ]

        if model_name in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_array = np.mean(np.array(shap_values), axis=0)
            else:
                shap_array = shap_values

            return shap_array, X_sample.columns.tolist()

        try:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            return shap_values, X_sample.columns.tolist()
        except Exception as e:
            st.error(f"SHAP explainer failed: {e}")
            return None, X_sample.columns.tolist()

    shap_values, feature_names = compute_shap_values(selected_model, X_sample)

    if shap_values is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### SHAP Summary Plot")
            fig_summary, ax = plt.subplots(figsize=(6, 4))
            try:
                shap.summary_plot(
                    shap_values,
                    X_sample.values,
                    show=False,
                    plot_size=(10, 8),
                    feature_names=np.array(feature_names)
                )
                st.pyplot(fig_summary)
            except Exception as e:
                st.error(f"SHAP summary plot failed: {e}")
            plt.close()
        with st.expander("ℹ️ What do these plots show?"):
            st.markdown("""
            **Summary Plot**  
            - **Color** = feature value (**red** = `high`, **blue** = `low`, **purple** = `mid`)  
            - **Position** = SHAP value (**left** = `negative`, **right** = `positive`)  
            - **Density** = importance across samples  
            ---
            **Feature Importance Plot**  
            - Ranks features by average absolute SHAP value  
            - **Longer** bars = `higher` influence  
            - Helps identify top drivers of prediction
            """)

        with col2:
            st.markdown("##### SHAP Feature Importance")
            fig_bar, ax = plt.subplots(figsize=(6, 4))
            try:
                raw_values = shap_values.values if hasattr(shap_values, "values") else shap_values
                shap.summary_plot(
                    np.abs(raw_values),
                    X_sample.values,
                    show=False,
                    plot_type="bar",
                    plot_size=(10, 8),
                    feature_names=np.array(feature_names)
                )
                st.pyplot(fig_bar)
            except Exception as e:
                st.error(f"SHAP feature importance plot failed: {e}")
            plt.close()

# --- SHAP Dependence & Decision Section ---
with st.expander("🔍 SHAP Dependence & Decision Analysis"):
    col1, col2 = st.columns(2)

    # --- Dependence Plot ---
    with col1:
        st.markdown("##### SHAP Dependence Plot")

        display_labels = [col for col in X_sample.columns]

        # Session state setup
        if "selected_feature" not in st.session_state:
            st.session_state.selected_feature = display_labels[0]
        if "color_feature" not in st.session_state:
            st.session_state.color_feature = display_labels[1]

        selected_label = st.selectbox("Feature to analyze", display_labels, index=0)
        color_label = st.selectbox("Color by feature", display_labels, index=1)

        selected_feature = selected_label
        color_feature = color_label

        st.session_state.selected_feature = selected_label
        st.session_state.color_feature = color_label

        fig_dep, ax = plt.subplots(figsize=(5, 3))
        try:
            shap.dependence_plot(
                selected_feature,
                shap_values.values if hasattr(shap_values, "values") else shap_values,
                X_sample,
                interaction_index=color_feature,
                show=False,
                ax=ax
            )
            st.pyplot(fig_dep)
        except Exception as e:
            st.error(f"Dependence plot failed: {e}")
        plt.close()

    # --- Decision Plot ---
    
    with col2:
        st.markdown("##### SHAP Decision Plot")
        sample_index = st.slider("Select test sample index", 0, len(X_sample) - 1, 0)
        try:
            fig_decision, ax = plt.subplots(figsize=(6, 4))
            shap.decision_plot(
                base_value=shap_values.base_values[sample_index] if hasattr(shap_values, "base_values") else np.mean(shap_values),
                shap_values=shap_values[sample_index],
                feature_names=list(X_sample.columns),
                feature_order="importance",
                show=False
            )
            st.pyplot(fig_decision)
        except Exception as e:
            st.error(f"Decision plot failed: {e}")
        plt.close()
    
    with st.expander("🩺 Feature Values for Selected Sample"):
        sample_data = X_sample.iloc[sample_index]
        sample_df = pd.DataFrame({
            "Feature": sample_data.index,
            "Value": sample_data.values
        })
        st.dataframe(sample_df)
    
    with st.expander("ℹ️ What do these plots show?"):
        st.markdown("""
        **Dependence Plot**  
        - Shows how a single feature’s value affects its SHAP contribution  
        - `X-axis` = feature value  
        - `Y-axis` = SHAP value (impact on prediction)  
        - `Color` = interaction with another feature  
        - Reveals non-linear effects and feature interactions
        ---
        **Waterfall Plot**  
        - Breaks down how each feature pushes the prediction from the base value  
        - **Left to right** = cumulative SHAP contributions  
        - Highlights the most influential features for a single prediction  
        - Great for explaining individual risk scores
        """)

# --- Confusion Matrix Section
st.subheader("🔢 Confusion Matrix")
with st.expander("📊 Confusion Matrix - Model Performance"):
    model = models[selected_model_name]
    threshold = thresholds.get(selected_model_name, 0.5)

    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        # Fallback to decision_function or direct prediction
        try:
            y_pred_score = model.decision_function(X_test)
            y_pred = (y_pred_score >= threshold).astype(int)
        except AttributeError:
            # fallback to hard prediction
            y_pred = model.predict(X_test)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        This matrix shows how well the model distinguishes between positive and negative cases.  
        - **True Positives (`TP`)**: Correctly predicted positives  
        - **True Negatives (`TN`)**: Correctly predicted negatives  
        - **False Positives (`FP`)**: Predicted positive but actually negative  
        - **False Negatives (`FN`)**: Predicted negative but actually positive
        """)

        with st.expander("📊 Performance Metrics"):
            st.markdown(f"""
            ### **Overall Metrics**
            - **Accuracy**: `{acc * 100:.2f}%`
            - **Precision**: `{prec * 100:.2f}%`
            - **Recall (Sensitivity)**: `{rec * 100:.2f}%`
            - **F1 Score**: `{f1 * 100:.2f}%`

            ---

            ### **Confusion Matrix Breakdown**
            - **True Positives (`TP`)**: `{tp}`
            - **True Negatives (`TN`)**: `{tn}`
            - **False Positives (`FP`)**: `{fp}`
            - **False Negatives (`FN`)**: `{fn}`
            """)

    with col2:
        cm = confusion_matrix(y_test, y_pred)

        fig_cm, ax = plt.subplots(figsize=(4, 3))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_cm, use_container_width=False)

st.subheader("🛠️ Preprocessing & Modeling")
with st.expander("⚙️ How the Models Were Built"):
    st.markdown("""
    This dashboard is powered by a modular pipeline designed for clinical clarity, reproducibility, and interpretability.

    ### 🔄 Preprocessing Strategy
    - **Dataset**: CDC Diabetes Health Indicators (BRFSS 2015)  
      [View dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
    - **Target**: Binary classification — Healthy (0) vs Prediabetes (1)
    - **Balancing**:  
      - Downsampled majority class to 2:1 ratio  
      - Applied SMOTETomek to training set for class balance
    - **Feature Engineering**:  
      - Composite features: `DistressCombo`, `SocioEconBurden`, `CardioRisk`  
      - Log transforms, interaction terms, and outlier flags  
      - Dropped low-impact features (`Fruits`, `Veggies`)

    ---
                
    ### 🧠 Modeling Approach
    - Trained six models:  
      `XGBoost`, `Random Forest`, `Extra Trees`, `HistGradientBoosting`, `Gradient Boosting`, `AdaBoost`
    - Tuned hyperparameters for depth, learning rate, and ensemble size
    - Selected optimal thresholds using precision-recall curves and F1 maximization
    - Saved models and thresholds for dashboard deployment

    ---
                
    ### 📊 Evaluation & Risk Stratification
    - Benchmarked models on test set using F1, precision, recall
    - Assigned risk tiers: `Low`, `Moderate`, `High`, `Very High` based on predicted probabilities
    - Promoted borderline cases using SHAP impact from key features

    ---
                
    ### 🔍 Interpretability with SHAP
    - Used TreeExplainer for SHAP analysis across models
    - Visualized feature impact by risk tier and income level
    - Compared SHAP skew between low-income and high-income groups

    This pipeline ensures that predictions are not only accurate, but also explainable and clinically meaningful.
    """)

# --- Download SHAP Values ---
st.subheader("📥 Download SHAP Values")
shap_df = pd.DataFrame(shap_values, columns=feature_names)
csv = shap_df.to_csv(index=False).encode("utf-8")
st.download_button("Download SHAP values as CSV", data=csv, file_name=f"{selected_model_name}_shap_values_prediabetes.csv", mime="text/csv")

# --- Streamlit Footer ---
st.markdown("""
<hr style="margin-top: 50px;">

<div style='text-align: center; font-size: 0.9em; color: gray;'>
    Built by Nicholas Laprade · 
    <a href='https://www.linkedin.com/in/nicholas-laprade/' target='_blank'>LinkedIn</a> · 
    <a href='https://github.com/nlaprade' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)