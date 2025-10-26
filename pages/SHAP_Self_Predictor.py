import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import pickle

from utils.paths import DATA_PATH, MODEL_DIR, IMAGES_DIR
from utils.data_utils import preprocessing, load_data
from utils.model_utils import load_all_models
from utils.predictor_utils import (
    binary_input, gender_input, education_input, income_input, age_input, compute_single_shap
)
from utils.metrics_utils import render_model_change

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

shap.initjs()

# --- Page Setup ---
st.set_page_config(page_title="Individual Prediction", page_icon=os.path.join(IMAGES_DIR, "icon.png"), layout="wide")
st.title("Individual Risk Prediction")
st.caption("Enter your health profile to receive a personalized risk prediction.")

# --- Load Models ---
model_paths = {
    "XGBoost": os.path.join(MODEL_DIR, "xgboost_prediabetes_model.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "randomforest_prediabetes_model.pkl"),
    "Extra Trees": os.path.join(MODEL_DIR, "extratrees_prediabetes_model.pkl"),
    "HistGradientBoosting": os.path.join(MODEL_DIR, "histgb_prediabetes_model.pkl"),
    "Gradient Boosting": os.path.join(MODEL_DIR, "gradientboosting_prediabetes_model.pkl")
}
models = load_all_models(model_paths)
current_model = st.session_state.get("current_model", next(iter(models)))
# --- Load Data & Preprocessing ---
df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test, min_max = preprocessing(df)

# --- Load Thresholds ---
threshold_path = os.path.join(MODEL_DIR, "thresholds.pkl")

if os.path.exists(threshold_path):
    with open(threshold_path, "rb") as f:
        thresholds = pickle.load(f)
else:
    thresholds = {name: 0.5 for name in models}

# --- Initialize session state ---
if "current_model" not in st.session_state:
    st.session_state.current_model = list(models.keys())[0]

if "temp_model" not in st.session_state:
    st.session_state.temp_model = st.session_state.current_model

if "model_switch_triggered" not in st.session_state:
    st.session_state.model_switch_triggered = False

# --- Callback to track selection change ---
def on_model_change():
    st.session_state.model_switch_triggered = True

render_model_change(models)

with st.container():
    with st.form("risk_form"):
        with st.container(border=True):
            st.markdown("### 🧍 Demographics")
            st.caption("These questions help assess social determinants linked to diabetes risk.")
            age = age_input("What is your age bracket?", help_text="Diabetes risk increases with age.")
            sex = gender_input("What is your sex?", help_text="Sex differences may influence risk profiles and care access.")
            education = education_input("What is your highest level of education?", help_text="Lower education may correlate with reduced health literacy and access.")
            income = income_input("What is your annual income range?", help_text="Lower income is linked to higher diabetes risk and care barriers.")

        with st.container(border=True):
            st.markdown("### 🩺 Health History")
            st.caption("These questions explore medical conditions commonly associated with elevated diabetes risk.")
            high_bp = binary_input("Have you ever been diagnosed with high blood pressure?", help_text="This factor is linked to increased diabetes risk.")
            high_chol = binary_input("Have you ever been diagnosed with high cholesterol?", help_text="Often correlates with metabolic syndrome.")
            chol_check = binary_input("Have you had your cholesterol checked in the past 5 years?", help_text="May indicate engagement in preventive care.")
            stroke = binary_input("Have you ever had a stroke?", help_text="History of stroke may reflect underlying vascular or metabolic issues.")
            heart_disease = binary_input("Have you ever been diagnosed with heart disease or had a heart attack?", help_text="Often co-occurs with diabetes and signals elevated cardiovascular risk.")
        
        with st.container(border=True):
            st.markdown("### 💪 Physical & Mental Health")
            st.caption("Self-rated health and recent symptoms help identify underlying risk factors.")
            gen_health = st.slider("How would you rate your general health? (1 = Excellent → 5 = Poor)", min_value=1, max_value=5, value=1, step=1,
                                help="Self-rated health often reflects underlying chronic conditions.")
            ment_health = st.slider("In the past 30 days, how many days was your mental health not good?", min_value=0, max_value=30, value=0, step=1,
                                help="Mental distress can influence lifestyle and self-care behaviors.")
            phys_health = st.slider("PIn the past 30 days, how many days was your physical health not good?", min_value=0, max_value=30, value=0, step=1,
                                help="Physical limitations may reduce activity and increase metabolic risk.")
            diff_walk = binary_input("Do you have difficulty walking or climbing stairs?",
                                help_text="Mobility issues often correlate with obesity and cardiovascular burden.")

        with st.container(border=True):
            st.markdown("### 🏃 Lifestyle & Behaviour")
            st.caption("Daily habits and behaviors play a major role in metabolic health.")
            smoker = binary_input("Have you smoked at least 100 cigarettes in your life?", help_text="Smoking history is associated with increased risk of chronic disease.")
            phys_activity = binary_input("Do you engage in regular physical activity?", help_text="Physical inactivity is a known contributor to insulin resistance.")
            fruits = binary_input("Do you consume fruits at least once per day?", help_text="Daily fruit intake supports metabolic health and may reduce diabetes risk.")
            veggies = binary_input("Do you consume vegetables at least once per day?", help_text="Vegetable consumption is protective against chronic disease.")
            alcohol = binary_input("Do you consume alcohol heavily?", help_text="Heavy alcohol use can impair glucose regulation and liver function.")

        with st.container(border=True):
            st.markdown("### 🏥 Access to Care")
            st.caption("Access and affordability of care affect early detection and disease management.")
            any_healthcare = binary_input("Do you have any form of health insurance or coverage?", help_text="Access to care influences early detection and management of diabetes.")
            no_doc_cost = binary_input("Have you ever avoided seeing a doctor due to cost?", help_text="May indicate barriers to preventive care.")

        with st.container(border=True):
            st.markdown("### ⚖️ Body Metrics")
            st.caption("BMI is a strong predictor of metabolic and cardiovascular risk.")
            bmi = st.slider("What is your Body Mass Index (BMI)?", min_value=int(min_max["BMI"][0]), max_value=int(min_max["BMI"][1]), value=25, step=1, help="Higher BMI is a strong predictor of metabolic and cardiovascular risk.")

        submitted = st.form_submit_button("**Predict Risk**")

st.markdown("""
    <style>
    div[data-testid="stForm"] {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        background: transparent !important;
    }</style>
""", unsafe_allow_html=True)

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
    model = models[current_model]
    threshold = thresholds.get(current_model, 0.5)

    risk_score = model.predict_proba(input_df)[0][1]
    risk_label = pd.cut([risk_score], bins=[0, 0.15, 0.35, 0.6, 1.0], labels=["Low", "Moderate", "High", "Very High"])[0]

    tab1, tab2 = st.tabs(["📈 Estimated Risk of Prediabetes", "🔍 Top Contributors to Your Risk Score"])

    with tab1:
        st.markdown("### 🧠 Predicted Risk Overview")
        st.markdown("This section summarizes your model-based risk score and its clinical interpretation.")

        # --- Score + Tier side-by-side ---
        score_col, tier_col = st.columns([1, 1])

        
        st.metric(
                label="Predicted Risk Score",
                value=f"{risk_score:.3f}",
                help="""
                    **What this score means:**  
                    This is a model-based estimate based on your health indicators.  
                    It reflects how lifestyle, history, and access to care relate to prediabetes risk.  

                    **Risk Tiering:**  
                    - 🟢 **Low**: 0.00 - 0.15  
                    - 🟡 **Moderate**: 0.16 - 0.35  
                    - 🟠 **High**: 0.36 - 0.60  
                    - 🔴 **Very High**: 0.61 - 1.00  
                    """)
        
        if risk_label == "Low":
            st.success("🟢 **Low Risk**\n\nThis profile suggests a low estimated risk of prediabetes.\n\nMaintain healthy habits like regular activity, balanced nutrition, and preventive care.")
        elif risk_label == "Moderate":
            st.warning("🟡 **Moderate Risk**\n\nThis profile indicates a moderate estimated risk.\n\nConsider reviewing lifestyle factors such as activity, diet, and access to care. Small changes can have meaningful impact.")
        elif risk_label == "High":
            st.warning("🟠 **High Risk**\n\nThis profile reflects a high estimated risk.\n\nMultiple factors may be contributing — elevated BMI, blood pressure, or inactivity. A healthcare provider can help explore these areas.")
        else:
            st.error("🔴 **Very High Risk**\n\nThis profile suggests a very high estimated risk.\n\nSeveral risk factors may be interacting. Consider engaging with a clinician to discuss personalized prevention and early intervention.")

        st.markdown("---")

    with tab2:
        shap_contribs, feature_names = compute_single_shap(models[current_model], input_df)
        
        if shap_contribs.ndim == 2:
            if shap_contribs.shape[0] == len(feature_names) and shap_contribs.shape[1] == models[current_model].n_classes_:
                # Transpose to (n_classes, n_features)
                shap_contribs = shap_contribs.T
                pred_class = models[current_model].predict(input_df)[0]
                shap_contribs = shap_contribs[pred_class]
            elif shap_contribs.shape[0] == 1:
                shap_contribs = shap_contribs.flatten()
            else:
                st.error(f"Unexpected SHAP shape: {shap_contribs.shape}")
                st.stop()

        top_indices = np.argsort(np.abs(shap_contribs))[::-1][:5]
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
            emoji = "👎" if shap_val > 0 else "👍"
            color = "#fa0000" if shap_val > 0 else "#35ff02"
            display_name = feature_aliases.get(feature, feature)
            st.markdown(f"""
<span style="color:{color}">
{emoji} <strong>{display_name}</strong> contributed <strong>{shap_val:+.3f}</strong> to your risk → This increased risk due to a value of <strong>{input_val}</strong>
</span>
""", unsafe_allow_html=True)


        st.markdown("---")
with st.expander("Want to Learn More?"):
    st.markdown("Diabetes/Prediabetes Resources")
    st.markdown("[CDC: Prediabetes Basics](https://www.cdc.gov/diabetes/prevention-type-2/prediabetes-prevent-type-2.html)", unsafe_allow_html=True)
    st.markdown("[WHO: Diabetes Overview](https://www.who.int/news-room/fact-sheets/detail/diabetes)", unsafe_allow_html=True)
    st.markdown("[American Diabetes Association](https://diabetes.org/)", unsafe_allow_html=True)
    st.markdown("[Diabetes Canada: Prediabetes](https://www.diabetes.ca/about-diabetes/prediabetes-1)", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Built by Nicholas Laprade — [LinkedIn](https://www.linkedin.com/in/nicholas-laprade) • [GitHub](https://github.com/nlaprade)")