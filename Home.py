"""
Author: Nicholas Laprade
Date: 2025-10-19
Topic: CDC Diabetes Health Indicators - Dashboard
Dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
"""

import streamlit as st
import pickle
import os
import shap

from utils.paths import DATA_PATH, MODEL_DIR, IMAGES_DIR
from utils.data_utils import load_data, preprocessing
from utils.model_utils import load_all_models
from utils.metrics_utils import compute_confusion_metrics, plot_confusion_matrix, render_threshold_slider
from sklearn.metrics import classification_report

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

shap.initjs()

# --- Streamlit Setup ---
st.set_page_config(
    page_title="CDC Prediabetes Classification Dashboard",
    page_icon=os.path.join(IMAGES_DIR, "icon.png"),
    layout="wide"
)

st.title("CDC Prediabetes Classification Dashboard")
st.caption("Built for clinical insight, model benchmarking, and transparent feature attribution.")

# --- Load Models ---
model_paths = {
    "XGBoost": os.path.join(MODEL_DIR, "xgboost_prediabetes_model.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "randomforest_prediabetes_model.pkl"),
    "Extra Trees": os.path.join(MODEL_DIR, "extratrees_prediabetes_model.pkl"),
    "HistGradientBoosting": os.path.join(MODEL_DIR, "histgb_prediabetes_model.pkl"),
    "Gradient Boosting": os.path.join(MODEL_DIR, "gradientboosting_prediabetes_model.pkl")
}
models = load_all_models(model_paths)

# --- Load Data ---
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

if "thresholds" not in st.session_state:
    st.session_state.thresholds = thresholds.copy()

# Ensure slider is initialized with the correct model threshold
if "threshold_slider" not in st.session_state:
    current_model = st.session_state.current_model
    st.session_state.threshold_slider = st.session_state.thresholds.get(current_model, 0.5)
# --- Callback to track selection change ---

def on_model_change():
    st.session_state.model_switch_triggered = True

# --- Sidebar Content ---
with st.sidebar:
    st.subheader("Model Configuration")

    st.selectbox(
        "Choose Model",
        list(models.keys()),
        index=list(models.keys()).index(st.session_state.temp_model),
        key="model_selector",
        on_change=on_model_change
    )

    if st.session_state.model_switch_triggered:
        st.session_state.temp_model = st.session_state.model_selector

    if st.session_state.temp_model != st.session_state.current_model:
        st.warning("⚠️ Switching models may take time on cloud-hosted dashboards.")
        confirm_switch = st.button("✅ Confirm Model Switch")
        cancel_switch = st.button("⛔ Cancel Model Change")

        if confirm_switch:
            st.session_state.current_model = st.session_state.temp_model
            st.session_state.model_switch_triggered = False
            st.toast(f"✅ Switched to {st.session_state.current_model}")
            st.rerun()

        elif cancel_switch:
            st.session_state.temp_model = st.session_state.current_model
            st.session_state.model_switch_triggered = False
            st.toast("⛔ Model switch cancelled")
            st.rerun()

# --- Threshold Configuration ---
st.markdown("## 🔧 Threshold Configuration")
st.info("""
Changing the threshold means changing the sensitivity.  
- **Lower Threshold** → more samples classified as class 1 *(higher recall, lower precision)*  
- **Higher Threshold** → fewer samples classified as class 1 *(lower recall, higher precision)*
""")

render_threshold_slider(thresholds)

# --- Use the confirmed model and threshold ---
model = models[st.session_state.current_model]
threshold = st.session_state.thresholds[st.session_state.current_model]

# --- Tabs Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance",
    "📁 Dataset Info",
    "🧪 Feature Engineering",
    "💽 Model Overview",
    "TEST"
])

# --- Tab 1: Model Performance ---
with tab1:
    st.subheader("🔢 Confusion Matrix")
    metrics = compute_confusion_metrics(model, X_test, y_test, threshold)
    breakdown = metrics["breakdown"]

    # --- Initialize session state ---
    if "show_raw_metrics" not in st.session_state:
        st.session_state.show_raw_metrics = False

    # --- Toggle button logic ---
    def toggle_metrics():
        st.session_state.show_raw_metrics = not st.session_state.show_raw_metrics

    st.button(
        f"Switch to {'Raw Counts' if not st.session_state.show_raw_metrics else 'Performance Metrics'}",
        on_click=toggle_metrics
    )

    # --- Layout ---
    col1, col2 = st.columns([1.5, 2.5])
    with col1:
        st.markdown("### Confusion Matrix Breakdown")
        if not st.session_state.show_raw_metrics:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            st.metric("Precision", f"{metrics['precision']:.2%}")
            st.metric("Recall", f"{metrics['recall']:.2%}")
            st.metric("F1 Score", f"{metrics['f1']:.2%}")
            st.metric("Specificity", f"{metrics['specificity']:.2%}")
        else:
            st.metric("True Positives (TP)", breakdown["TP"])
            st.metric("True Negatives (TN)", breakdown["TN"])
            st.metric("False Positives (FP)", breakdown["FP"])
            st.metric("False Negatives (FN)", breakdown["FN"])

    with col2:
         fig_cm = plot_confusion_matrix(metrics["confusion_matrix"])
         st.pyplot(fig_cm, width='content')

# --- Tab 2: Dataset Info ---
with tab2:
    with st.container(border=True):
        st.markdown("""
        - Source: [UCI Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
        - Over 250,000 U.S. adults surveyed
        - Binary classification: Healthy (0) vs Prediabetes (1)
        """)

    with st.expander("📘 Column Descriptions"):
        st.markdown("""
        | #  | Column               | Description                                                 |
        |----|----------------------|-------------------------------------------------------------|
        | 0  | Diabetes_012         | 0 = No, 1 = Prediabetes, 2 = Yes                            |
        | 1  | HighBP               | High blood pressure (1 = Yes, 0 = No)                       |
        | 2  | HighChol             | High cholesterol (1 = Yes, 0 = No)                          |
        | 3  | CholCheck            | Cholesterol check in past 5 years                          |
        | 4  | BMI                  | Body Mass Index                                             |
        | 5  | Smoker               | Current smoker                                              |
        | 6  | Stroke               | History of stroke                                           |
        | 7  | HeartDiseaseorAttack | Heart disease or attack history                            |
        | 8  | PhysActivity         | Physically active                                           |
        | 9  | Fruits               | Eats fruit daily                                            |
        | 10 | Veggies              | Eats vegetables daily                                       |
        | 11 | HvyAlcoholConsump    | Heavy alcohol consumption                                   |
        | 12 | AnyHealthCare        | Has health care coverage                                    |
        | 13 | NoDocbcCost          | Avoided doctor due to cost                                  |
        | 14 | GenHlth              | General health rating (1 = Excellent → 5 = Poor)            |
        | 15 | MentHlth             | Days mental health was poor                                 |
        | 16 | PhysHlth             | Days physical health was poor                               |
        | 17 | DiffWalk             | Difficulty walking                                          |
        | 18 | Sex                  | Gender (0 = Male, 1 = Female)                               |
        | 19 | Age                  | Age in years                                                |
        | 20 | Education            | Education level (1 = None → 6 = College graduate)           |
        | 21 | Income               | Income level (1 = <$10k → 8 = ≥$75k)                        |
        """)

# --- Tab 3: Feature Engineering ---
with tab3:
    with st.container(border=True):
        st.markdown("""
            These engineered features were designed to enhance clinical relevance, capture key risk interactions, and improve model interpretability.  
            Each one reflects a meaningful combination or transformation of raw inputs tailored for prediabetes prediction.
            """)
        st.markdown("""
        - **BMI_Outlier**: Flags extreme BMI values beyond ±3 standard deviations  
        - **LowActivity_HighBMI**: No physical activity and BMI > 30  
        - **LogBMI**: Log-transformed BMI for normalization  
        - **Income_Age**: Ratio of income to age  
        - **DistressCombo**: Weighted combo of mental + physical health burden  
        - **SocioEconBurden**: Composite of low income, low education, and cost-related care avoidance  
        - **LowEdu**: Flags education level ≤ 2  
        - **BMI_GenHlth**: Interaction between BMI and general health rating  
        - **CardioRisk**: Sum of cardiovascular risk indicators  
        """)

# --- Model Overview ---
with tab4:
    with  st.container(border=True):
        st.markdown("""
            These models were chosen for their strong performance on structured health data and compatibility with SHAP for interpretability:

            - **`XGBoost`**  
            A high-performance gradient boosting model known for its accuracy and speed. Widely used in clinical machine learning tasks due to its robustness and SHAP support.

            - **`Random Forest`**  
            An ensemble of decision trees that reduces overfitting and handles feature interactions well. Offers reliable performance and intuitive feature importance.

            - **`Extra Trees`**  
            Similar to Random Forest but uses more randomness during tree construction. Often faster and can improve generalization.

            - **`HistGradientBoosting`**  
            A fast, scalable boosting model optimized for large datasets. Supports missing values natively and integrates well with SHAP.

            - **`Gradient Boosting`**  
            A classic boosting method that builds trees sequentially to correct errors. Included for comparison with more modern variants.
            """)

    with st.container(border=True):
        st.markdown("""
            ### Why These Models?
            These models balance `predictive power`, `clinical transparency`, and `interpretability`, making them ideal for risk prediction in prediabetes.  
            Each was tuned for depth, learning rate, and ensemble size using cross-validation.

            - Thresholds were selected using precision-recall curves and F1 maximization  
            - SHAP values were used to audit decision boundaries and promote borderline cases  
            - Models were saved with calibrated thresholds for deployment stability
            """)

    with st.container(border=True):
        st.markdown("""
            ### Deployment Strategy
            - All models were wrapped in a modular pipeline for reproducibility  
            - SHAP explanations are rendered live for both global and local interpretability  
            - Risk tiers (`Low`, `Moderate`, `High`, `Very High`) are assigned based on predicted probabilities and SHAP impact
            """)

    with st.container(border=True):
        st.markdown("""
            ### Model Evaluation Highlights
            - **`Accuracy`** Consistently high across folds, with XGBoost and HistGradientBoosting leading  
            - **`F1 Score`** Optimized for borderline cases to reduce false negatives in high-risk groups  
            - **`Calibration`** Probability outputs checked for clinical reliability and interpretability  
            - **`SHAP Insights`** Top predictors include BMI, Age, and Blood Pressure, which aligns with clinical evidence

            This approach ensures that predictions are not only accurate, but also explainable and clinically meaningful.
            """)

    with st.container(border=True):
        st.markdown("""
            ### Model Documentation
            - [XGBoost](https://xgboost.readthedocs.io/en/latest/)
            - [Random Forest (sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
            - [Extra Trees (sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
            - [HistGradientBoosting (sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
            - [Gradient Boosting (sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
            """)

from sklearn.metrics import classification_report

with tab5:
    st.markdown("### 🏁 Model Benchmark")

    # Collect metrics per model
    benchmark_rows = []
    results = {}
    for name, model in models.items():
        threshold = thresholds.get(name, 0.5)
        metrics = compute_confusion_metrics(model, X_test, y_test, threshold=threshold)
        benchmark_rows.append([
            name,
            f"{metrics['accuracy']:.2%}",
            f"{metrics['precision']:.2%}",
            f"{metrics['recall']:.2%}",
            f"{metrics['f1']:.2%}"
        ])
        results[name] = {"recall": metrics["recall"]}

    # Render as Markdown table (flush-left formatting)
    table_header = (
        "| Model Name           | Accuracy | Precision | Recall | F1 Score |\n"
        "|----------------------|----------|-----------|--------|----------|"
    )
    table_rows = "\n".join([
        f"| {row[0]:<20} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |"
        for row in benchmark_rows
    ])
    st.markdown(table_header + "\n" + table_rows)
    best_score = max(results, key=lambda k: results[k]["recall"])
    best_recall = results[best_score]["recall"]

    st.success(f"🏆 Best Model by Recall: **{best_score}** ({best_recall:.2%})")

    with st.expander("❓ Why Only Measure Recall?"):
        with st.container():
            st.markdown("""
                ### 🎯 Why Focus on Recall

                In this dashboard, we prioritize **recall** because our goal is to identify as many individuals at risk for prediabetes as possible.  
                This may include flagging some false positives, but it ensures we catch nearly all true cases.
                """)

        with st.container():
            st.markdown("""
                #### ✅ What Recall Measures

                - Recall quantifies how many actual prediabetes cases the model successfully detects.  
                - A high recall means fewer false negatives. We avoid missing people who truly need further evaluation.
                """)

        with st.container():
            st.markdown("""
                #### 🧠 Why That Matters in Tiered Risk Systems

                - This dashboard uses a **tiering system** after the initial prediction to stratify risk into categories such as low, moderate, and high.  
                - By maximizing recall, we ensure that:
                    - Potential cases are not missed at the first stage.  
                    - Downstream logic, such as SHAP-based risk buckets or clinical review, can refine the signal.  
                - In preventive medicine, missing a true positive is more costly than flagging a few extra false positives.
                """)

        with st.container():
            st.markdown("""
                #### 📊 Precision Tradeoff Is Acceptable

                - Precision may decrease, meaning more false positives.  
                - This is a strategic tradeoff. The tiering system acts as a **filter**, helping clinicians or downstream logic reclassify borderline cases with more context.
                """)


# --- Footer ---
st.markdown("---")
st.caption("Built by Nicholas Laprade — [LinkedIn](https://www.linkedin.com/in/nicholas-laprade) • [GitHub](https://github.com/nlaprade)")