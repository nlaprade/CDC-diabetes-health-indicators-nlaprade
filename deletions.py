### REMOVE
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
### REMOVE

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