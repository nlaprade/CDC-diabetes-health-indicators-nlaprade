import streamlit as st
import shap

# --- Dictionaries ---
yes_no_map = {"Yes": 1.0, "No": 0.0}
gender_map = {"Male": 1.0, "Female": 0.0}
education_map = {"Never Attended/Kindergaten": 1, "Grades 1-8": 2, "Grades 11-12": 3, "Grade 12/GED": 4, "College 1-3 Years": 5, 
                 "College 4 Years or More": 6}
income_map = {"< $10,000": 1, "$10,000 - < $15,000": 2, "$15,000 - < $20,000": 3, "$20,000 - < $25,000": 4, "$25,000 - < $35,000": 5, 
              "$35,000 - < $50,000": 6, "$50,000 - < $75,000": 7, "$75,000 or More": 8}

age_map = {"18 - 24": 1, "25 - 29": 2, "30 - 34": 3, "35 - 39": 4, "40 - 44": 5, "45 - 49": 6, "50 - 54": 7, "55 - 59": 8, "60 - 64": 9, 
           "65 - 69": 10, "70 - 74": 11, "75 - 79": 12, "80+": 13}

# --- Input Functions ---
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
