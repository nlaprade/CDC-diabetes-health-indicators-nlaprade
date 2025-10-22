import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

original_cols = [
    "Age", "BMI", "MentHlth", "PhysHlth", "GenHlth",
    "Education", "Income"
]

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df

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
