
# Diabetes Risk Prediction and SHAP Analysis Dashboard

An interactive dashboard for predicting prediabetes risk using machine learning and SHAP-based interpretability. Designed for clarity, modular benchmarking, and user-centered transparency.

## Description

This projects showcases a robust machine learning pipeline and dashboard for estimating diabetes risk based on user-defined health indicators. It emphasizes interpretability-first modeling, risk tiering, and clean UX.

## Dataset

The dataset used in this project contains over 250,000 health records from the Behavioral Risk Factor Surveillance System (BRFSS).

- **Source**: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)  
- **License**: Publicly available for educational and non-commercial use  
- **Preprocessing**: Feature engineering, outlier flags, SMOTETomek balancing, undersampling

## Features

- Manual input prediction with real-time feedback
- SHAP summary, dependence, and decision plots
- Model comparison with precision, recall, and F1
- Threshold tuning per model
- Expandable metrics cards and confusion matrix breakdown
- Clean UI with relevant tooltips and grouped inputs

## Installation


```bash
  git clone https://github.com/nlaprade/CDC-diabetes-health-indicators.git
  cd CDC-diabetes-health-indicators
  pip install -r requirements.txt
```
    
## Tools Stack

| Tool         | Purpose                                      |
|--------------|----------------------------------------------|
| **Streamlit**| Interactive dashboard and UI                 |
| **scikit-learn** | Model training, evaluation, preprocessing |
| **XGBoost**  | Ensemble modeling and SHAP-based interpretability |
| **Random Forest** | Tree-based modeling and feature impact analysis |
| **Extra Trees**| High-variance ensemble modeling and SHAP integration |
| **GradientBoosting** | Boosted modeling and interpretability via SHAP|
| **SHAP**     | Model interpretability and feature contribution plots    |
| **pandas**   | Data manipulation and feature engineering              |
| **matplotlib** | Visualization of SHAP values and performance metrics                  |
| **pickle** | Saving and loading trained models and thresholds |
| **SMOTETomek**| Resampling for class balance during training |

All dependencies are listed in `requirements.txt`.

## Authors

Nicholas Laprade 
- [LinkedIn](https://www.linkedin.com/in/nicholas-laprade/)
- [GitHub](https://github.com/nlaprade)


