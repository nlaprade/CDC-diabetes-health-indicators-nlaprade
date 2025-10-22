import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # points to project root
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_012_health_indicators_BRFSS2015.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "assets")