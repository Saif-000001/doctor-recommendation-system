from pathlib import Path

class Settings:
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = DATA_DIR / "models"
    
    # Data files
    DOCTOR_DATA_PATH = DATA_DIR / "Doctor_Versus_Disease.csv"
    DESCRIPTION_DATA_PATH = DATA_DIR / "Disease_Description.csv"

    # Model
    LOGISTIC_REGRESSION_MODEL = MODEL_DIR / "Logistic_Regression_model.pkl"
    DECISION_TREE_MODEL = MODEL_DIR / "Decision_Tree_model.pkl"
    RANDOM_FOREST_MODEL = MODEL_DIR / "Random_Forest_model.pkl"
    SVM_MODEL = MODEL_DIR / "SVM_model.pkl"
    NAIVE_BAYES_MODEL = MODEL_DIR / "Naive_Bayes_model.pkl"
    KNN_MODEL = MODEL_DIR / "KNN_model.pkl"

    
    # API settings
    API_TITLE = "Disease Prediction API"
    API_DESCRIPTION = "API for predicting diseases based on symptoms"
    API_VERSION = "1.0.0"

settings = Settings()


                