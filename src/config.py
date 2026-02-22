import os
from pathlib import Path

"""
Configuration file for the Breast Cancer Detection System
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "breast_cancer_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Dataset file
DATASET_PATH = RAW_DATA_DIR / "breast-cancer.csv"

# Column names
TARGET_COLUMN = "diagnosis"  # M (Malignant) or B (Benign)
ID_COLUMN = "id"

# Columns to drop
DROP_COLUMNS = ["id", "Unnamed: 32"]  # Unnamed: 32 is empty column in dataset


# Train/test split
TEST_SIZE = 0.2  # 20% for testing, 80% for training
RANDOM_STATE = 42  # For reproducibility
CROSS_VAL_FOLDS = 5  # 5-fold cross-validation

# HYPERPARAMETERS FOR MODELS
# Random Forest hyperparameters for tuning
RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200, 300],  # Number of trees
    "max_depth": [None, 5, 10, 15],  # Tree depth
    "min_samples_split": [2, 5],  # Min samples to split node
    "min_samples_leaf": [1, 2]  # Min samples in leaf
}

# Logistic Regression hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "C": [0.01, 0.1, 1, 10],  # Regularization strength
    "penalty": ["l2"],  # Regularization type
    "max_iter": [5000]  # Max iterations
}

# SVM hyperparameters
SVM_PARAMS = {
    "C": [0.1, 1, 10],  # Regularization
    "kernel": ["rbf", "linear"],  # Kernel type
    "gamma": ["scale", "auto"]  # Kernel coefficient
}

# API CONFIGURATION

API_HOST = "0.0.0.0"  # Listen on all interfaces
API_PORT = 8000  # Default port
API_TITLE = "Breast Cancer Detection API"
API_DESCRIPTION = "AI-powered early breast cancer risk detection system"
API_VERSION = "1.0.0"


# Probability thresholds for risk levels
LOW_RISK_THRESHOLD = 0.3  # Below 30% = Low Risk
HIGH_RISK_THRESHOLD = 0.7  # Above 70% = High Risk
# Between 30-70% = Medium Risk
FEATURE_NAMES = []  # Will be filled by preprocessing module