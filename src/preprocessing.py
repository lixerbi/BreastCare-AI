"""
Data preprocessing and feature engineering module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging
import joblib

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    #Handle data preprocessing and feature engineering
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def clean_data(self, df: pd.DataFrame):

        logger.info("Starting data cleaning")
        df_clean = df.copy()

        # Drop unnecessary columns
        cols_to_drop = [col for col in config.DROP_COLUMNS if col in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns: {cols_to_drop}")
        
        # Convert target to numeric (M=1, B=0)
        if config.TARGET_COLUMN in df_clean.columns:
            df_clean[config.TARGET_COLUMN] = df_clean[config.TARGET_COLUMN].map({"M": 1, "B": 0})
            logger.info("Target variable mapped: M->1, B->0")
        
        # Handle missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")
            # For numerical columns, fill with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        #Separate features and target returns tuple of (features, target)

        X = df.drop(columns=[config.TARGET_COLUMN])
        y = df[config.TARGET_COLUMN]
        
        self.feature_names = X.columns.tolist()
        config.FEATURE_NAMES = self.feature_names
        
        logger.info(f"Features prepared: {len(self.feature_names)} features")
        return X, y
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = None, 
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or config.TEST_SIZE
        random_state = random_state or config.RANDOM_STATE
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Train class distribution:\n{y_train.value_counts()}")
        logger.info(f"Test class distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        logger.info("Scaling features")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, path: str = None):
        
        path = path or config.SCALER_PATH
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")
    
    @staticmethod
    def load_scaler(path: str = None) -> StandardScaler:
        path = path or config.SCALER_PATH
        scaler = joblib.load(path)
        logger.info(f"Scaler loaded from {path}")
        return scaler


def preprocess_pipeline(df: pd.DataFrame) -> Tuple:
    #Complete preprocessing pipeline ,Args, df- Raw DataFrame 
    # returns tuple of (X_train_scaled, X_test_scaled, y_train, y_test, preprocessor)
    preprocessor = DataPreprocessor()
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Prepare features
    X, y = preprocessor.prepare_features(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor