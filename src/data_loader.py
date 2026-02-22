"""
Data loading and validation module
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, data_path: Path = None):

        self.data_path = data_path or config.DATASET_PATH
        
    def load_data(self) :
        #Load breast cancer dataset from CSV and returns dataFrame containing the raw data
        
        try:
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully-Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame):
        
        required_columns = [config.TARGET_COLUMN]
        
        # Checking if required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Checking for empty dataframe
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        # Checking target column values
        if config.TARGET_COLUMN in df.columns:
            unique_values = df[config.TARGET_COLUMN].unique()
            logger.info(f"Target column unique values: {unique_values}")
        
        logger.info("Data validation passed")
        return True
    
    def get_data_info(self, df: pd.DataFrame) :

        #Get basic information about the dataset returns a dictionary containing dataset information
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[config.TARGET_COLUMN].value_counts().to_dict() if config.TARGET_COLUMN in df.columns else None
        }
        return info


def load_and_validate_data():
    loader = DataLoader()
    df = loader.load_data()
    if not loader.validate_data(df):
        raise ValueError("Data validation failed")
    
    return df