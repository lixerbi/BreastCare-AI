"""
Prediction module
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

from src import config
from src.train import ModelTrainer
from src.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """Handle predictions with trained model"""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize Predictor
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        self.model_path = model_path or config.MODEL_PATH
        self.scaler_path = scaler_path or config.SCALER_PATH
        
        self.model = None
        self.scaler = None
        
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model and scaler"""
        try:
            self.model = ModelTrainer.load_model(self.model_path)
            self.scaler = DataPreprocessor.load_scaler(self.scaler_path)
            logger.info("Model and scaler loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Artifact not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def predict_single(self, features: List[float]) -> Dict[str, Any]:
        """
        Make prediction for a single sample
        
        Args:
            features: List of feature values
            
        Returns:
            Dictionary containing prediction results
        """
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        malignancy_probability = probabilities[1]
        benign_probability = probabilities[0]
        
        # Get risk level
        risk_level = self._get_risk_level(malignancy_probability)
        
        result = {
            "prediction": int(prediction),
            "prediction_label": "Malignant" if prediction == 1 else "Benign",
            "malignancy_probability": float(malignancy_probability),
            "benign_probability": float(benign_probability),
            "risk_level": risk_level,
            "confidence": float(max(probabilities))
        }
        
        return result
    
    def predict_batch(self, features_list: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples
        
        Args:
            features_list: List of feature lists
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for features in features_list:
            result = self.predict_single(features)
            results.append(result)
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a DataFrame
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            DataFrame with predictions added
        """
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Add predictions to dataframe
        df_result = df.copy()
        df_result['prediction'] = predictions
        df_result['prediction_label'] = df_result['prediction'].map({0: 'Benign', 1: 'Malignant'})
        df_result['malignancy_probability'] = probabilities[:, 1]
        df_result['benign_probability'] = probabilities[:, 0]
        df_result['risk_level'] = df_result['malignancy_probability'].apply(self._get_risk_level)
        
        return df_result
    
    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """
        Convert probability to risk level
        
        Args:
            probability: Malignancy probability
            
        Returns:
            Risk level string
        """
        if probability < config.LOW_RISK_THRESHOLD:
            return "Low Risk"
        elif probability < config.HIGH_RISK_THRESHOLD:
            return "Medium Risk"
        else:
            return "High Risk"