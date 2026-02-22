"""
Model training module
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Dict, Any
import logging
import joblib

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    #Handle model training and selection
    
    def __init__(self):
        """Initialize ModelTrainer"""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
    
    def get_base_models(self) -> Dict[str, Any]:
        """
        Get dictionary of base models to train
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE),
            "SVM": SVC(probability=True, random_state=config.RANDOM_STATE)
        }
        return models
    
    def train_base_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train multiple base models
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training base models")
        models = self.get_base_models()
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=config.CROSS_VAL_FOLDS, 
                scoring='roc_auc'
            )
            mean_cv_score = cv_scores.mean()
            
            logger.info(f"{name} - CV ROC-AUC: {mean_cv_score:.4f} (+/- {cv_scores.std():.4f})")
            
            self.models[name] = {
                "model": model,
                "cv_score": mean_cv_score
            }
            
            # Track best model
            if mean_cv_score > self.best_score:
                self.best_score = mean_cv_score
                self.best_model = model
                self.best_model_name = name
        
        logger.info(f"Best base model: {self.best_model_name} (CV ROC-AUC: {self.best_score:.4f})")
        return self.models
    
    def hyperparameter_tuning(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        model_type: str = "Random Forest"
    ) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to tune
            
        Returns:
            Best model after tuning
        """
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        
        if model_type == "Random Forest":
            base_model = RandomForestClassifier(random_state=config.RANDOM_STATE)
            param_grid = config.RANDOM_FOREST_PARAMS
        elif model_type == "Logistic Regression":
            base_model = LogisticRegression(random_state=config.RANDOM_STATE)
            param_grid = config.LOGISTIC_REGRESSION_PARAMS
        elif model_type == "SVM":
            base_model = SVC(probability=True, random_state=config.RANDOM_STATE)
            param_grid = config.SVM_PARAMS
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=config.CROSS_VAL_FOLDS,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
        
        # Update best model if tuned model is better
        if grid_search.best_score_ > self.best_score:
            self.best_model = grid_search.best_estimator_
            self.best_model_name = f"{model_type} (Tuned)"
            self.best_score = grid_search.best_score_
            logger.info(f"New best model: {self.best_model_name}")
        
        return grid_search.best_estimator_
    
    def save_model(self, model: Any = None, path: str = None):

        #Save trained model to disk
        
        model = model or self.best_model
        path = path or config.MODEL_PATH
        
        if model is None:
            logger.error("No model to save")
            return
        
        # Ensure models directory exists
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path: str = None) -> Any:
       
        path = path or config.MODEL_PATH
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model


def train_pipeline(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    tune_hyperparameters: bool = True
) -> ModelTrainer:
    """
    Complete model training pipeline
    
    Args:
        X_train: Training features
        y_train: Training target
        tune_hyperparameters: Whether to perform hyperparameter tuning
        
    Returns:
        ModelTrainer instance with trained models
    """
    trainer = ModelTrainer()
    
    # Train base models
    trainer.train_base_models(X_train, y_train)
    
    # Hyperparameter tuning
    if tune_hyperparameters:
        trainer.hyperparameter_tuning(X_train, y_train, model_type="Random Forest")
    
    # Save the best model
    trainer.save_model()
    
    return trainer