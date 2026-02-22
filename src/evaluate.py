"""
Model evaluation module
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict
import logging

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handle model evaluation and metrics"""
    
    def __init__(self, model: Any):
        """
        Initialize ModelEvaluator
        
        Args:
            model: Trained model to evaluate
        """
        self.model = model
        self.metrics = {}
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {self.metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {self.metrics['roc_auc']:.4f}")
        
        return self.metrics
    
    def print_classification_report(self, y_test: np.ndarray, y_pred: np.ndarray):
        """
        Print detailed classification report
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
        """
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
    def plot_confusion_matrix(
        self, 
        y_test: np.ndarray, 
        y_pred: np.ndarray,
        save_path: str = None
    ):
        """
        Plot confusion matrix
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        save_path: str = None
    ):
        """
        Plot ROC curve
        
        Args:
            X_test: Test features
            y_test: Test target
            save_path: Path to save the plot
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        save_path: str = None
    ):
        """
        Plot Precision-Recall curve
        
        Args:
            X_test: Test features
            y_test: Test target
            save_path: Path to save the plot
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def get_risk_assessment(self, probability: float) -> str:
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


def evaluate_model(
    model: Any, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    create_plots: bool = False
) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Print detailed report
    y_pred = model.predict(X_test)
    evaluator.print_classification_report(y_test, y_pred)
    
    # Create plots if requested
    if create_plots:
        evaluator.plot_confusion_matrix(y_test, y_pred)
        evaluator.plot_roc_curve(X_test, y_test)
        evaluator.plot_precision_recall_curve(X_test, y_test)
    
    return metrics