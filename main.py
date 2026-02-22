"""
Main pipeline script for training the breast cancer detection model
"""
import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_and_validate_data
from src.preprocessing import preprocess_pipeline
from src.train import train_pipeline
from src.evaluate import evaluate_model
from src import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(tune_hyperparameters=True, create_plots=False):
    """
    Run the complete ML pipeline
    
    Args:
        tune_hyperparameters: Whether to perform hyperparameter tuning
        create_plots: Whether to create evaluation plots
    """
    try:
        logger.info("=" * 50)
        logger.info("BREAST CANCER DETECTION MODEL TRAINING PIPELINE")
        logger.info("=" * 50)
        
        # Step 1: Load and validate data
        logger.info("\n[STEP 1] Loading and validating data...")
        df = load_and_validate_data()
        logger.info(f"Dataset shape: {df.shape}")
        
        # Step 2: Preprocess data
        logger.info("\n[STEP 2] Preprocessing data...")
        X_train_scaled, X_test_scaled, y_train, y_test, preprocessor = preprocess_pipeline(df)
        
        # Save scaler
        preprocessor.save_scaler()
        
        # Step 3: Train models
        logger.info("\n[STEP 3] Training models...")
        trainer = train_pipeline(X_train_scaled, y_train, tune_hyperparameters)
        
        logger.info(f"\nBest model: {trainer.best_model_name}")
        logger.info(f"Best CV score: {trainer.best_score:.4f}")
        
        # Step 4: Evaluate model
        logger.info("\n[STEP 4] Evaluating model on test set...")
        metrics = evaluate_model(
            trainer.best_model, 
            X_test_scaled, 
            y_test,
            create_plots=create_plots
        )
        
        # Step 5: Summary
        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info(f"Model saved to: {config.MODEL_PATH}")
        logger.info(f"Scaler saved to: {config.SCALER_PATH}")
        logger.info("\nFinal Test Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info("=" * 50)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train breast cancer detection model')
    parser.add_argument(
        '--no-tuning', 
        action='store_true',
        help='Skip hyperparameter tuning (faster training)'
    )
    parser.add_argument(
        '--plots', 
        action='store_true',
        help='Create evaluation plots'
    )
    
    args = parser.parse_args()
    
    main(
        tune_hyperparameters=not args.no_tuning,
        create_plots=args.plots
    )