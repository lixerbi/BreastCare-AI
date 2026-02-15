"""
Main pipeline script for training the breast cancer detection model
"""
import sys
from pathlib import Path
import logging
import argparse

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
    """Run the complete ML pipeline"""
    try:
        logger.info("="*50)
        logger.info("BREAST CANCER DETECTION MODEL TRAINING PIPELINE")
        logger.info("="*50)
        
        # Load data
        logger.info("\n[STEP 1] Loading and validating data...")
        df = load_and_validate_data()
        
        # Preprocess
        logger.info("\n[STEP 2] Preprocessing data...")
        X_train_scaled, X_test_scaled, y_train, y_test, preprocessor = preprocess_pipeline(df)
        preprocessor.save_scaler()
        
        # Train
        logger.info("\n[STEP 3] Training models...")
        trainer = train_pipeline(X_train_scaled, y_train, tune_hyperparameters)
        
        # Evaluate
        logger.info("\n[STEP 4] Evaluating model...")
        metrics = evaluate_model(trainer.best_model, X_test_scaled, y_test, create_plots)
        
        logger.info("\n" + "="*50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")
        logger.info("="*50)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-tuning', action='store_true')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    
    main(not args.no_tuning, args.plots)
