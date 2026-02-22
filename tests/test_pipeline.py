"""
Unit tests for the ML pipeline
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer
from src.predict import Predictor
from src import config


class TestDataLoader:
    """Test data loading functionality"""
    
    def test_data_loader_initialization(self):
        """Test DataLoader can be initialized"""
        loader = DataLoader()
        assert loader is not None
        assert loader.data_path == config.DATASET_PATH
    
    def test_validate_data_with_empty_dataframe(self):
        """Test validation fails with empty DataFrame"""
        loader = DataLoader()
        df_empty = pd.DataFrame()
        assert not loader.validate_data(df_empty)
    
    def test_get_data_info(self):
        """Test data info extraction"""
        loader = DataLoader()
        df = pd.DataFrame({
            'diagnosis': ['M', 'B', 'M'],
            'feature1': [1, 2, 3]
        })
        info = loader.get_data_info(df)
        
        assert 'shape' in info
        assert 'columns' in info
        assert info['shape'] == (3, 2)


class TestDataPreprocessor:
    """Test data preprocessing functionality"""
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor can be initialized"""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert preprocessor.scaler is not None
    
    def test_clean_data(self):
        """Test data cleaning"""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'diagnosis': ['M', 'B', 'M'],
            'Unnamed: 32': [np.nan, np.nan, np.nan],
            'feature1': [1.0, 2.0, 3.0]
        })
        
        df_clean = preprocessor.clean_data(df)
        
        # Check id and Unnamed columns are dropped
        assert 'id' not in df_clean.columns
        assert 'Unnamed: 32' not in df_clean.columns
        
        # Check diagnosis is mapped to 0/1
        assert df_clean['diagnosis'].dtype in [np.int64, np.int32]
        assert set(df_clean['diagnosis'].unique()).issubset({0, 1})
    
    def test_prepare_features(self):
        """Test feature preparation"""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'diagnosis': [1, 0, 1],
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        
        X, y = preprocessor.prepare_features(df)
        
        assert X.shape[1] == 2  # 2 features
        assert len(y) == 3  # 3 samples
        assert 'diagnosis' not in X.columns
    
    def test_split_data(self):
        """Test data splitting"""
        preprocessor = DataPreprocessor()
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_scale_features(self):
        """Test feature scaling"""
        preprocessor = DataPreprocessor()
        X_train = pd.DataFrame(np.random.rand(100, 5))
        X_test = pd.DataFrame(np.random.rand(20, 5))
        
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        
        # Check scaled data has mean ~ 0 and std ~ 1
        assert np.abs(X_train_scaled.mean()) < 0.1
        assert np.abs(X_train_scaled.std() - 1.0) < 0.1


class TestModelTrainer:
    """Test model training functionality"""
    
    def test_trainer_initialization(self):
        """Test ModelTrainer can be initialized"""
        trainer = ModelTrainer()
        assert trainer is not None
        assert trainer.models == {}
        assert trainer.best_model is None
    
    def test_get_base_models(self):
        """Test base model dictionary creation"""
        trainer = ModelTrainer()
        models = trainer.get_base_models()
        
        assert len(models) == 3
        assert "Logistic Regression" in models
        assert "Random Forest" in models
        assert "SVM" in models
    
    def test_train_base_models(self):
        """Test base model training"""
        trainer = ModelTrainer()
        
        # Create synthetic data
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        
        models = trainer.train_base_models(X_train, y_train)
        
        assert len(models) == 3
        assert trainer.best_model is not None
        assert trainer.best_score > 0


class TestPredictor:
    """Test prediction functionality"""
    
    @pytest.fixture
    def mock_predictor(self, tmp_path):
        """Create a mock predictor with dummy model"""
        # This test would need actual model files
        # Skipping for now
        pytest.skip("Requires trained model files")
    
    def test_risk_level_low(self):
        """Test low risk categorization"""
        from src.predict import Predictor
        
        risk = Predictor._get_risk_level(0.2)
        assert risk == "Low Risk"
    
    def test_risk_level_medium(self):
        """Test medium risk categorization"""
        from src.predict import Predictor
        
        risk = Predictor._get_risk_level(0.5)
        assert risk == "Medium Risk"
    
    def test_risk_level_high(self):
        """Test high risk categorization"""
        from src.predict import Predictor
        
        risk = Predictor._get_risk_level(0.9)
        assert risk == "High Risk"


class TestConfiguration:
    """Test configuration settings"""
    
    def test_config_paths_exist(self):
        """Test that config paths are defined"""
        assert config.PROJECT_ROOT is not None
        assert config.DATA_DIR is not None
        assert config.MODELS_DIR is not None
    
    def test_config_values(self):
        """Test configuration values"""
        assert config.TEST_SIZE == 0.2
        assert config.RANDOM_STATE == 42
        assert config.CROSS_VAL_FOLDS == 5
    
    def test_risk_thresholds(self):
        """Test risk threshold configuration"""
        assert config.LOW_RISK_THRESHOLD < config.HIGH_RISK_THRESHOLD
        assert 0 < config.LOW_RISK_THRESHOLD < 1
        assert 0 < config.HIGH_RISK_THRESHOLD < 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])