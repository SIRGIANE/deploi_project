"""
Tests unitaires pour le projet Climate MLOps
Tests pour les pipelines de données, modèles et API
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import os
import sys

# Imports des modules à tester
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import WeatherDataPipeline
from src.data_validation import DataQualityValidator
from src.marrakech_data_loader import MarrakechWeatherDataLoader
from src.config import Config
from src.train_model import WeatherModelTrainer, MetricsCalculator

class TestDataQualityValidator:
    """Tests pour le validateur de données"""
    
    @pytest.fixture
    def validator(self):
        return DataQualityValidator()

    @pytest.fixture
    def valid_dataframe(self):
        """DataFrame valide pour les tests"""
        dates = pd.date_range('2018-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'datetime': dates,
            'temperature_2m_mean': np.random.normal(20, 5, 100),
            'temperature_2m_max': np.random.normal(25, 5, 100),
            'temperature_2m_min': np.random.normal(15, 5, 100)
        })
    
    def test_validate_structure(self, validator, valid_dataframe):
        """Test validation structure"""
        results = validator.validate_basic_structure(valid_dataframe)
        assert results['row_count']['result'] is True
        assert results['required_columns']['result'] is True

    def test_validate_temperature(self, validator, valid_dataframe):
        """Test validation température"""
        results = validator.validate_temperature_data(valid_dataframe)
        assert results['temperature_2m_mean']['result'] is True

class TestMarrakechLoader:
    """Tests pour le loader de données"""
    
    def test_feature_creation(self, tmp_path):
        """Test de création des features"""
        # Create a dummy file
        d = tmp_path / "dummy.csv"
        d.write_text("col1,col2")
        
        loader = MarrakechWeatherDataLoader(str(d))
        
        # Create dummy processed data
        dates = pd.date_range('2018-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'datetime': dates,
            'temperature_2m_mean': np.random.normal(20, 5, 50)
        })
        
        features = loader.create_weather_features(df)
        
        assert 'Year' in features.columns
        assert 'Month' in features.columns
        assert 'Temp_lag_1' in features.columns

class TestWeatherDataPipeline:
    """Tests pour le pipeline de données climatiques"""
    
    @pytest.fixture
    def pipeline(self, tmp_path):
        """Instance du pipeline pour les tests"""
        # Create a dummy file
        d = tmp_path / "dummy.csv"
        d.write_text("col1,col2")
        return WeatherDataPipeline(data_file=str(d))
    
    @patch('src.marrakech_data_loader.MarrakechWeatherDataLoader.load_weather_data')
    def test_step1_download(self, mock_load, pipeline):
        """Test étape 1"""
        mock_df = pd.DataFrame({'col': [1, 2]})
        mock_load.return_value = mock_df
        
        with patch('pandas.DataFrame.to_csv'):
            result = pipeline.step1_download_raw_data()
            assert len(result) == 2
            
    @patch('src.marrakech_data_loader.MarrakechWeatherDataLoader.preprocess_weather_data')
    def test_step2_preprocess(self, mock_process, pipeline):
        """Test étape 2"""
        mock_df = pd.DataFrame({'col': [1, 2]})
        mock_process.return_value = mock_df
        
        with patch('pandas.DataFrame.to_csv'):
            # Mock reading raw file or pass df
            result = pipeline.step2_preprocess_data(mock_df)
            assert len(result) == 2

class TestMetricsCalculator:
    """Tests pour le calculateur de métriques"""
    
    def test_calculate_metrics(self):
        """Test du calcul des métriques"""
        y_true = np.array([[10], [20], [30]])
        y_pred = np.array([[11], [19], [32]])
        target_names = ['temp']
        
        metrics = MetricsCalculator.calculate_multi_target_metrics(
            y_true, y_pred, y_true, y_pred, target_names
        )
        
        assert 'avg_test_rmse' in metrics
        assert metrics['avg_test_rmse'] > 0

class TestWeatherModelTrainer:
    """Tests pour l'entraîneur de modèle"""
    
    @patch('src.data_pipeline.WeatherDataPipeline.run_full_pipeline')
    def test_prepare_data(self, mock_pipeline, tmp_path):
        """Test préparation des données"""
        # Mock pipeline results
        X = np.random.rand(10, 5)
        y = np.random.rand(10, 1)
        mock_pipeline.return_value = {
            'ml_data': {
                'X_train': X, 'X_test': X,
                'y_train': y, 'y_test': y,
                'feature_names': ['f1', 'f2', 'f3', 'f4', 'f5'],
                'target_names': ['t1']
            }
        }
        
        trainer = WeatherModelTrainer()
        trainer.prepare_data()
        
        assert trainer.X_train is not None
        assert trainer.X_train.shape == (10, 5)

if __name__ == "__main__":
    pytest.main([__file__])
