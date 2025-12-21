"""
Tests unitaires pour l'entraînement des modèles ML
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json

from src.train_model import WeatherModelTrainer, MetricsCalculator
from src.data_pipeline import WeatherDataPipeline


class TestMetricsCalculator:
    """Suite de tests pour le calculateur de métriques"""
    
    @pytest.mark.unit
    def test_metrics_calculation(self):
        """Test le calcul des métriques"""
        # Données factices
        y_train_true = np.random.rand(100, 3)
        y_train_pred = y_train_true + np.random.randn(100, 3) * 0.1
        y_test_true = np.random.rand(30, 3)
        y_test_pred = y_test_true + np.random.randn(30, 3) * 0.1
        target_names = ['temp_mean', 'temp_min', 'temp_max']
        
        metrics = MetricsCalculator.calculate_multi_target_metrics(
            y_train_true, y_train_pred, y_test_true, y_test_pred, target_names
        )
        
        assert isinstance(metrics, dict)
        assert 'avg_test_rmse' in metrics
        assert 'avg_test_mae' in metrics
        assert 'avg_test_r2' in metrics
        
        # Vérifier que les métriques sont positives
        assert metrics['avg_test_rmse'] > 0
        assert metrics['avg_test_mae'] > 0
    
    @pytest.mark.unit
    def test_metrics_1d_arrays(self):
        """Test avec des arrays 1D"""
        y_train_true = np.random.rand(100)
        y_train_pred = y_train_true + np.random.randn(100) * 0.1
        y_test_true = np.random.rand(30)
        y_test_pred = y_test_true + np.random.randn(30) * 0.1
        target_names = ['temperature']
        
        metrics = MetricsCalculator.calculate_multi_target_metrics(
            y_train_true, y_train_pred, y_test_true, y_test_pred, target_names
        )
        
        assert isinstance(metrics, dict)
        assert metrics['avg_test_rmse'] > 0


class TestWeatherModelTrainer:
    """Suite de tests pour l'entraîneur de modèles"""
    
    @pytest.fixture
    def trainer(self):
        """Créer une instance du trainer pour les tests"""
        return WeatherModelTrainer(
            mlflow_uri="file:./mlruns",
            experiment_name="test_experiment"
        )
    
    @pytest.fixture
    def trainer_with_data(self, trainer):
        """Créer un trainer avec données préparées"""
        trainer.prepare_data()
        return trainer
    
    @pytest.mark.unit
    def test_trainer_initialization(self, trainer):
        """Test l'initialisation du trainer"""
        assert trainer.mlflow_uri is not None
        assert trainer.experiment_name is not None
        assert trainer.pipeline is not None
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_prepare_data(self, trainer):
        """Test la préparation des données"""
        results = trainer.prepare_data()
        
        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert trainer.y_train is not None
        assert trainer.y_test is not None
        assert trainer.feature_names is not None
        assert trainer.target_names is not None
        
        # Vérifier les shapes
        assert trainer.X_train.shape[0] > trainer.X_test.shape[0]
        assert trainer.X_train.shape[1] == trainer.X_test.shape[1]
    
    @pytest.mark.unit
    def test_validate_data_prepared(self, trainer):
        """Test la validation que les données sont prêtes"""
        # Devrait lever une erreur si les données ne sont pas préparées
        with pytest.raises(ValueError):
            trainer._validate_data_prepared()
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_train_random_forest(self, trainer_with_data):
        """Test l'entraînement du Random Forest"""
        model, metrics = trainer_with_data.train_random_forest()
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'avg_test_rmse' in metrics
        assert 'avg_test_mae' in metrics
        assert 'avg_test_r2' in metrics
        
        # Vérifier que les métriques sont raisonnables
        assert metrics['avg_test_rmse'] > 0
        assert 0 <= metrics['avg_test_r2'] <= 1
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_train_gradient_boosting(self, trainer_with_data):
        """Test l'entraînement du Gradient Boosting"""
        model, metrics = trainer_with_data.train_gradient_boosting()
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'avg_test_rmse' in metrics
        assert 'avg_test_mae' in metrics
        assert 'avg_test_r2' in metrics
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_train_linear_regression(self, trainer_with_data):
        """Test l'entraînement de la régression linéaire"""
        model, metrics = trainer_with_data.train_linear_regression()
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'avg_test_rmse' in metrics
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_compare_models(self, trainer_with_data):
        """Test la comparaison des modèles"""
        metrics1 = {'avg_test_rmse': 1.5, 'avg_test_mae': 1.0, 'avg_test_r2': 0.95}
        metrics2 = {'avg_test_rmse': 1.2, 'avg_test_mae': 0.8, 'avg_test_r2': 0.97}
        metrics3 = {'avg_test_rmse': 2.0, 'avg_test_mae': 1.5, 'avg_test_r2': 0.90}
        
        models_results = {
            'Model1': metrics1,
            'Model2': metrics2,
            'Model3': metrics3
        }
        
        best_model = trainer_with_data.compare_models(models_results)
        
        assert best_model == 'Model2'  # Le meilleur (RMSE le plus faible)
    
    @pytest.mark.unit
    def test_save_results(self, trainer, tmp_path):
        """Test la sauvegarde des résultats"""
        results = {
            'model': 'test_model',
            'rmse': 1.5,
            'mae': 1.0,
            'r2': 0.95
        }
        
        filepath = str(tmp_path / "results.json")
        trainer.save_results(results, filepath)
        
        assert Path(filepath).exists()
        
        # Vérifier que le fichier contient les bonnes données
        with open(filepath, 'r') as f:
            saved_results = json.load(f)
        
        assert saved_results['model'] == 'test_model'
        assert saved_results['rmse'] == 1.5
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_training(self, trainer):
        """Test l'entraînement complet"""
        results = trainer.run_full_training()
        
        assert 'data_preparation' in results
        assert 'models_performance' in results
        assert 'best_model' in results
        assert 'best_hyperparameters' in results
        
        # Vérifier que les données sont correctes
        data_prep = results['data_preparation']
        assert data_prep['train_samples'] > 0
        assert data_prep['test_samples'] > 0
        assert data_prep['feature_count'] > 0
        
        # Vérifier les performances des modèles
        models_perf = results['models_performance']
        assert len(models_perf) >= 3  # Au moins 3 modèles
