"""
Tests unitaires pour le pipeline de données météorologiques
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.data_pipeline import WeatherDataPipeline
from src.config import Config


class TestWeatherDataPipeline:
    """Suite de tests pour le pipeline de données"""
    
    @pytest.fixture
    def temp_dir(self):
        """Créer un répertoire temporaire pour les tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def pipeline(self, temp_dir):
        """Créer une instance du pipeline pour les tests"""
        return WeatherDataPipeline(
            raw_path=f"{temp_dir}/raw",
            processed_path=f"{temp_dir}/processed",
            features_path=f"{temp_dir}/features"
        )
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_pipeline_initialization(self, pipeline):
        """Test que le pipeline s'initialise correctement"""
        assert pipeline.raw_path.exists()
        assert pipeline.processed_path.exists()
        assert pipeline.features_path.exists()
        assert not pipeline.is_fitted
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_step1_download_raw_data(self, pipeline):
        """Test le chargement des données brutes"""
        raw_data = pipeline.step1_download_raw_data()
        
        assert raw_data is not None
        assert isinstance(raw_data, pd.DataFrame)
        assert len(raw_data) > 0
        assert raw_data.shape[1] > 0
        
        # Vérifier que le fichier a été sauvegardé
        raw_file = pipeline.raw_path / "weather_data_raw.csv"
        assert raw_file.exists()
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_step2_preprocess_data(self, pipeline):
        """Test le preprocessing des données"""
        # D'abord charger les données brutes
        raw_data = pipeline.step1_download_raw_data()
        
        # Ensuite les prétraiter
        processed_data = pipeline.step2_preprocess_data(raw_data)
        
        assert processed_data is not None
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        
        # Vérifier que le fichier a été sauvegardé
        processed_file = pipeline.processed_path / "weather_data_processed.csv"
        assert processed_file.exists()
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_step3_create_features(self, pipeline):
        """Test la création des features"""
        # Pipeline complet jusqu'aux features
        raw_data = pipeline.step1_download_raw_data()
        processed_data = pipeline.step2_preprocess_data(raw_data)
        features_data = pipeline.step3_create_features(processed_data)
        
        assert features_data is not None
        assert isinstance(features_data, pd.DataFrame)
        assert len(features_data) > 0
        assert features_data.shape[1] >= 40  # Au moins 40 features (raw + engineered)
        
        # Vérifier que le fichier a été sauvegardé
        features_file = pipeline.features_path / "weather_data_features.csv"
        assert features_file.exists()
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_validate_data(self, pipeline):
        """Test la validation des données"""
        raw_data = pipeline.step1_download_raw_data()
        
        is_valid, errors = pipeline.validate_data(raw_data)
        
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        if is_valid:
            assert len(errors) == 0
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_prepare_ml_data(self, pipeline):
        """Test la préparation des données pour ML"""
        # Pipeline complet
        raw_data = pipeline.step1_download_raw_data()
        processed_data = pipeline.step2_preprocess_data(raw_data)
        features_data = pipeline.step3_create_features(processed_data)
        
        ml_data = pipeline.prepare_ml_data(features_data)
        
        assert 'X_train' in ml_data
        assert 'X_test' in ml_data
        assert 'y_train' in ml_data
        assert 'y_test' in ml_data
        assert 'feature_names' in ml_data
        assert 'target_names' in ml_data
        
        # Vérifier les shapes
        assert ml_data['X_train'].ndim == 2
        assert ml_data['X_test'].ndim == 2
        assert ml_data['y_train'].ndim == 2
        assert ml_data['y_test'].ndim == 2
        
        # Vérifier le train/test split (~80/20)
        total_samples = len(ml_data['X_train']) + len(ml_data['X_test'])
        train_ratio = len(ml_data['X_train']) / total_samples
        assert 0.75 < train_ratio < 0.85
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_pipeline_scaler(self, pipeline):
        """Test que le scaler fonctionne correctement"""
        raw_data = pipeline.step1_download_raw_data()
        processed_data = pipeline.step2_preprocess_data(raw_data)
        features_data = pipeline.step3_create_features(processed_data)
        
        ml_data = pipeline.prepare_ml_data(features_data)
        
        # Vérifier que le scaler a été ajusté
        assert pipeline.is_fitted
        assert pipeline.scaler is not None
        
        # Vérifier que les données ont été normalisées
        X_train = ml_data['X_train']
        assert np.abs(X_train.mean()) < 0.1  # Moyenne proche de 0
        assert np.abs(X_train.std() - 1.0) < 0.1  # Std proche de 1
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_pipeline_save_and_load(self, pipeline, temp_dir):
        """Test la sauvegarde et le chargement du pipeline"""
        # Préparer les données
        raw_data = pipeline.step1_download_raw_data()
        processed_data = pipeline.step2_preprocess_data(raw_data)
        features_data = pipeline.step3_create_features(processed_data)
        pipeline.prepare_ml_data(features_data)
        
        # Sauvegarder
        filepath = f"{temp_dir}/test_pipeline.joblib"
        pipeline.save_pipeline(filepath)
        assert Path(filepath).exists()
        
        # Créer un nouveau pipeline et charger
        new_pipeline = WeatherDataPipeline(
            raw_path=f"{temp_dir}/raw2",
            processed_path=f"{temp_dir}/processed2",
            features_path=f"{temp_dir}/features2"
        )
        new_pipeline.load_pipeline(filepath)
        
        assert new_pipeline.is_fitted
        assert new_pipeline.scaler is not None
    
    @pytest.mark.unit
    @pytest.mark.data
    def test_run_full_pipeline(self, pipeline):
        """Test l'exécution complète du pipeline"""
        results = pipeline.run_full_pipeline()
        
        assert 'ml_data' in results
        assert 'stats' in results
        assert 'raw_data' in results
        assert 'processed_data' in results
        assert 'features_data' in results
        
        stats = results['stats']
        assert stats['raw_shape'][0] > 0
        assert stats['processed_shape'][0] > 0
        assert stats['features_shape'][0] > 0
        assert stats['pipeline_saved']
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_pipeline_integration(self, pipeline):
        """Test d'intégration complet du pipeline"""
        results = pipeline.run_full_pipeline()
        
        # Vérifier les statistiques
        stats = results['stats']
        
        # Vérifier que les données diminuent légèrement (suppression des NaN)
        assert stats['raw_shape'][0] >= stats['processed_shape'][0]
        assert stats['processed_shape'][0] >= stats['features_shape'][0]
        
        # Vérifier les features
        # Note: feature_count = nombre de features sélectionnées pour ML (22)
        # pas le nombre total de colonnes créées (50)
        assert stats['feature_count'] >= 22
        
        # Vérifier les cibles
        assert len(stats['target']) >= 1
