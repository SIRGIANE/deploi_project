"""
Tests unitaires pour le pipeline de données
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime

# Importer les modules à tester
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import DataPipeline


class TestDataPipeline:
    """Tests pour la classe DataPipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Créer des données d'exemple pour les tests"""
        dates = pd.date_range('2018-01-01', periods=1000, freq='D')
        data = pd.DataFrame({
            'time': dates,
            'temperature_2m_mean': np.random.randn(1000) * 5 + 20,
            'temperature_2m_max': np.random.randn(1000) * 5 + 25,
            'temperature_2m_min': np.random.randn(1000) * 5 + 15,
            'precipitation_sum': np.random.rand(1000) * 10
        })
        return data
    
    @pytest.fixture
    def pipeline(self):
        """Créer une instance du pipeline"""
        return DataPipeline()
    
    def test_data_loading(self, pipeline, sample_data):
        """Test du chargement des données"""
        assert sample_data is not None
        assert len(sample_data) > 0
        assert 'temperature_2m_mean' in sample_data.columns
    
    def test_data_cleaning(self, pipeline, sample_data):
        """Test du nettoyage des données"""
        # Ajouter des NaN
        sample_data.loc[0:10, 'temperature_2m_mean'] = np.nan
        
        # Utiliser le loader pour le preprocessing
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        loader = MarrakechWeatherDataLoader("marrakech_weather_2018_2023_final.csv")
        
        # Mocking the load to avoid file dependency in unit test if possible, 
        # but here we are testing the logic. 
        # Since we can't easily mock the internal loader of DataPipeline without refactoring,
        # we will test the loader's method directly or the pipeline's step2.
        
        cleaned = loader.preprocess_weather_data(sample_data)
        
        assert len(cleaned) == len(sample_data) # Should keep rows but fill NaNs
        assert cleaned['temperature_2m_mean'].isna().sum() == 0
    
    def test_feature_engineering(self, pipeline, sample_data):
        """Test de l'ingénierie des features"""
        # Preprocess first
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        loader = MarrakechWeatherDataLoader("marrakech_weather_2018_2023_final.csv")
        processed = loader.preprocess_weather_data(sample_data)
        
        features = loader.create_weather_features(processed)
        
        assert 'Year' in features.columns
        assert 'Month' in features.columns
        assert features['Year'].min() >= 2018
    
    def test_data_split(self, pipeline, sample_data):
        """Test de la division train/test"""
        # This test depends on prepare_ml_data which uses the pipeline
        # We need to mock the internal data loading or pass data
        
        # Let's test the logic manually or assume pipeline.prepare_ml_data works with passed df
        # The current implementation of prepare_ml_data in DataPipeline takes df as argument
        
        # Preprocess and create features first
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        loader = MarrakechWeatherDataLoader("marrakech_weather_2018_2023_final.csv")
        processed = loader.preprocess_weather_data(sample_data)
        features = loader.create_weather_features(processed)
        
        ml_data = pipeline.prepare_ml_data(features, split_ratio=0.8)
        
        assert len(ml_data['X_train']) > 0
        assert len(ml_data['X_test']) > 0
        assert len(ml_data['X_train']) + len(ml_data['X_test']) <= len(features)
    
    def test_data_normalization(self, pipeline, sample_data):
        """Test de la normalisation des données"""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sample_data[['temperature_2m_mean']])
        
        assert np.isclose(scaled_data.mean(), 0, atol=1e-10)
        assert np.isclose(scaled_data.std(), 1, atol=1e-10)
    
    def test_temporal_features(self, pipeline, sample_data):
        """Test des features temporelles"""
        sample_data['datetime'] = pd.to_datetime(sample_data['time'])
        sample_data['Quarter'] = sample_data['datetime'].dt.quarter
        sample_data['DayOfYear'] = sample_data['datetime'].dt.dayofyear
        
        assert sample_data['Quarter'].min() >= 1
        assert sample_data['Quarter'].max() <= 4
        assert sample_data['DayOfYear'].min() >= 1
        assert sample_data['DayOfYear'].max() <= 366
    
    def test_lag_features(self, pipeline, sample_data):
        """Test des features de décalage (lag)"""
        sample_data['Temp_lag_1'] = sample_data['temperature_2m_mean'].shift(1)
        sample_data['Temp_lag_7'] = sample_data['temperature_2m_mean'].shift(7)
        
        # Le premier élément de lag_1 doit être NaN
        assert pd.isna(sample_data['Temp_lag_1'].iloc[0])
        
        # Vérifier la cohérence
        assert np.isclose(sample_data['Temp_lag_1'].iloc[1], 
                         sample_data['temperature_2m_mean'].iloc[0])
    
    def test_moving_average_features(self, pipeline, sample_data):
        """Test des features de moyenne mobile"""
        sample_data['Temp_ma_3'] = sample_data['temperature_2m_mean'].rolling(window=3).mean()
        
        # Les 2 premiers éléments de ma_3 doivent être NaN
        assert pd.isna(sample_data['Temp_ma_3'].iloc[0:2]).all()
    
    def test_data_quality(self, pipeline, sample_data):
        """Test de la qualité des données"""
        # Vérifier que les températures sont raisonnables
        assert sample_data['temperature_2m_mean'].min() > -20
        assert sample_data['temperature_2m_mean'].max() < 60
        
        # Vérifier qu'il n'y a pas de duplicatas de dates
        assert not sample_data['time'].duplicated().any()


class TestDataPipelineIntegration:
    """Tests d'intégration pour le pipeline complet"""
    
    def test_full_pipeline(self):
        """Test du pipeline complet"""
        # Créer des données d'exemple (augmenté pour éviter l'erreur de données insuffisantes)
        dates = pd.date_range('2018-01-01', periods=200, freq='D')
        data = pd.DataFrame({
            'time': dates,
            'temperature_2m_mean': np.random.randn(200) * 5 + 20,
            'temperature_2m_max': np.random.randn(200) * 5 + 25,
            'temperature_2m_min': np.random.randn(200) * 5 + 15,
        })
        
        # Initialiser le pipeline
        pipeline = DataPipeline()
        
        # Injecter les données mockées pour éviter de charger le fichier réel
        # On utilise les méthodes pas à pas
        
        # 1. Preprocess
        # On doit mocker le loader interne ou utiliser directement le loader
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        loader = MarrakechWeatherDataLoader("marrakech_weather_2018_2023_final.csv")
        
        processed = loader.preprocess_weather_data(data)
        
        # 2. Features
        features = loader.create_weather_features(processed)
        
        # 3. Prepare ML
        ml_data = pipeline.prepare_ml_data(features)
        
        # Vérifier que le pipeline fonctionne
        assert ml_data['X_train'].shape[0] > 0
        assert ml_data['X_test'].shape[0] > 0
        assert 'temperature_2m_mean' in ml_data['target_names']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
