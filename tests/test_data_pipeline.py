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
        dates = pd.date_range('1750-01-01', periods=3000, freq='MS')
        data = pd.DataFrame({
            'dt': dates,
            'LandAverageTemperature': np.random.randn(3000) * 2 + 10,
            'LandAverageTemperatureUncertainty': np.random.rand(3000) * 0.5
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
        assert 'LandAverageTemperature' in sample_data.columns
    
    def test_data_cleaning(self, pipeline, sample_data):
        """Test du nettoyage des données"""
        # Ajouter des NaN
        sample_data.loc[0:10, 'LandAverageTemperature'] = np.nan
        
        cleaned = sample_data.dropna(subset=['LandAverageTemperature'])
        
        assert len(cleaned) < len(sample_data)
        assert cleaned['LandAverageTemperature'].isna().sum() == 0
    
    def test_feature_engineering(self, pipeline, sample_data):
        """Test de l'ingénierie des features"""
        sample_data['dt'] = pd.to_datetime(sample_data['dt'])
        sample_data['Year'] = sample_data['dt'].dt.year
        sample_data['Month'] = sample_data['dt'].dt.month
        
        assert 'Year' in sample_data.columns
        assert 'Month' in sample_data.columns
        assert sample_data['Year'].min() > 1700
    
    def test_data_split(self, pipeline, sample_data):
        """Test de la division train/test"""
        split_date = pd.to_datetime('2010-01-01')
        train = sample_data[sample_data['dt'] < split_date]
        test = sample_data[sample_data['dt'] >= split_date]
        
        assert len(train) + len(test) == len(sample_data)
        assert train['dt'].max() < split_date
        assert test['dt'].min() >= split_date
    
    def test_data_normalization(self, pipeline, sample_data):
        """Test de la normalisation des données"""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(sample_data[['LandAverageTemperature']])
        
        assert np.isclose(scaled_data.mean(), 0, atol=1e-10)
        assert np.isclose(scaled_data.std(), 1, atol=1e-10)
    
    def test_temporal_features(self, pipeline, sample_data):
        """Test des features temporelles"""
        sample_data['dt'] = pd.to_datetime(sample_data['dt'])
        sample_data['Quarter'] = sample_data['dt'].dt.quarter
        sample_data['DayOfYear'] = sample_data['dt'].dt.dayofyear
        
        assert sample_data['Quarter'].min() >= 1
        assert sample_data['Quarter'].max() <= 4
        assert sample_data['DayOfYear'].min() >= 1
        assert sample_data['DayOfYear'].max() <= 366
    
    def test_lag_features(self, pipeline, sample_data):
        """Test des features de décalage (lag)"""
        sample_data['Temp_lag_1'] = sample_data['LandAverageTemperature'].shift(1)
        sample_data['Temp_lag_12'] = sample_data['LandAverageTemperature'].shift(12)
        
        # Le premier élément de lag_1 doit être NaN
        assert pd.isna(sample_data['Temp_lag_1'].iloc[0])
        assert pd.isna(sample_data['Temp_lag_12'].iloc[0])
        
        # Vérifier la cohérence
        assert np.isclose(sample_data['Temp_lag_1'].iloc[1], 
                         sample_data['LandAverageTemperature'].iloc[0])
    
    def test_moving_average_features(self, pipeline, sample_data):
        """Test des features de moyenne mobile"""
        sample_data['Temp_ma_3'] = sample_data['LandAverageTemperature'].rolling(window=3).mean()
        sample_data['Temp_ma_12'] = sample_data['LandAverageTemperature'].rolling(window=12).mean()
        
        # Les 2 premiers éléments de ma_3 doivent être NaN
        assert pd.isna(sample_data['Temp_ma_3'].iloc[0:2]).all()
        # Les 11 premiers éléments de ma_12 doivent être NaN
        assert pd.isna(sample_data['Temp_ma_12'].iloc[0:11]).all()
    
    def test_data_quality(self, pipeline, sample_data):
        """Test de la qualité des données"""
        # Vérifier que les températures sont raisonnables
        assert sample_data['LandAverageTemperature'].min() > -50
        assert sample_data['LandAverageTemperature'].max() < 50
        
        # Vérifier qu'il n'y a pas de duplicatas de dates
        assert not sample_data['dt'].duplicated().any()


class TestDataPipelineIntegration:
    """Tests d'intégration pour le pipeline complet"""
    
    def test_full_pipeline(self):
        """Test du pipeline complet"""
        # Créer des données d'exemple
        dates = pd.date_range('1750-01-01', periods=1000, freq='MS')
        data = pd.DataFrame({
            'dt': dates,
            'LandAverageTemperature': np.random.randn(1000) * 2 + 10,
        })
        
        # Appliquer les transformations
        data['Year'] = data['dt'].dt.year
        data['Month'] = data['dt'].dt.month
        data['Temp_lag_1'] = data['LandAverageTemperature'].shift(1)
        data['Temp_ma_3'] = data['LandAverageTemperature'].rolling(window=3).mean()
        
        # Nettoyer
        data = data.dropna()
        
        # Vérifier que le pipeline fonctionne
        assert len(data) > 0
        assert 'Year' in data.columns
        assert 'Month' in data.columns
        assert 'Temp_lag_1' in data.columns
        assert 'Temp_ma_3' in data.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
