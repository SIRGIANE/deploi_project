"""
Tests unitaires pour les modèles ML
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class TestModelTraining:
    """Tests pour l'entraînement des modèles"""
    
    @pytest.fixture
    def sample_dataset(self):
        """Créer un dataset d'exemple"""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
        
        return X, y
    
    def test_model_initialization(self, sample_dataset):
        """Test de l'initialisation du modèle"""
        X, y = sample_dataset
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        assert model is not None
        assert model.n_estimators == 10
    
    def test_model_training(self, sample_dataset):
        """Test de l'entraînement du modèle"""
        X, y = sample_dataset
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Vérifier que le modèle a été entraîné
        assert hasattr(model, 'estimators_')
        assert len(model.estimators_) == 10
    
    def test_model_prediction(self, sample_dataset):
        """Test de la prédiction du modèle"""
        X, y = sample_dataset
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()
    
    def test_model_performance(self, sample_dataset):
        """Test des performances du modèle"""
        X, y = sample_dataset
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        # Le modèle devrait avoir des performances raisonnables
        assert r2 > 0.5
        assert rmse < 10
    
    def test_feature_importance(self, sample_dataset):
        """Test de l'importance des features"""
        X, y = sample_dataset
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importances = model.feature_importances_
        
        assert importances.shape == (20,)
        assert np.isclose(importances.sum(), 1.0)
    
    def test_model_scalability(self):
        """Test de la scalabilité du modèle"""
        # Avec un dataset plus grand
        X_large = np.random.randn(1000, 20)
        y_large = np.random.randn(1000)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_large, y_large)
        
        predictions = model.predict(X_large)
        assert predictions.shape == y_large.shape


class TestModelValidation:
    """Tests de validation des modèles"""
    
    @pytest.fixture
    def trained_model(self):
        """Créer un modèle entraîné"""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model, X, y
    
    def test_cross_validation(self, trained_model):
        """Test de la validation croisée"""
        from sklearn.model_selection import cross_val_score
        
        model, X, y = trained_model
        
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        assert len(scores) == 5
        assert all(s < 1.0 for s in scores)
    
    def test_model_overfitting(self, trained_model):
        """Test de détection du surapprentissage"""
        model, X_train, y_train = trained_model
        
        # Créer un dataset de test
        X_test = np.random.randn(50, 20)
        y_test = np.random.randn(50)
        
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        
        # Le modèle ne devrait pas être drastiquement overfitting
        assert train_r2 - test_r2 < 0.5
    
    def test_model_stability(self):
        """Test de la stabilité du modèle"""
        np.random.seed(42)
        
        # Entraîner plusieurs fois avec les mêmes données
        predictions_list = []
        
        for _ in range(3):
            X = np.random.randn(100, 20)
            y = np.random.randn(100)
            
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            pred = model.predict(X)
            predictions_list.append(pred)
        
        # Les prédictions devraient être identiques (même seed)
        assert np.allclose(predictions_list[0], predictions_list[1])


class TestModelComparison:
    """Tests de comparaison entre modèles"""
    
    @pytest.fixture
    def test_data(self):
        """Créer des données de test"""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)
        return X, y
    
    def test_multiple_models_comparison(self, test_data):
        """Comparer plusieurs modèles"""
        X, y = test_data
        
        models = {
            'RF_10': RandomForestRegressor(n_estimators=10, random_state=42),
            'RF_50': RandomForestRegressor(n_estimators=50, random_state=42),
            'RF_100': RandomForestRegressor(n_estimators=100, random_state=42),
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X, y)
            r2 = model.score(X, y)
            results[name] = r2
        
        # Vérifier que tous les modèles donnent des scores raisonnables
        assert all(v > 0 for v in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
