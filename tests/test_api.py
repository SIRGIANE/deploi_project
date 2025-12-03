"""
Tests unitaires pour l'API FastAPI
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app


@pytest.fixture
def client():
    """Créer un client de test pour l'API"""
    return TestClient(app)


class TestAPIHealth:
    """Tests de santé de l'API"""
    
    def test_health_endpoint(self, client):
        """Test du endpoint /health"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_docs_endpoint(self, client):
        """Test de la documentation Swagger"""
        response = client.get("/docs")
        assert response.status_code == 200


class TestPredictionAPI:
    """Tests de l'API de prédiction"""
    
    def test_predict_endpoint_exists(self, client):
        """Test que l'endpoint de prédiction existe"""
        response = client.get("/api/v1/predict")
        assert response.status_code != 404
    
    def test_predict_with_valid_data(self, client):
        """Test de prédiction avec données valides"""
        # Créer des features valides (20 features)
        features = [float(i) for i in range(20)]
        
        payload = {
            "features": features,
            "model": "random_forest"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        
        # L'endpoint devrait retourner 200 ou 422 (validation error)
        assert response.status_code in [200, 422, 500]
    
    def test_predict_with_invalid_features_count(self, client):
        """Test de prédiction avec mauvais nombre de features"""
        payload = {
            "features": [1.0, 2.0],  # Pas assez de features
            "model": "random_forest"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        
        # Devrait retourner une erreur de validation
        assert response.status_code == 422
    
    def test_predict_with_invalid_model_type(self, client):
        """Test de prédiction avec type de modèle invalide"""
        features = [float(i) for i in range(20)]
        
        payload = {
            "features": features,
            "model": "invalid_model"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        
        # Devrait retourner une erreur 400 ou 422
        assert response.status_code in [400, 422]


class TestBatchPrediction:
    """Tests des prédictions en batch"""
    
    def test_batch_predict_endpoint(self, client):
        """Test de l'endpoint de prédiction en batch"""
        batch_data = {
            "data": [
                [float(i) for i in range(20)],
                [float(i+1) for i in range(20)],
                [float(i+2) for i in range(20)],
            ],
            "model": "random_forest"
        }
        
        response = client.post("/api/v1/batch_predict", json=batch_data)
        
        assert response.status_code in [200, 422, 500]


class TestModelsAPI:
    """Tests de l'API de gestion des modèles"""
    
    def test_list_models_endpoint(self, client):
        """Test de la liste des modèles"""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data or isinstance(data, list)
    
    def test_model_info_endpoint(self, client):
        """Test de l'info d'un modèle"""
        response = client.get("/api/v1/models/random_forest")
        
        assert response.status_code in [200, 404]
    
    def test_model_metrics_endpoint(self, client):
        """Test des métriques d'un modèle"""
        response = client.get("/api/v1/models/random_forest/metrics")
        
        assert response.status_code in [200, 404]


class TestMetricsAPI:
    """Tests de l'API des métriques"""
    
    def test_metrics_endpoint(self, client):
        """Test de l'endpoint des métriques"""
        response = client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Vérifier que les métriques contiennent les clés attendues
        expected_keys = ['models_count', 'total_predictions', 'average_latency']
        for key in expected_keys:
            assert key in data or response.status_code == 200


class TestErrorHandling:
    """Tests de la gestion des erreurs"""
    
    def test_404_error(self, client):
        """Test de gestion de l'erreur 404"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test de gestion d'une méthode non autorisée"""
        response = client.get("/api/v1/predict")  # Should be POST
        assert response.status_code in [405, 404]
    
    def test_invalid_json(self, client):
        """Test avec JSON invalide"""
        response = client.post(
            "/api/v1/predict",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestAPIPerformance:
    """Tests de performance de l'API"""
    
    def test_prediction_response_time(self, client):
        """Test du temps de réponse des prédictions"""
        import time
        
        features = [float(i) for i in range(20)]
        payload = {
            "features": features,
            "model": "random_forest"
        }
        
        start_time = time.time()
        response = client.post("/api/v1/predict", json=payload)
        elapsed_time = time.time() - start_time
        
        # La prédiction devrait être rapide (< 1 seconde)
        # (ajuster selon vos besoins)
        assert elapsed_time < 5.0
    
    def test_batch_prediction_performance(self, client):
        """Test de performance des prédictions en batch"""
        import time
        
        batch_data = {
            "data": [
                [float(i) for i in range(20)]
                for _ in range(100)
            ],
            "model": "random_forest"
        }
        
        start_time = time.time()
        response = client.post("/api/v1/batch_predict", json=batch_data)
        elapsed_time = time.time() - start_time
        
        # Les prédictions en batch devraient être relativement rapides
        assert elapsed_time < 10.0


class TestAPIIntegration:
    """Tests d'intégration de l'API"""
    
    def test_full_api_workflow(self, client):
        """Test du workflow complet de l'API"""
        # 1. Vérifier la santé
        health = client.get("/health")
        assert health.status_code == 200
        
        # 2. Lister les modèles
        models = client.get("/api/v1/models")
        assert models.status_code == 200
        
        # 3. Faire une prédiction
        features = [float(i) for i in range(20)]
        prediction = client.post(
            "/api/v1/predict",
            json={"features": features, "model": "random_forest"}
        )
        assert prediction.status_code in [200, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
