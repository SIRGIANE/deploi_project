# 🌡️ Climate MLOps - Prédiction de Températures Climatiques

Un projet MLOps complet pour la prédiction des températures climatiques utilisant des données historiques de Berkeley Earth.

## 🎯 Objectifs du Projet

- **Analyse exploratoire** des données climatiques (1750-2015)
- **Développement de modèles ML** (Random Forest, LSTM, ARIMA)
- **Pipeline de données automatisé** avec validation
- **API FastAPI** pour servir les prédictions
- **Tracking des expériences** avec MLflow
- **Déploiement containerisé** avec Docker

## 🏗️ Architecture du Projet

```
climate-mlops/
├── 📊 01_exploratory_analysis.ipynb    # Analyse exploratoire
├── 🤖 02_model_development.ipynb       # Développement des modèles
├── src/
│   ├── 🔧 data_pipeline.py             # Pipeline de données
│   ├── 🚀 train_model.py               # Script d'entraînement
│   └── 🌐 api.py                       # API FastAPI
├── 🐳 docker-compose.yml               # Développement
├── 🐳 docker-compose.prod.yml          # Production
├── 📦 requirements.txt                 # Dépendances Python
└── 📋 README.md                        # Documentation
```

## 🚀 Démarrage Rapide

### 1. Prérequis
```bash
# Docker et Docker Compose installés
docker --version
docker-compose --version
```

### 2. Cloner et démarrer
```bash
git clone <your-repo>
cd climate-mlops

# Démarrage de l'environnement de développement
docker-compose up -d

# Ou pour la production
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Accès aux services
- **📊 Jupyter Lab** : http://localhost:8889
- **📈 MLflow** : http://localhost:5050  
- **🌐 API** : http://localhost:8000 (production uniquement)
- **📚 Documentation API** : http://localhost:8000/docs

## 📊 Utilisation

### Analyse Exploratoire
1. Ouvrez Jupyter Lab (http://localhost:8889)
2. Exécutez `01_exploratory_analysis.ipynb`
3. Visualisez les tendances climatiques historiques

### Développement de Modèles
1. Exécutez `02_model_development.ipynb`
2. Suivez l'entraînement des modèles :
   - **Random Forest** : Modèle d'ensemble robuste
   - **LSTM** : Réseau de neurones pour séries temporelles
   - **ARIMA** : Modèle statistique classique
3. Consultez MLflow pour comparer les performances

### Entraînement Automatisé
```bash
# Entraînement de base
docker-compose exec jupyter python src/train_model.py

# Avec optimisation des hyperparamètres
docker-compose exec jupyter python src/train_model.py --optimize --trials 100
```

### API de Prédiction
```bash
# Test de l'API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2025,
    "month": 12,
    "use_lag_features": true
  }'

# Prédictions par batch
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {"year": 2025, "month": 1},
      {"year": 2025, "month": 6},
      {"year": 2025, "month": 12}
    ],
    "model_name": "random_forest"
  }'
```

## 🔧 Pipeline de Données

Le pipeline automatisé (`src/data_pipeline.py`) effectue :

1. **Chargement** : Données Kaggle Berkeley Earth
2. **Validation** : Contrôle qualité automatique
3. **Nettoyage** : Traitement des valeurs manquantes
4. **Feature Engineering** :
   - Features temporelles (année, mois, saison)
   - Features cycliques (sin/cos pour saisonnalité)
   - Features de lag (1, 3, 6, 12 mois)
   - Moyennes mobiles (3, 6, 12 mois)
   - Tendances et volatilité
5. **Normalisation** : StandardScaler pour ML
6. **Division** : Train/Test temporel (2010 comme split)

## 🤖 Modèles Disponibles

### Random Forest
- **Type** : Ensemble learning
- **Avantages** : Robuste, interprétable
- **Features** : Importance des variables
- **Performance** : RMSE ~0.5°C

### LSTM (Deep Learning)
- **Type** : Réseau de neurones récurrent
- **Avantages** : Capture les dépendances temporelles
- **Architecture** : 2 couches LSTM + Dense
- **Séquences** : 12 mois de contexte

### Régression Linéaire (Baseline)
- **Type** : Modèle de référence
- **Usage** : Comparaison de performance
- **Simplicité** : Interprétation facile

## 📈 MLflow Tracking

Toutes les expériences sont trackées automatiquement :

- **Paramètres** : Hyperparamètres des modèles
- **Métriques** : RMSE, MAE, R²
- **Artifacts** : Modèles sauvegardés
- **Comparaison** : Interface web intuitive

```python
# Accès programmatique
import mlflow
mlflow.set_tracking_uri("http://localhost:5050")
runs = mlflow.search_runs(experiment_ids=["1"])
```

## 🌐 API Documentation

### Endpoints Principaux

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info de l'API |
| `/health` | GET | Status de santé |
| `/models` | GET | Liste des modèles |
| `/predict` | POST | Prédiction unique |
| `/predict/batch` | POST | Prédictions multiples |
| `/retrain` | POST | Réentraînement |

### Exemple de Réponse
```json
{
  "predicted_temperature": 9.23,
  "confidence_interval": {
    "lower": 8.73,
    "upper": 9.73
  },
  "model_used": "random_forest",
  "prediction_date": "2024-12-02T15:30:00",
  "input_features": {
    "year": 2025,
    "month": 12,
    "model_type": "sklearn"
  }
}
```

## 🔄 Optimisation des Hyperparamètres

Utilisation d'**Optuna** pour l'optimisation automatique :

```python
# Dans train_model.py
best_params = {
    'n_estimators': 250,
    'max_depth': 20,
    'min_samples_split': 3,
    'min_samples_leaf': 1
}
```

## 📊 Métriques de Performance

### Évaluation
- **RMSE** : Root Mean Square Error
- **MAE** : Mean Absolute Error  
- **R²** : Coefficient de détermination
- **Validation temporelle** : Split chronologique

### Résultats Typiques
- **Random Forest** : RMSE ~0.4°C, R² ~0.95
- **LSTM** : RMSE ~0.5°C, R² ~0.93
- **Baseline** : RMSE ~0.8°C, R² ~0.85

## 🐳 Déploiement

### Développement
```bash
docker-compose up -d
# Services : Jupyter + MLflow
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
# Services : API + MLflow + Jupyter + Scheduler
```

### Monitoring
- **Health checks** automatiques
- **Restart policies** configurées
- **Logs** centralisés avec Docker

## 🔧 Configuration

### Variables d'Environnement
```env
MLFLOW_TRACKING_URI=http://localhost:5050
PYTHONPATH=/workspace/src
```

### Ports
- **8889** : Jupyter Lab
- **5050** : MLflow UI  
- **8000** : API FastAPI

## 📝 Développement

### Installation locale
```bash
pip install -r requirements.txt
```

### Tests
```bash
pytest tests/
```

### Formatage du code
```bash
black src/
flake8 src/
```

## 🚀 Prochaines Étapes

1. **Tests unitaires** : Couverture complète
2. **CI/CD** : GitHub Actions
3. **Monitoring avancé** : Prometheus + Grafana
4. **Data drift detection** : Evidently AI
5. **A/B Testing** : Comparaison de modèles en production
6. **Scaling** : Kubernetes deployment

## 📚 Ressources

- **Données** : [Berkeley Earth](http://berkeleyearth.org/data/)
- **MLflow** : [Documentation](https://mlflow.org/docs/latest/index.html)
- **FastAPI** : [Guide](https://fastapi.tiangolo.com/)
- **Optuna** : [Tutoriels](https://optuna.readthedocs.io/)

## 🤝 Contribution

1. Fork le projet
2. Créez une branch (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Créez une Pull Request

## 📄 Licence

MIT License - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

**🎯 Projet développé dans le cadre d'un apprentissage MLOps appliqué aux données climatiques**





┌─────────────────────────────────────────────────────────┐
│  Push Code → GitHub Actions CI/CD Pipeline             │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
    Tests (tests.yml)  Train (train.yml)  Docker (docker.yml)
        │                  │                  │
   ✅ Unit Tests      ✅ DVC Pull      ✅ Security Scan
   ✅ Linting         ✅ Train Models   ✅ Build Images
   ✅ Coverage        ✅ Evaluate       ✅ Push to Docker Hub
                      ✅ MLflow Log
                      ✅ Register Best
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              Staging       Production