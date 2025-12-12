# Climate MLOps - Prédiction Météo Marrakech 🌤️

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLOps](https://img.shields.io/badge/MLOps-Enabled-green.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)]()

Projet MLOps avancé pour la prédiction météorologique de **Marrakech** (2018-2023) avec des pratiques DevOps/MLOps modernes.

## 🎯 Objectif

Prédire les variables météorologiques de Marrakech en utilisant des données historiques (2018-2023) :
- **Températures** : min, max, moyenne, température ressentie
- **Précipitations** : cumuls journaliers et hebdomadaires
- **Vent** : vitesse maximale et moyenne
- **Autres** : humidité, pression atmosphérique

## 📊 Dataset

**Source** : `marrakech_weather_2018_2023_final.csv`
- **Période** : 2018-01-01 → 2023-12-31 (6 ans)
- **Fréquence** : Données journalières
- **Volume** : 2191 observations
- **Variables** : 21 features météorologiques

### Variables principales :
- `temperature_2m_max/min/mean (°C)` - Températures quotidiennes
- `apparent_temperature_max/min (°C)` - Températures ressenties
- `precipitation_sum (mm)` - Précipitations totales
- `rain_sum (mm)` - Pluie totale
- `wind_speed_10m_max (km/h)` - Vitesse maximale du vent
- `wind_gusts_10m_max (km/h)` - Rafales maximales

## 🚀 Quick Start

### 1. Installation

```bash
# Cloner le repo
git clone https://github.com/SIRGIANE/climate-mlops
cd climate-mlops

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Exécution du pipeline de données

```bash
# Pipeline complet : chargement → preprocessing → features
python src/data_pipeline.py
```

**Sortie attendue** :
- `data/raw/weather_data_raw.csv` - Données brutes
- `data/processed/weather_data_processed.csv` - Données nettoyées
- `data/features/weather_data_features.csv` - Features enrichies (49 colonnes)

### 3. Entraînement du modèle

```bash
# Test rapide avec RandomForest
python test_marrakech_model.py

# OU entraînement complet avec MLflow
python src/train_model.py
```

### 4. API de prédiction

```bash
# Lancer l'API FastAPI
python src/api.py

# L'API sera disponible sur http://localhost:8000
# Documentation : http://localhost:8000/docs
```

## 📂 Architecture du Projet

```
climate-mlops/
├── marrakech_weather_2018_2023_final.csv   # Dataset principal
├── src/
│   ├── config.py                           # Configuration centralisée
│   ├── marrakech_data_loader.py           # Chargeur de données Marrakech
│   ├── data_pipeline.py                    # Pipeline de traitement (3 étapes)
│   ├── train_model.py                      # Entraînement des modèles
│   ├── api.py                              # API FastAPI
│   └── evaluate_model.py                   # Évaluation et métriques
├── data/
│   ├── raw/                                # Données brutes
│   ├── processed/                          # Données préprocessées
│   └── features/                           # Features ML + matrices numpy
├── models/                                 # Modèles entraînés (.pkl)
├── mlruns/                                 # Expériences MLflow
├── notebooks/                              # Notebooks d'exploration
└── tests/                                  # Tests unitaires

```

## 🔧 Pipeline de Données (3 Étapes)

### Étape 1 : Chargement des données 📥
```python
from src.data_pipeline import WeatherDataPipeline

pipeline = WeatherDataPipeline()
raw_data = pipeline.step1_download_raw_data()
# Résultat : (2191, 22)
```

### Étape 2 : Preprocessing 🔧
- Nettoyage des valeurs manquantes (interpolation linéaire)
- Détection et traitement des outliers (méthode IQR)
- Suppression des doublons
- Tri chronologique

```python
processed_data = pipeline.step2_preprocess_data(raw_data)
# Résultat : (2191, 22) - Données nettoyées
```

### Étape 3 : Feature Engineering 🎯
- **Features temporelles** (11) : Year, Month, Day, Quarter, sin/cos cycliques
- **Lag features** (5) : Retards de 1, 3, 7, 14, 30 jours
- **Moving averages** (4) : Moyennes mobiles sur 3, 7, 14, 30 jours
- **Tendance/Volatilité** (2) : Tendance sur 30j, volatilité sur 7j
- **Features de précipitations** (2) : Cumuls sur 7 et 30 jours
- **Features de vent** (2) : Moyennes mobiles sur 7 jours

```python
features_data = pipeline.step3_create_features(processed_data)
# Résultat : (2161, 49) - Features enrichies
```

## 🤖 Modèles ML

### RandomForest (par défaut)
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
```

### Résultats attendus :
- **temperature_2m_mean** : R² = 0.9843 (RMSE: 0.85°C)
- **temperature_2m_min** : R² = 0.9055 (RMSE: 1.82°C)
- **apparent_temperature_min** : R² = 0.8772 (RMSE: 2.47°C)

## 🐳 Docker

```bash
# Build et lancement avec Docker Compose
docker-compose up --build

# Services disponibles :
# - API : http://localhost:8000
# - MLflow : http://localhost:5050
# - Airflow : http://localhost:8080
```

## 📊 MLflow Tracking

```bash
# Lancer le serveur MLflow
mlflow server --host 0.0.0.0 --port 5050

# Interface : http://localhost:5050
```

## 🔍 Monitoring et Évaluation

### Vérification du drift de données
```bash
python src/check_data_drift.py
```

### Génération de Model Card
```bash
python src/generate_model_card.py
```

### Comparaison de modèles
```bash
python src/model_comparison.py
```

## 🧪 Tests

```bash
# Tests unitaires
pytest tests/

# Test du pipeline complet
python src/data_pipeline.py

# Test d'entraînement rapide
python test_marrakech_model.py
```

## 📈 Utilisation de l'API

### Exemple de requête :

```python
import requests

# Prédiction de température
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "year": 2024,
        "month": 6,
        "day": 15
    }
)

print(response.json())
# {
#   "temperature_predicted": 28.5,
#   "confidence_interval": [26.8, 30.2],
#   "model_version": "1.0.0"
# }
```

### Documentation interactive :
📚 **Swagger UI** : http://localhost:8000/docs

## 🔧 Configuration

Fichier `src/config.py` :
```python
DATA_PATH = "marrakech_weather_2018_2023_final.csv"
MLFLOW_TRACKING_URI = "http://localhost:5050"
MLFLOW_EXPERIMENT_NAME = "Marrakech_Weather_Prediction"
```

Variables d'environnement supportées :
- `DATA_PATH` - Chemin vers le dataset
- `MLFLOW_TRACKING_URI` - URI du serveur MLflow
- `API_PORT` - Port de l'API (défaut: 8000)

## 🛠️ Technologies

- **Python 3.8+** - Langage principal
- **Pandas / NumPy** - Manipulation de données
- **Scikit-learn** - Machine Learning
- **MLflow** - Tracking et gestion des modèles
- **FastAPI** - API REST
- **Docker** - Conteneurisation
- **Airflow** - Orchestration (optionnel)
- **DVC** - Versioning des données (optionnel)

## 📝 Notes de Migration

Ce projet utilise désormais le dataset **Marrakech Weather 2018-2023** au lieu du dataset Kaggle global.

### Avantages :
✅ Données locales (pas de téléchargement Kaggle requis)  
✅ Focus géographique sur Marrakech  
✅ Période récente (2018-2023)  
✅ 21 variables météorologiques complètes  
✅ Données journalières (2191 jours)  

### Fichiers modifiés :
- `src/config.py` - Configuration du chemin de données
- `src/marrakech_data_loader.py` - Nouveau loader créé
- `src/data_pipeline.py` - Utilisation du nouveau loader

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📄 Licence

MIT License

## 👥 Auteurs

Climate MLOps Team

---

**Note** : Pour toute question, consultez la documentation dans `/docs` ou ouvrez une issue.
