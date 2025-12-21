# 🌡️ Guide Complet de Démarrage - Climate MLOps

## 📋 Table des Matières
1. [Démarrage Rapide](#démarrage-rapide)
2. [Détails des Services](#détails-des-services)
3. [MLflow - Configuration et Utilisation](#mlflow---configuration-et-utilisation)
4. [Apache Airflow - Configuration et DAGs](#apache-airflow---configuration-et-dags)
5. [Dashboard - Guide Complet](#dashboard---guide-complet)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Démarrage Rapide


```bash
# Depuis le répertoire climate-mlops
./START.sh
```

Cela démarrera:
- ✅ **API FastAPI** (port 8000) avec Dashboard
- ✅ **MLflow** (port 5050) pour le tracking des modèles

### Option 2: Démarrage Complet avec Airflow (Docker Compose)

```bash
# Démarrage de TOUS les services (API + MLflow + Airflow + Databases)
./START.sh full
```

Cela démarrera:
- ✅ **API FastAPI** (port 8000)
- ✅ **MLflow** (port 5050)
- ✅ **Airflow Webserver** (port 8080) - Interface web
- ✅ **Airflow Scheduler** - Planification automatique
- ✅ **Airflow Worker** - Exécution des tâches
- ✅ **PostgreSQL** - Base de données Airflow
- ✅ **Redis** - Broker Celery pour les workers

### Arrêt des Services

```bash
# Option 1 (si utilisation de START.sh simple)
./STOP.sh

# Option 2 (si utilisation de Docker Compose)
docker-compose down
```

---

## 📊 Détails des Services

### 1. API FastAPI (Port 8000)

**URL**: http://localhost:8000

**Endpoints disponibles:**
- **Dashboard**: http://localhost:8000/dashboard
  - Visualisation des données météo
  - Graphiques en temps réel
  - Comparaison prédictions vs réalité
  
- **API Documentation**: http://localhost:8000/docs
  - Swagger UI complet
  - Testez les endpoints directement
  
- **Interface Web**: http://localhost:8000/web
  - Formulaire de prédiction manuelle
  
- **Health Check**: http://localhost:8000/health
  - Vérification de l'état de l'API

**Endpoints clés:**
```bash
# Santé de l'API
curl http://localhost:8000/health

# Lister les modèles disponibles
curl http://localhost:8000/models

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Year": 2024, "Month": 6, ...}}'
```

### 2. MLflow (Port 5050)

**URL**: http://localhost:5050

**Fonctionnalités:**
- Tracking des expériences d'entraînement
- Visualisation des métriques
- Gestion des versions de modèles
- Comparaison entre runs

**Structure:**
```
MLflow/
├── Expériences
│   └── Climate_Marrakech
│       ├── Run 1: RandomForest (R² = 0.98)
│       ├── Run 2: GradientBoosting (R² = 0.96)
│       └── Run 3: LinearRegression (baseline)
├── Métriques
│   ├── temperature_2m_mean_r2
│   ├── temperature_2m_max_rmse
│   └── ...
└── Artefacts
    ├── rf_model.pkl
    ├── scaler.pkl
    └── data_pipeline.joblib
```

---

## 🔄 MLflow - Configuration et Utilisation

### Configuration Automatique

MLflow est configuré automatiquement avec:
- **Backend Local**: `mlruns/` (fichiers locaux)
- **Expérience**: `Climate_Marrakech`
- **Artifacts**: `mlruns/artifacts/`

### Entraîner un Modèle et le Tracker dans MLflow

```bash
# Lancer l'entraînement complet (crée automatiquement des runs MLflow)
python src/train_model.py

# Ou spécifier une expérience personnalisée
python src/train_model.py \
  --mlflow-uri "http://localhost:5050" \
  --experiment-name "Custom_Experiment"
```

### Visualiser les Résultats dans MLflow

1. Ouvrez: http://localhost:5050
2. Allez dans **Experiments** → **Climate_Marrakech**
3. Comparez les modèles:
   - Cliquez sur les runs pour voir les détails
   - Comparez les métriques entre modèles
   - Téléchargez les modèles sauvegardés

### Récupérer un Modèle depuis MLflow

```python
import mlflow

# Se connecter à MLflow
mlflow.set_tracking_uri("file:./mlruns")

# Charger un modèle spécifique
model_uri = "runs:/RUN_ID/random_forest_model"
model = mlflow.sklearn.load_model(model_uri)

# Ou charger la version de production
model_uri = "models:/climate-model/production"
model = mlflow.sklearn.load_model(model_uri)
```

---

## 🔀 Apache Airflow - Configuration et DAGs

### 🚀 Démarrage d'Airflow

```bash
# Démarrer Airflow complet avec Docker Compose
./START.sh full

# Ou manuellement
docker-compose up -d airflow-postgres redis airflow-init airflow-webserver airflow-scheduler airflow-worker

# Attendre ~30 secondes que tout démarre
sleep 30
```

### 📊 Accès à l'Interface Airflow

**URL**: http://localhost:8080

**Identifiants:**
- **Username**: `admin`
- **Password**: `admin`

### 🏗️ Architecture d'Airflow

```
Airflow/
├── Webserver (Port 8080)          ← Interface web
├── Scheduler                       ← Planification des DAGs
├── Worker (Celery)                ← Exécution des tâches
├── PostgreSQL (Port 5432)         ← Base de données
├── Redis (Port 6379)              ← Message broker
│
└── DAG: climate_data_pipeline
    ├── step1_load_data            (Charger données brutes)
    ├── step2_preprocess_data      (Prétraiter)
    ├── step3_create_features      (Feature engineering)
    ├── step4_train_model          (Entraîner RandomForest)
    ├── step5_validate_api         (Vérifier API)
    └── notify_success             (Notification de succès)
```

### 🎯 Pipeline de Données Airflow

Le DAG exécute automatiquement le pipeline complet:

1. **step1_load_data** (5 min)
   - Charge les données brutes depuis `marrakech_weather_2018_2023_final.csv`
   - Logs dans MLflow: `raw_data_rows`, `raw_data_cols`

2. **step2_preprocess_data** (2 min)
   - Prétraite les données (nettoyage, normalisation)
   - Logs dans MLflow: `processed_data_rows`, `processed_data_cols`

3. **step3_create_features** (3 min)
   - Crée 49 features avancées
   - Logs dans MLflow: `features_rows`, `features_cols`

4. **step4_train_model** (10 min)
   - Entraîne le RandomForest
   - Logs dans MLflow: métriques R², RMSE, MAE

5. **step5_validate_api** (1 min)
   - Vérifie que l'API est opérationnelle

6. **notify_success**
   - Affiche un message de succès

**Durée totale**: ~20 minutes

### ⏰ Planification du DAG

Le DAG s'exécute **automatiquement chaque jour à minuit** (UTC).

**Modifier la fréquence:**

Éditez `airflow/dags/climate_pipeline_dag.py`:

```python
dag = DAG(
    'climate_data_pipeline',
    default_args=default_args,
    schedule_interval='@daily',          # ← Modifier ici
    # schedule_interval='@hourly',       # Horaire
    # schedule_interval='0 0 * * *',     # Personnalisé (minuit UTC)
    # schedule_interval='0 6 * * *',     # 6h du matin UTC
    catchup=False,
)
```

### 🎮 Contrôler le DAG depuis l'Interface Web

#### Activer/Désactiver le DAG

1. Allez à: http://localhost:8080
2. Recherchez le DAG: `climate_data_pipeline`
3. Cliquez sur le **toggle** pour activer/désactiver

#### Déclencher Manuellement

1. Allez à: http://localhost:8080/dags/climate_data_pipeline
2. Cliquez sur le bouton **"Trigger DAG"** (en haut à droite)
3. Optionnel: Entrez une date de démarrage personnalisée

#### Voir les Logs des Tâches

1. Allez à: http://localhost:8080/dags/climate_data_pipeline
2. Cliquez sur **"Graph View"** ou **"Tree View"**
3. Cliquez sur une tâche (rectangle)
4. Allez dans l'onglet **"Logs"**

#### Visualiser l'Exécution

1. **Tree View**: Timeline verticale de l'exécution
2. **Graph View**: DAG en diagramme
3. **Gantt Chart**: Timeline horizontale avec durées
4. **Calendar**: Historique des exécutions

### 🔗 Intégration Airflow + MLflow

Le DAG se connecte automatiquement à MLflow:

```python
mlflow.set_experiment("Climate_Marrakech_Airflow")
with mlflow.start_run(run_name="step1_load_data"):
    # Les métriques s'enregistrent automatiquement
    mlflow.log_metric("raw_data_rows", 2191)
```

**Visualiser dans MLflow:**
1. Ouvrez: http://localhost:5050
2. Allez à **Experiments** → **Climate_Marrakech_Airflow**
3. Cliquez sur les runs créés par Airflow

### 🐛 Dépanner les Tâches Airflow

#### Voir les Logs Détaillés

```bash
# Logs d'une tâche spécifique
docker-compose logs airflow-scheduler 2>&1 | tail -100

# Logs du worker
docker-compose logs airflow-worker 2>&1 | tail -100

# Tous les logs Airflow
docker-compose logs -f airflow-webserver airflow-scheduler airflow-worker
```

#### Tester le DAG Localement

```bash
# Tester une seule tâche
docker-compose exec airflow-webserver airflow tasks test climate_data_pipeline step1_load_data 2024-12-13

# Tester le DAG complet (sans scheduler)
docker-compose exec airflow-webserver airflow dags test climate_data_pipeline 2024-12-13
```

#### Vérifier la Connexion à PostgreSQL

```bash
# Accédez à la base de données Airflow
docker-compose exec airflow-postgres psql -U airflow -d airflow

# Lister les tables
\dt

# Lister les DAGs
select dag_id, is_paused from dag;

# Quitter
\q
```

#### Reset Complet d'Airflow

```bash
# ⚠️  ATTENTION: Cela supprime tous les données et logs!

docker-compose down -v

# Redémarrer
docker-compose up -d
```

---

## 📊 Dashboard - Guide Complet

### Accès au Dashboard

**URL**: http://localhost:8000/dashboard

### Fonctionnalités

#### 1. **KPI Cards** (Haut de page)
- Température actuelle (min/max/moyenne)
- Humidité relative
- Vitesse du vent
- Précipitations

#### 2. **Graphiques Interactifs**
- **Températures**: Évolution min/max/moyenne sur 7 jours
- **Humidité**: Variation quotidienne
- **Précipitations**: Cumuls quotidiens
- **Vent**: Vitesse maximale

#### 3. **Section Prédictions**
- Affiche les prédictions du modèle ML pour le prochain jour
- Comparaison avec les données réelles historiques

#### 4. **Tableau de Comparaison**
| Date | Réel Temp | Prédiction | Écart | Erreur % |
|------|-----------|------------|-------|----------|
| 2024-12-13 | 22.5°C | 22.8°C | +0.3°C | 1.3% |
| 2024-12-12 | 21.2°C | 21.0°C | -0.2°C | 0.9% |

#### 5. **Téléchargement**
- Bouton pour télécharger les graphiques en PNG
- Export des données en CSV (à venir)

### Mode Clair/Sombre

Basculez avec le bouton en haut à droite du dashboard.

Votre préférence est sauvegardée localement (localStorage).

### Filtrage des Données

Sélectionnez une plage de dates pour zoomer sur des périodes spécifiques.

---

## 🔧 Démarrage Complet Pas à Pas

### Étape 1: Installation des dépendances (première fois seulement)

```bash
# Option A: Pip (rapide)
pip install -r requirements.txt

# Option B: Conda (isolation complète)
conda create -n climate-mlops python=3.10
conda activate climate-mlops
pip install -r requirements.txt
```

### Étape 2: Démarrer les services

```bash
# Mode simple (API + MLflow uniquement)
./START.sh

# Attendez que tout soit prêt (~10 secondes)
# Vous verrez:
# ✅ MLflow démarré (PID: XXXX)
# ✅ API démarrée (PID: YYYY)
```

### Étape 3: Vérifier que tout fonctionne

```bash
# Vérifier l'API
curl http://localhost:8000/health

# Vérifier MLflow
curl http://localhost:5050/health || echo "MLflow prêt"
```

### Étape 4: Accédez aux interfaces

Ouvrez dans votre navigateur:
- **Dashboard**: http://localhost:8000/dashboard
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5050

### Étape 5: Entraîner un modèle (optionnel)

```bash
# Lance l'entraînement complet
# Les résultats s'afficheront automatiquement dans MLflow
python src/train_model.py
```

---

## 🔍 Vérification du Statut

### Afficher les services en cours d'exécution

```bash
# Voir tous les processus Python actifs
ps aux | grep -E "mlflow|uvicorn|python main"

# Voir les logs en temps réel
tail -f logs/api.log
tail -f logs/mlflow.log
```

### Tester les Endpoints API

```bash
# Health check
curl http://localhost:8000/health | jq

# Lister les modèles
curl http://localhost:8000/models | jq

# Prédiction simple
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Year": 2024,
      "Month": 12,
      "Quarter": 4,
      "DayOfYear": 347,
      "WeekOfYear": 50,
      "Month_sin": -0.866,
      "Month_cos": 0.5,
      "DayOfYear_sin": 0.9,
      "DayOfYear_cos": 0.43,
      "Temp_lag_1": 22.5,
      "Temp_lag_3": 23.1,
      "Temp_lag_7": 24.2,
      "Temp_lag_14": 25.0,
      "Temp_lag_30": 26.3,
      "Temp_ma_3": 23.0,
      "Temp_ma_7": 23.5,
      "Temp_ma_14": 24.0,
      "Temp_ma_30": 25.0,
      "Temp_trend_30d": 0.05,
      "Temp_volatility_7d": 1.2,
      "Temp_diff_1d": 0.3,
      "Temp_diff_7d": -0.5
    }
  }' | jq
```

---

## 🐛 Troubleshooting

### Problème: "Port 8000 déjà utilisé"

```bash
# Trouver le processus utilisant le port
lsof -i :8000

# Tuer le processus
kill -9 <PID>

# Ou utiliser un port différent
API_PORT=8001 python main.py
```

### Problème: "Port 5050 déjà utilisé"

```bash
# Arrêter les instances MLflow existantes
pkill -f "mlflow server"

# Ou utiliser un port différent
mlflow server --port 5051
```

### Problème: "Dataset non trouvé"

```bash
# Vérifier que le fichier existe
ls -lh marrakech_weather_2018_2023_final.csv

# S'il manque, télécharger depuis Kaggle
# Ou utiliser un autre fichier de données
```

### Problème: "Erreur d'import dans le code"

```bash
# S'assurer que vous êtes dans le bon répertoire
cd /Users/macadmin/Desktop/climate-mlops

# Vérifier les dépendances
pip list | grep -E "fastapi|mlflow|scikit-learn|pandas"

# Réinstaller si nécessaire
pip install -r requirements.txt --force-reinstall
```

### Problème: "MLflow ne se connecte pas à la base de données"

```bash
# Supprimer la base de données corrupted et recommencer
rm -rf mlruns/

# Redémarrer MLflow
./STOP.sh
./START.sh
```

### Problème: "Les modèles ne se sauvegardent pas"

```bash
# Vérifier que le dossier models/ existe
mkdir -p models

# Vérifier les permissions
chmod 755 models

# Vérifier l'espace disque
df -h
```

---

## 📚 Commandes Utiles

```bash
# Voir les logs en direct
tail -f logs/api.log

# Compter les lignes du dataset
wc -l marrakech_weather_2018_2023_final.csv

# Vérifier la structure du dataset
head -5 marrakech_weather_2018_2023_final.csv

# Lancer les tests
pytest tests/ -v

# Lancer une prédiction depuis Python
python -c "
from src.api import model_manager
import numpy as np
model_manager.load_models()
features = np.random.randn(1, 49)  # 49 features
result = model_manager.predict('random_forest', features)
print('Prédiction:', result)
"
```

---

## 🎯 Prochaines Étapes

1. **Entraîner le modèle**: `python src/train_model.py`
2. **Visualiser dans MLflow**: http://localhost:5050
3. **Faire des prédictions**: http://localhost:8000/dashboard
4. **Configurer Airflow**: Décommenter les services Docker Compose
5. **Mettre en production**: Utiliser docker-compose.prod.yml

---

## 📞 Support

Si vous avez des problèmes:
1. Vérifiez les logs: `tail -f logs/api.log`
2. Testez la connectivité: `curl http://localhost:8000/health`
3. Redémarrez les services: `./STOP.sh && ./START.sh`
4. Consultez le README.md pour plus de détails
