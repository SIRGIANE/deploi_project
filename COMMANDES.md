# 🚀 Commandes Essentielles - Climate MLOps

## ⚡ Démarrage Rapide

```bash
# Mode SIMPLE (API + MLflow en local) - ⭐ RECOMMANDÉ POUR MACOS
./START.sh

# Mode COMPLET (Airflow + Docker Compose)
./START.sh full

# Arrêter les services
./STOP.sh

# Tests de vérification
./TEST.sh
```

---

## 🌐 URLs d'Accès

### Mode Simple
| Service | URL | Identifiants |
|---------|-----|--------------|
| 📊 Dashboard | http://localhost:8000/dashboard | - |
| 📚 API Docs | http://localhost:8000/docs | - |
| 🔬 MLflow | http://localhost:5050 | - |
| 🌐 Interface Web | http://localhost:8000/web | - |
| 💚 Health Check | http://localhost:8000/health | - |

### Mode Complet (+ Airflow)
| Service | URL | Identifiants |
|---------|-----|--------------|
| 🔀 Airflow | http://localhost:8080 | admin / admin |
| 📊 Dashboard | http://localhost:8000/dashboard | - |
| 🔬 MLflow | http://localhost:5050 | - |

---

## 📊 Services lancés

### Mode Simple
```
✅ API FastAPI (port 8000)
✅ MLflow (port 5050)
```

### Mode Complet
```
✅ API FastAPI (port 8000)
✅ MLflow (port 5050)
✅ Airflow Webserver (port 8080)
✅ Airflow Scheduler
✅ Airflow Worker
✅ PostgreSQL (port 5432)
✅ Redis (port 6379)
```

---

## 🔄 Workflow Typique

### 1️⃣ Démarrer
```bash
./START.sh
# Attendre 10-15 secondes
```

### 2️⃣ Entraîner un modèle
```bash
python src/train_model.py
# Les résultats s'enregistrent automatiquement dans MLflow
```

### 3️⃣ Visualiser dans MLflow
```
Ouvrir: http://localhost:5050
Aller à: Experiments → Climate_Marrakech
```

### 4️⃣ Faire des prédictions
```
Ouvrir: http://localhost:8000/dashboard
Voir les prédictions et graphiques
```

### 5️⃣ Arrêter
```bash
./STOP.sh
```

---

## 🐳 Commandes Docker Compose

```bash
# Voir le statut de tous les services
docker-compose ps

# Voir les logs
docker-compose logs -f

# Logs d'un service spécifique
docker-compose logs -f airflow-webserver

# Redémarrer un service
docker-compose restart airflow-scheduler

# Arrêter complètement
docker-compose down

# Arrêter et supprimer les données
docker-compose down -v
```

---

## 🔄 Airflow - Commandes Essentielles

```bash
# Lister les DAGs
docker-compose exec airflow-webserver airflow dags list

# Activer un DAG
docker-compose exec airflow-webserver airflow dags unpause climate_data_pipeline

# Désactiver un DAG
docker-compose exec airflow-webserver airflow dags pause climate_data_pipeline

# Tester une tâche
docker-compose exec airflow-webserver airflow tasks test climate_data_pipeline step1_load_data 2024-12-13

# Voir les logs d'une tâche
docker-compose exec airflow-webserver airflow tasks logs climate_data_pipeline step1_load_data 2024-12-13

# Déclencher le DAG manuellement
docker-compose exec airflow-webserver airflow dags trigger climate_data_pipeline
```

---

## 🔬 MLflow - Commandes Essentielles

```bash
# Lister les expériences
mlflow experiments list

# Lister les runs
mlflow runs list --experiment-id 1

# Voir les détails d'un run
mlflow runs describe <RUN_ID>

# Accéder à l'UI
# http://localhost:5050
```

---

## 🧪 Tests et Vérification

```bash
# Test complet du système
./TEST.sh

# Health check de l'API
curl http://localhost:8000/health | jq

# Lister les modèles
curl http://localhost:8000/models | jq

# Tester une prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Year": 2024, "Month": 12, "Quarter": 4, "DayOfYear": 347, "WeekOfYear": 50, "Month_sin": -0.866, "Month_cos": 0.5, "DayOfYear_sin": 0.9, "DayOfYear_cos": 0.43, "Temp_lag_1": 22.5, "Temp_lag_3": 23.1, "Temp_lag_7": 24.2, "Temp_lag_14": 25.0, "Temp_lag_30": 26.3, "Temp_ma_3": 23.0, "Temp_ma_7": 23.5, "Temp_ma_14": 24.0, "Temp_ma_30": 25.0, "Temp_trend_30d": 0.05, "Temp_volatility_7d": 1.2, "Temp_diff_1d": 0.3, "Temp_diff_7d": -0.5}}' | jq
```

---

## 📋 Fichiers de Configuration

- 📄 `.env` - Variables d'environnement
- 📄 `docker-compose.yml` - Configuration Docker Compose
- 📄 `requirements.txt` - Dépendances Python
- 📄 `params.yaml` - Paramètres du modèle
- 📁 `airflow/dags/climate_pipeline_dag.py` - DAG Airflow

---

## 🐛 Troubleshooting Rapide

### Port déjà utilisé
```bash
# Trouver le processus
lsof -i :8000

# Tuer le processus
kill -9 <PID>
```

### Réinitialiser MLflow
```bash
rm -rf mlruns/
./STOP.sh
./START.sh
```

### Réinitialiser Airflow
```bash
docker-compose down -v
docker-compose up -d
```

### Vérifier les logs
```bash
tail -f logs/api.log
tail -f logs/mlflow.log
docker-compose logs -f
```

---

## 📚 Documentation Complète

Voir `GUIDE_DEMARRAGE.md` pour la documentation complète :
- Configuration détaillée
- Guide d'utilisation d'Airflow
- Troubleshooting avancé
- Monitoring en production
