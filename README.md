# 🌦️ Climate MLOps - Marrakech Weather Prediction System

A complete MLOps pipeline for weather prediction in Marrakech, Morocco. This system automatically collects weather data, trains multiple machine learning models, and provides real-time predictions through a beautiful dashboard.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Open-Meteo    │───▶│   Data Pipeline │───▶│   ML Training   │
│      API        │    │   (Airflow)     │    │ (3 ML Models)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   FastAPI       │    │   Dashboard     │
│   Database      │    │   Backend       │    │  (Chart.js)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Features

- **📅 Daily Data Collection**: Automatic weather data collection from Open-Meteo API
- **🤖 Weekly Model Training**: 3 ML models retrained every 7 days with automatic comparison
- **🏆 Model Selection**: Automatic best model selection based on performance metrics
- **📊 Interactive Dashboard**: Real-time weather metrics with beautiful charts
- **🔄 MLOps Pipeline**: Complete pipeline with Airflow orchestration
- **🗄️ Dual Storage**: PostgreSQL database + CSV files backup
- **🚀 FastAPI Backend**: REST API for predictions and data access
- **🐳 Docker Ready**: Containerized deployment with Docker Compose

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Data Source** | Open-Meteo Weather API |
| **Orchestration** | Apache Airflow |
| **Database** | PostgreSQL |
| **ML Models** | RandomForest + Gradient Boosting + Linear Regression |
| **ML Framework** | Scikit-learn |
| **Hyperparameter Optimization** | Optuna |
| **Backend API** | FastAPI |
| **Frontend** | HTML5 + Chart.js |
| **Containerization** | Docker + Docker Compose |
| **Data Processing** | Pandas + NumPy |
| **Experiment Tracking** | MLflow |

## 📋 Prerequisites

- Docker & Docker Compose
- Python 3.9+
- 8GB RAM minimum
- 10GB disk space

## 🚀 Quick Start (Architecture Unifiée)

Ce projet utilise désormais une **Image Docker Unique** pour tous les services, simplifiant le déploiement et la maintenance.

### 1. Démarrer le projet
```bash
make up
```
Cette commande va :
- Construire l'image unifiée (si nécessaire)
- Lancer tous les services (API, Airflow, MLflow, Jupyter)
- Vérifier automatiquement la santé du système

### 2. Commandes Utiles (Makefile)

| Commande | Action |
|----------|--------|
| `make up` | Démarrer tous les services |
| `make down` | Arrêter les services |
| `make restart` | Redémarrage complet |
| `make logs` | Voir les logs en direct |
| `make status` | Vérifier l'état des services |
| `make test` | Lancer les tests unitaires |
| `make clean` | Tout nettoyer (supprime les données) |

### 3. Accès aux Services

| Service | URL | Credentials | Docker Image |
|---------|-----|-------------|--------------|
| **Dashboard** | http://localhost:8000/dashboard | - | `climate-mlops:latest` |
| **API Docs** | http://localhost:8000/docs | - | `climate-mlops:latest` |
| **Airflow** | http://localhost:8080 | `admin` / `admin` | `climate-mlops:latest` |
| **MLflow** | http://localhost:5050 | - | `climate-mlops:latest` |
| **Jupyter** | http://localhost:8889 | (Token vide) | `climate-mlops:latest` |

### 4. Vérification Manuelle
Vous pouvez aussi lancer le script de vérification :
```bash
python3 verify_deployment.py
```

## 📊 Data Pipeline Details

### 1. Data Collection (`src/marrakech_data_loader.py`)
- Fetches data from Open-Meteo API
- Handles historical data loading (2018-2023 baseline)
- Manages PostgreSQL connections with auto-detection
- Implements data validation and quality checks

### 2. Data Processing (`src/data_pipeline.py`)
- Outlier detection using IQR method
- Missing value interpolation
- Feature engineering (temporal, lag, moving averages)
- Data normalization and scaling

### 3. Model Training (`src/train_model.py`)
- **Multi-Model Training**: RandomForest, Gradient Boosting, Linear Regression
- **Hyperparameter Optimization**: Optuna for RandomForest tuning
- **Automatic Model Comparison**: Performance-based selection
- **Cross-validation**: Time series splits for robust evaluation
- **Model Evaluation**: Comprehensive metrics (RMSE, MAE, R²)
- **MLflow Tracking**: Experiment logging and model registry

### 4. API Backend (`src/api.py`)
- FastAPI REST endpoints
- Real-time predictions
- Data export functionality
- Health checks and monitoring

## 🐛 Troubleshooting

### Common Issues

**1. Containers not starting**
```bash
# Check Docker resources
docker system df
docker system prune

# Restart services
docker-compose down
docker-compose up -d
```

**2. Database connection errors**
```bash
# Check PostgreSQL logs
docker-compose logs weather-db

# Verify connection
docker-compose exec weather-db psql -U postgres -d weather_db -c "\dt"
```

**3. Airflow scheduler issues**
```bash
# Restart Airflow
docker-compose restart airflow-scheduler

# Check DAG status
docker-compose logs airflow-scheduler
```

**4. API not responding**
```bash
# Check FastAPI logs
docker-compose logs weather-api

# Restart API service
docker-compose restart weather-api
```

### Logs and Monitoring

```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f weather-api
docker-compose logs -f airflow-scheduler
docker-compose logs -f weather-db

# Check application logs
tail -f logs/api.log
tail -f logs/mlflow.log
```

## 📊 ML Models & Performance

The system trains and compares **3 different models** automatically:

### 🤖 **Model Ensemble**

| Model | Type | Features | Auto-Tuning |
|-------|------|----------|-------------|
| **🌲 Random Forest** | Ensemble | Feature importance analysis | ✅ Optuna optimization |
| **📈 Gradient Boosting** | Ensemble | Sequential learning | ✅ Pre-configured params |
| **📊 Linear Regression** | Baseline | Fast & interpretable | ➖ No tuning needed |

### 🏆 **Automatic Model Selection**

The system automatically:
- Trains all 3 models in parallel
- Evaluates performance using cross-validation
- Compares RMSE, MAE, and R² scores
- Selects the best performing model
- Saves the winner for predictions

### 📈 **Performance Metrics**

Each model is evaluated on:
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)  
- **R²**: Coefficient of determination (higher is better)
- **Training Time**: Model efficiency comparison

## 📈 Performance Metrics

The system tracks the following metrics for each model:

- **Model Performance**: RMSE, MAE, R² score for each model
- **Model Comparison**: Automatic best model selection
- **Hyperparameter Tracking**: Optuna optimization results
- **Training Efficiency**: Training time and resource usage
- **Data Quality**: Missing values, outliers, data freshness
- **Pipeline Health**: Success rate, execution time
- **API Performance**: Response time, error rate

## 🔄 Backup & Recovery

### Data Backup
```bash
# Backup PostgreSQL
docker-compose exec weather-db pg_dump -U postgres weather_db > backup.sql

# Backup CSV files
cp -r data/ backup_data/

# Backup models
cp -r models/ backup_models/
```

### Recovery
```bash
# Restore PostgreSQL
docker-compose exec -i weather-db psql -U postgres weather_db < backup.sql

# Restore CSV files
cp -r backup_data/* data/

# Restore models
cp -r backup_models/* models/
```

## 📚 API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/dashboard` | Interactive dashboard |
| `POST` | `/predict` | Weather prediction (best model) |
| `GET` | `/api/v1/data/latest` | Latest weather data |
| `POST` | `/api/v1/collection/trigger` | Trigger data collection |
| `GET` | `/api/v1/models/info` | Current model information |
| `GET` | `/api/v1/models/performance` | All models performance |
| `GET` | `/api/v1/metrics` | System metrics |

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v
pytest tests/test_data_pipeline.py -v
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 MLflow Integration

Track experiments and models:

```bash
# View MLflow UI (if running)
# Navigate to: http://localhost:5000

# Check experiment runs
ls mlruns/

# View specific experiment
cat mlruns/*/meta.yaml
```

## 🚀 Azure Cloud Deployment

Ce projet est optimisé pour un déploiement sur **Azure Container Apps**.

### 1. Configuration Initiale
```bash
az login --use-device-code
az account set --subscription "1815cb03-0ab6-4382-9f78-d03c507c84e4"

export RESOURCE_GROUP="rg-projet"
export LOCATION="switzerlandnorth"
export ACR_NAME="climatemlopsreg$(date +%s)"
```

### 2. Build & Push
```bash
# Créer le registre
az acr create -g $RESOURCE_GROUP -n $ACR_NAME --sku Basic --admin-enabled true

# Pousser l'image
az acr login --name $ACR_NAME
export ACR_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
docker tag climate-mlops-weather-api:latest $ACR_SERVER/climate-mlops:latest
docker push $AC_SERVER/climate-mlops:latest
```

### 3. Déploiement
```bash
# Déployer sur Container Apps
az containerapp create \
  --name weather-api \
  --resource-group $RESOURCE_GROUP \
  --environment climate-mlops-env \
  --image $ACR_SERVER/climate-mlops:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 --memory 2.0Gi
```

### 4. Nettoyage
```bash
# Tout supprimer après usage
az group delete --name $RESOURCE_GROUP --yes --no-wait
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests (`./TEST.sh`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Open-Meteo** for providing free weather data API
- **Apache Airflow** for workflow orchestration
- **FastAPI** for the modern API framework
- **Chart.js** for beautiful data visualization
- **MLflow** for experiment tracking

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Docker Compose logs: `docker-compose logs -f`
3. Check application logs in the `logs/` directory
4. Open an issue on GitHub

## 🏆 Project Status

- ✅ **Data Collection**: Automated daily collection from Open-Meteo API
- ✅ **Multi-Model Training**: RandomForest + Gradient Boosting + Linear Regression
- ✅ **Model Selection**: Automatic best model selection with performance comparison
- ✅ **Hyperparameter Tuning**: Optuna optimization for RandomForest
- ✅ **API**: FastAPI backend with real-time predictions  
- ✅ **Dashboard**: Interactive web interface with Chart.js
- ✅ **Orchestration**: Airflow DAG scheduling
- ✅ **Storage**: PostgreSQL + CSV dual storage
- ✅ **Monitoring**: Comprehensive logging and health checks
- ✅ **Testing**: Unit tests with pytest
- ✅ **Docker**: Containerized deployment

---

**🎉 Congratulations! You now have a complete MLOps weather prediction system with 3 ML models running automatically!**

The system will collect weather data daily at 6:00 AM, retrain all 3 models every 7 days, automatically select the best performer, and serve predictions without any manual intervention.
