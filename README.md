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

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository>
cd climate-mlops
```

### 2. Start the Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Wait for services to initialize (2-3 minutes)
docker-compose logs -f
```

### 3. Access the Applications

| Service | URL | Credentials |
|---------|-----|-------------|
| **Dashboard** | http://localhost:8000/dashboard | - |
| **FastAPI** | http://localhost:8000 | - |
| **Airflow** | http://localhost:8080 | `airflow` / `airflow` |
| **PostgreSQL** | localhost:5433 | `postgres` / `postgres` |

### 4. Verify the Setup

```bash
# Check if all containers are running
docker-compose ps

# Test the API
curl http://localhost:8000/health

# Check Airflow DAGs
curl http://localhost:8080/health
```

## 📊 Dashboard Features

The interactive dashboard (`http://localhost:8000/dashboard`) provides:

- **📈 Temperature Trends**: Min, Max, and Mean temperature charts
- **💧 Humidity Tracking**: Relative humidity over time  
- **🌧️ Precipitation Data**: Daily rainfall measurements
- **💨 Wind Speed**: Wind speed and direction analysis
- **🤖 ML Predictions**: Model predictions vs actual values
- **🌓 Dark/Light Mode**: Toggle between themes
- **📥 Data Export**: Download data as JSON

## 🔄 Automated Pipeline

The system runs automatically with the following schedule:

### Daily (6:00 AM)
```
✅ Collect weather data from Open-Meteo API
✅ Store in PostgreSQL + CSV backup  
✅ Validate data quality
✅ Check if retraining needed
```

### Weekly (Every 7 days)
```
🤖 Preprocess data (outliers, features)
🤖 Train 3 ML models in parallel:
   📊 Linear Regression (baseline)
   🌲 Random Forest (with Optuna optimization)  
   📈 Gradient Boosting
🤖 Compare model performances (RMSE, MAE, R²)
🤖 Select and save best performing model
🤖 Update training metadata
```

## 📁 Project Structure

```
climate-mlops/
├── 📂 src/                          # Source code
│   ├── api.py                       # FastAPI backend
│   ├── marrakech_data_loader.py     # Data collection & processing
│   ├── data_pipeline.py             # ML data pipeline
│   ├── train_model.py               # Model training
│   ├── config.py                    # Configuration settings
│   ├── templates/                   # HTML templates
│   └── static/                      # Static assets
├── 📂 airflow/
│   └── dags/
│       └── climate_pipeline_dag.py  # Airflow DAG
├── 📂 data/                         # Data storage
│   ├── raw/                         # Raw weather data
│   ├── processed/                   # Cleaned data
│   ├── features/                    # ML features
│   └── cumulative_weather_data.csv  # Main dataset
├── 📂 models/                       # Trained models
│   ├── rf_model.pkl                 # RandomForest model
│   ├── scaler.pkl                   # Data scaler
│   └── data_pipeline.joblib         # Pipeline object
├── 📂 results/                      # Training results
│   └── weather_training_results.json
├── 📂 mlruns/                       # MLflow experiments
├── 📂 tests/                        # Unit tests
├── docker-compose.yml              # Docker services
├── Dockerfile.airflow              # Airflow container
├── requirements.txt                # Python dependencies
├── START.sh                        # Quick start script
├── STOP.sh                         # Stop services script
└── TEST.sh                         # Run tests script
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):

```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=weather_db

# API
API_HOST=0.0.0.0
API_PORT=8000

# Airflow
AIRFLOW_UID=50000
```

### Marrakech Coordinates

The system is configured for Marrakech, Morocco:
- **Latitude**: 31.6295°N
- **Longitude**: 7.9811°W
- **Timezone**: Africa/Casablanca

## 🧪 Manual Testing

### Test Data Collection

```bash
# Trigger manual data collection
curl -X POST http://localhost:8000/api/v1/collection/trigger

# Check collection status
curl http://localhost:8000/api/v1/collection/status
```

### Test Model Prediction

```bash
# Get weather prediction (uses best performing model)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature_2m_max": 25.5,
    "temperature_2m_min": 15.2,
    "precipitation_sum": 0.0,
    "windspeed_10m_max": 12.8
  }'
```

### Test Airflow DAG

```bash
# Access Airflow UI: http://localhost:8080
# Username: airflow, Password: airflow

# Or trigger via CLI
docker-compose exec airflow-webserver airflow dags trigger climate_data_pipeline
```

## 🛠️ Development

### Using Helper Scripts

```bash
# Start all services
./START.sh

# Stop all services  
./STOP.sh

# Run tests
./TEST.sh
```

### Manual Development Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run API locally (without Docker)
python main.py

# Run data collection manually
python collect_today_data.py

# Run tests
pytest tests/ -v
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

## 🚀 Production Deployment

For production deployment:

```bash
# Use production Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up --scale weather-api=3 -d

# Monitor resources
docker stats
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
