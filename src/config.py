"""
Configuration centralisée pour le projet Climate MLOps
Contient toutes les constantes et paramètres de configuration
"""
import os
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

class Config:
    """Configuration centralisée de l'application"""
    
    # === CHEMINS ET RÉPERTOIRES ===
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = os.getenv(
        "DATA_PATH", 
        str(PROJECT_ROOT / "marrakech_weather_2018_2023_final.csv")
    )
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    LOGS_DIR = "logs"
    
    # === MLFLOW CONFIGURATION ===
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Marrakech_Weather_Prediction")
    
    # === API CONFIGURATION ===
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8001"))
    API_TITLE = "Marrakech Weather Prediction API"
    API_VERSION = "2.0.0"
    
    #=== DATA PROCESSING ===
    REQUIRED_COLUMNS = ['time', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'temperature_2m_mean (°C)']
    DEFAULT_SPLIT_DATE = os.getenv("SPLIT_DATE", "2022-01-01")
    MAX_MISSING_THRESHOLD = 0.5  # 50% max de valeurs manquantes
    MIN_DATA_SIZE_AFTER_CLEANING = 100
    
    #=== FEATURE ENGINEERING ===
    LAG_PERIODS = [1, 3, 7, 14, 30]
    MOVING_AVERAGE_WINDOWS = [3, 7, 14, 30]
    TREND_WINDOW = 30
    VOLATILITY_WINDOW = 7
    
    FEATURE_COLUMNS = [
        'Year', 'Month', 'Quarter', 'DayOfYear', 'WeekOfYear',
        'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos',
        'Temp_lag_1', 'Temp_lag_3', 'Temp_lag_7', 'Temp_lag_14', 'Temp_lag_30',
        'Temp_ma_3', 'Temp_ma_7', 'Temp_ma_14', 'Temp_ma_30',
        'Temp_trend_30d', 'Temp_volatility_7d',
        'Temp_diff_1d', 'Temp_diff_7d'
    ]
    
    # === MODEL PARAMETERS ===
    DEFAULT_RF_PARAMS = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
    
    OPTUNA_RF_SEARCH_SPACE = {
        'n_estimators': (50, 500),
        'max_depth': (5, 30),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    }
    
    # === PREDICTION CONSTANTS ===
    DEFAULT_CONFIDENCE_INTERVAL = 0.5
    DEFAULT_BASE_TEMPERATURE = 20.0  # Température moyenne de Marrakech
    WARMING_TREND_PER_YEAR = 0.02
    SEASONAL_VARIATION_AMPLITUDE = 5.0
    DEFAULT_VOLATILITY = 1.5
    YEAR_VALIDATION_RANGE = (2018, 2024)
    MONTH_VALIDATION_RANGE = (1, 12)
    
    # === CACHE CONFIGURATION ===
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 heure
    
    # === LOGGING CONFIGURATION ===
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # === FEATURE FLAGS ===
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    ENABLE_DATA_VALIDATION = os.getenv("ENABLE_DATA_VALIDATION", "true").lower() == "true"
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    # === FILE PATHS ===
    MODEL_PIPELINE_PATH = f"{MODELS_DIR}/data_pipeline.joblib"
    TRAINING_RESULTS_PATH = f"{RESULTS_DIR}/weather_training_results.json"
    
    # === WEATHER API CONFIGURATION ===
    WEATHER_API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    MARRAKECH_LAT = 31.6295
    MARRAKECH_LON = -7.9811
    WEATHER_API_PARAMS = {
        "latitude": MARRAKECH_LAT,
        "longitude": MARRAKECH_LON,
        "hourly": [
            "temperature_2m", 
            "apparent_temperature", 
            "relative_humidity_2m", 
            "precipitation", 
            "rain", 
            "snowfall", 
            "weathercode", 
            "windspeed_10m", 
            "windgusts_10m", 
            "winddirection_10m", 
            "shortwave_radiation", 
            "et0_fao_evapotranspiration"
        ],
        "timezone": "Africa/Casablanca"
    }
    
    # === DATA STORAGE CONFIGURATION ===
    DATA_STORAGE_DIR = os.getenv("DATA_STORAGE_DIR", "data/incoming")
    DATA_ARCHIVE_DIR = os.getenv("DATA_ARCHIVE_DIR", "data/archive")
    DATA_BACKUP_DIR = os.getenv("DATA_BACKUP_DIR", "data/backups")
    
    # Base de données PostgreSQL (production)
    DATABASE_URL = os.getenv(
        "DATABASE_URL", 
        "postgresql://user:password@localhost:5433/weather_data"
    )
    
    # Configuration de l'archivage
    ARCHIVE_ENABLED = os.getenv("ARCHIVE_ENABLED", "true").lower() == "true"
    ARCHIVE_INTERVAL_DAYS = int(os.getenv("ARCHIVE_INTERVAL_DAYS", "30"))
    
    # === CONTINUOUS TRAINING CONFIGURATION ===
    
    # Configuration du retraining automatique
    RETRAINING_ENABLED = os.getenv("RETRAINING_ENABLED", "true").lower() == "true"
    RETRAINING_INTERVAL_DAYS = int(os.getenv("RETRAINING_INTERVAL_DAYS", "7"))
    NEW_DATA_BUFFER_SIZE = int(os.getenv("NEW_DATA_BUFFER_SIZE", "7"))  # Attendre 7 jours avant retraining
    
    # === DATA INGESTION CONFIGURATION ===
    DATA_INGESTION_ENABLED = os.getenv("DATA_INGESTION_ENABLED", "true").lower() == "true"
    DATA_INGESTION_SCHEDULE = os.getenv("DATA_INGESTION_SCHEDULE", "@weekly")  # Format Airflow
    INGESTION_BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", "1000"))
    INGESTION_TIMEOUT_SECONDS = int(os.getenv("INGESTION_TIMEOUT_SECONDS", "3600"))
    INGESTION_RETRIES = int(os.getenv("INGESTION_RETRIES", "3"))
    
    # === DATA DRIFT DETECTION CONFIGURATION ===
    DATA_DRIFT_DETECTION_ENABLED = os.getenv("DATA_DRIFT_DETECTION_ENABLED", "true").lower() == "true"
    DRIFT_DETECTION_METHOD = os.getenv("DRIFT_DETECTION_METHOD", "statistical")  # statistical, ml-based, unsupervised
    DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
    DRIFT_WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "30"))  # jours pour calculer les stats
    DRIFT_MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "100"))
    
    # Seuils de drift par métrique
    DRIFT_THRESHOLDS = {
        'kolmogorov_smirnov': 0.15,  # Test KS
        'chi_square': 0.05,           # Test chi-carré
        'wasserstein': 0.2,           # Distance Wasserstein
        'psi': 0.1,                   # Population Stability Index
        'js_divergence': 0.15         # Jensen-Shannon divergence
    }
    
    # === MODEL EVALUATION CONFIGURATION ===
    MODEL_EVALUATION_ENABLED = os.getenv("MODEL_EVALUATION_ENABLED", "true").lower() == "true"
    MODEL_PERFORMANCE_THRESHOLD = float(os.getenv("MODEL_PERFORMANCE_THRESHOLD", "0.05"))  # 5% amélioration requise
    EVALUATION_METRICS = [
        'rmse', 'mae', 'r2_score', 'mape', 'mean_absolute_percentage_error'
    ]
    
    # Critères de promotion du modèle
    MODEL_PROMOTION_CRITERIA = {
        'min_rmse_improvement': 0.05,       # 5% minimum
        'min_r2_improvement': 0.02,         # 2% minimum
        'min_data_points': 100,             # Au moins 100 points d'évaluation
        'max_training_time_seconds': 3600,  # Max 1 heure d'entraînement
        'require_positive_tests': True      # Tous les tests doivent passer
    }
    
    # === MODEL COMPARISON CONFIGURATION ===
    ENABLE_MODEL_COMPARISON = os.getenv("ENABLE_MODEL_COMPARISON", "true").lower() == "true"
    COMPARISON_METRICS = ['rmse', 'mae', 'r2_score', 'mean_absolute_percentage_error']
    COMPARISON_WINDOW_DAYS = int(os.getenv("COMPARISON_WINDOW_DAYS", "7"))
    
    # === ENHANCED MODEL SELECTION ===
    MODEL_SELECTION_WEIGHTS = {
        'rmse': float(os.getenv("MODEL_SELECTION_WEIGHT_RMSE", "0.4")),    # 40% - Primary error metric
        'r2': float(os.getenv("MODEL_SELECTION_WEIGHT_R2", "0.3")),        # 30% - Goodness of fit
        'mae': float(os.getenv("MODEL_SELECTION_WEIGHT_MAE", "0.2")),      # 20% - Absolute error
        'time': float(os.getenv("MODEL_SELECTION_WEIGHT_TIME", "0.1"))     # 10% - Training speed
    }
    
    # Advanced model promotion criteria
    MODEL_PROMOTION_CRITERIA = {
        'min_rmse_improvement': float(os.getenv("MIN_RMSE_IMPROVEMENT", "0.05")),        # 5% minimum RMSE improvement
        'min_r2_improvement': float(os.getenv("MIN_R2_IMPROVEMENT", "0.02")),            # 2% minimum R² improvement
        'min_r2_threshold': float(os.getenv("MIN_R2_THRESHOLD", "0.7")),                # Minimum acceptable R²
        'min_data_points': int(os.getenv("MIN_DATA_POINTS", "100")),                    # Minimum test samples
        'max_training_time_seconds': int(os.getenv("MAX_TRAINING_TIME", "3600")),       # Max 1 hour training
        'require_positive_tests': os.getenv("REQUIRE_ALL_TESTS", "true").lower() == "true",  # All checks must pass
        'min_score_improvement': float(os.getenv("MIN_SCORE_IMPROVEMENT", "5.0")),      # Minimum composite score improvement
        'stability_check_enabled': os.getenv("STABILITY_CHECK", "true").lower() == "true"  # Check model stability
    }
    
    # Model deployment strategies
    DEPLOYMENT_STRATEGIES = {
        'auto_deploy_on_improvement': os.getenv("AUTO_DEPLOY", "false").lower() == "true",
        'require_manual_approval': os.getenv("MANUAL_APPROVAL", "true").lower() == "true",
        'rollback_on_failure': os.getenv("ROLLBACK_ENABLED", "true").lower() == "true",
        'shadow_mode_duration_days': int(os.getenv("SHADOW_MODE_DAYS", "3"))  # Test in shadow mode first
    }
    
    # === MONITORING & ALERTING CONFIGURATION ===
    MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    ALERT_ENABLED = os.getenv("ALERT_ENABLED", "true").lower() == "true"
    
    # Seuils d'alerte
    ALERT_THRESHOLDS = {
        'drift_score': 0.5,               # Score de drift critique
        'model_performance_degradation': 0.1,  # 10% dégradation
        'data_quality_score': 0.7,        # Score de qualité minimal
        'prediction_latency_ms': 1000,    # Latence maximale
        'error_rate': 0.05                # Taux d'erreur maximal (5%)
    }
    
    # Canaux de notification
    NOTIFICATION_CHANNELS = {
        'email': os.getenv("NOTIFICATION_EMAIL_ENABLED", "false").lower() == "true",
        'slack': os.getenv("NOTIFICATION_SLACK_ENABLED", "false").lower() == "true",
        'webhook': os.getenv("NOTIFICATION_WEBHOOK_ENABLED", "false").lower() == "true",
        'logging': os.getenv("NOTIFICATION_LOGGING_ENABLED", "true").lower() == "true"
    }
    
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL", "mlops-team@example.com")
    
    # === PIPELINE ORCHESTRATION CONFIGURATION ===
    PIPELINE_SCHEDULE = os.getenv("PIPELINE_SCHEDULE", "@weekly")
    PIPELINE_MAX_ACTIVE_RUNS = int(os.getenv("PIPELINE_MAX_ACTIVE_RUNS", "1"))
    PIPELINE_DEFAULT_RETRIES = int(os.getenv("PIPELINE_DEFAULT_RETRIES", "2"))
    PIPELINE_RETRY_DELAY_MINUTES = int(os.getenv("PIPELINE_RETRY_DELAY_MINUTES", "5"))
    PIPELINE_TIMEOUT_MINUTES = int(os.getenv("PIPELINE_TIMEOUT_MINUTES", "360"))  # 6 heures
    
    # Tâches optionnelles du pipeline
    PIPELINE_TASKS = {
        'ingest_data': os.getenv("TASK_INGEST_DATA_ENABLED", "true").lower() == "true",
        'check_drift': os.getenv("TASK_CHECK_DRIFT_ENABLED", "true").lower() == "true",
        'validate_data': os.getenv("TASK_VALIDATE_DATA_ENABLED", "true").lower() == "true",
        'run_dvc_pipeline': os.getenv("TASK_RUN_DVC_ENABLED", "true").lower() == "true",
        'train_models': os.getenv("TASK_TRAIN_MODELS_ENABLED", "true").lower() == "true",
        'evaluate_models': os.getenv("TASK_EVALUATE_MODELS_ENABLED", "true").lower() == "true",
        'register_model': os.getenv("TASK_REGISTER_MODEL_ENABLED", "true").lower() == "true",
        'update_docs': os.getenv("TASK_UPDATE_DOCS_ENABLED", "true").lower() == "true",
        'push_to_github': os.getenv("TASK_PUSH_GITHUB_ENABLED", "false").lower() == "true",
        'send_notification': os.getenv("TASK_SEND_NOTIFICATION_ENABLED", "true").lower() == "true"
    }
    
    # === MODEL REGISTRY CONFIGURATION ===
    MODEL_REGISTRY_ENABLED = os.getenv("MODEL_REGISTRY_ENABLED", "true").lower() == "true"
    MODEL_REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", "models/registry")
    MODEL_REGISTRY_BACKEND = os.getenv("MODEL_REGISTRY_BACKEND", "mlflow")  # mlflow, local, s3
    
    # Versioning des modèles
    MODEL_VERSION_PREFIX = "v"
    MODEL_STAGING_PATH = f"{MODEL_REGISTRY_PATH}/staging"
    MODEL_PRODUCTION_PATH = f"{MODEL_REGISTRY_PATH}/production"
    MODEL_ARCHIVE_PATH = f"{MODEL_REGISTRY_PATH}/archive"
    
    # === A/B TESTING CONFIGURATION ===
    AB_TESTING_ENABLED = os.getenv("AB_TESTING_ENABLED", "false").lower() == "true"
    AB_TEST_TRAFFIC_SPLIT = float(os.getenv("AB_TEST_TRAFFIC_SPLIT", "0.5"))  # 50/50 par défaut
    AB_TEST_MIN_DURATION_DAYS = int(os.getenv("AB_TEST_MIN_DURATION_DAYS", "7"))
    AB_TEST_MIN_SAMPLES = int(os.getenv("AB_TEST_MIN_SAMPLES", "1000"))
    
    # === DATA QUALITY MONITORING ===
    DATA_QUALITY_ENABLED = os.getenv("DATA_QUALITY_ENABLED", "true").lower() == "true"
    
    DATA_QUALITY_CHECKS = {
        'null_check': os.getenv("DQ_NULL_CHECK_ENABLED", "true").lower() == "true",
        'duplicate_check': os.getenv("DQ_DUPLICATE_CHECK_ENABLED", "true").lower() == "true",
        'range_check': os.getenv("DQ_RANGE_CHECK_ENABLED", "true").lower() == "true",
        'anomaly_detection': os.getenv("DQ_ANOMALY_DETECTION_ENABLED", "true").lower() == "true",
        'schema_validation': os.getenv("DQ_SCHEMA_VALIDATION_ENABLED", "true").lower() == "true"
    }
    
    # Seuils de qualité des données
    DATA_QUALITY_THRESHOLDS = {
        'max_null_percentage': 0.1,        # 10% max de valeurs nulles
        'max_duplicate_percentage': 0.05,  # 5% max de doublons
        'max_outlier_percentage': 0.05,    # 5% max d'anomalies
        'min_completeness_score': 0.9      # 90% minimum de complétude
    }
    
    # === TRAINING JOB CONFIGURATION ===
    TRAINING_PARALLELIZATION = os.getenv("TRAINING_PARALLELIZATION", "true").lower() == "true"
    TRAINING_NUM_PARALLEL_JOBS = int(os.getenv("TRAINING_NUM_PARALLEL_JOBS", "4"))
    TRAINING_CROSS_VALIDATION_FOLDS = int(os.getenv("TRAINING_CV_FOLDS", "5"))
    TRAINING_TEST_SIZE = float(os.getenv("TRAINING_TEST_SIZE", "0.2"))
    TRAINING_VALIDATION_SIZE = float(os.getenv("TRAINING_VALIDATION_SIZE", "0.1"))
    
    # === HYPERPARAMETER OPTIMIZATION ===
    HYPERPARAMETER_OPTIMIZATION_ENABLED = os.getenv("HPO_ENABLED", "true").lower() == "true"
    HYPERPARAMETER_OPTIMIZATION_METHOD = os.getenv("HPO_METHOD", "optuna")  # optuna, grid_search, random_search
    HYPERPARAMETER_OPTIMIZATION_TRIALS = int(os.getenv("HPO_TRIALS", "100"))
    HYPERPARAMETER_OPTIMIZATION_TIMEOUT_SECONDS = int(os.getenv("HPO_TIMEOUT", "7200"))  # 2 heures
    
    # === MODEL EXPLAINABILITY CONFIGURATION ===
    MODEL_EXPLAINABILITY_ENABLED = os.getenv("MODEL_EXPLAINABILITY_ENABLED", "true").lower() == "true"
    EXPLAINABILITY_METHOD = os.getenv("EXPLAINABILITY_METHOD", "shap")  # shap, lime, permutation
    FEATURE_IMPORTANCE_TOP_K = int(os.getenv("FEATURE_IMPORTANCE_TOP_K", "10"))
    
    # === BACKUP & VERSIONING ===
    BACKUP_ENABLED = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
    BACKUP_INTERVAL_DAYS = int(os.getenv("BACKUP_INTERVAL_DAYS", "7"))
    BACKUP_RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS", "90"))
    MODEL_BACKUP_PATH = os.getenv("MODEL_BACKUP_PATH", "models/backups")
    DATA_BACKUP_RETENTION_DAYS = int(os.getenv("DATA_BACKUP_RETENTION_DAYS", "30"))
    
    # === PERFORMANCE PROFILING ===
    PERFORMANCE_PROFILING_ENABLED = os.getenv("PERFORMANCE_PROFILING_ENABLED", "false").lower() == "true"
    PROFILE_TRAINING = os.getenv("PROFILE_TRAINING", "false").lower() == "true"
    PROFILE_INFERENCE = os.getenv("PROFILE_INFERENCE", "false").lower() == "true"
    
    # === EXPERIMENT TRACKING ===
    EXPERIMENT_TRACKING_ENABLED = os.getenv("EXPERIMENT_TRACKING_ENABLED", "true").lower() == "true"
    TRACK_ARTIFACTS = os.getenv("TRACK_ARTIFACTS", "true").lower() == "true"
    TRACK_SYSTEM_METRICS = os.getenv("TRACK_SYSTEM_METRICS", "true").lower() == "true"
    ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "mlruns/artifacts")
    
    # === PRODUCTION DEPLOYMENT CONFIGURATION ===
    DEPLOYMENT_ENABLED = os.getenv("DEPLOYMENT_ENABLED", "false").lower() == "true"
    DEPLOYMENT_ENVIRONMENT = os.getenv("DEPLOYMENT_ENVIRONMENT", "staging")  # staging, production
    DEPLOYMENT_STRATEGY = os.getenv("DEPLOYMENT_STRATEGY", "blue_green")  # blue_green, canary, rolling
    CANARY_TRAFFIC_PERCENTAGE = int(os.getenv("CANARY_TRAFFIC_PERCENTAGE", "10"))
    DEPLOYMENT_ROLLBACK_ENABLED = os.getenv("DEPLOYMENT_ROLLBACK_ENABLED", "true").lower() == "true"
    
    @classmethod
    def get_data_file_path(cls) -> Path:
        """Retourne le chemin vers le fichier de données principal"""
        return Path(cls.DATA_PATH)
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Création des répertoires nécessaires s'ils n'existent pas"""
        for directory in [cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def ensure_data_directories(cls) -> None:
        """Création des répertoires de données s'ils n'existent pas"""
        for directory in [cls.DATA_STORAGE_DIR, cls.DATA_ARCHIVE_DIR, cls.DATA_BACKUP_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validation de la configuration avec retour des erreurs"""
        errors = []
        
        # Validation des chemins
        data_path = Path(cls.DATA_PATH)
        if not data_path.exists():
            errors.append(f"Fichier de données introuvable: {cls.DATA_PATH}")
        
        # Validation des plages
        if cls.MAX_MISSING_THRESHOLD < 0 or cls.MAX_MISSING_THRESHOLD > 1:
            errors.append("MAX_MISSING_THRESHOLD doit être entre 0 et 1")
        
        # Validation des paramètres de fenêtre
        if cls.TREND_WINDOW <= 0 or cls.VOLATILITY_WINDOW <= 0:
            errors.append("Les fenêtres de tendance et volatilité doivent être positives")
        
        return errors
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Conversion de la configuration en dictionnaire"""
        return {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }

# Configuration spécialisée pour les tests
class TestConfig(Config):
    """Configuration pour les tests unitaires"""
    
    # Override pour les tests
    DATA_PATH = "tests/data/marrakech_weather_test.csv"
    MODELS_DIR = "tests/models"
    RESULTS_DIR = "tests/results"
    MLFLOW_TRACKING_URI = "file:./tests/mlruns"
    
    # Paramètres réduits pour les tests rapides
    DEFAULT_RF_PARAMS = {
        'n_estimators': 10,
        'max_depth': 5,
        'random_state': 42
    }
    
    LAG_PERIODS = [1, 3, 7]
    MOVING_AVERAGE_WINDOWS = [3, 7]

# Instance globale de configuration
config = Config()