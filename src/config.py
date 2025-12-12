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
    API_PORT = int(os.getenv("API_PORT", "8000"))
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