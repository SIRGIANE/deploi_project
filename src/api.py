"""
API FastAPI pour la prédiction de températures climatiques
Sert les modèles ML entraînés via des endpoints REST
"""

# Standard library imports
import logging
import os
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, Field

# Local imports
from src.data_pipeline import ClimateDataPipeline
from src.config import Config

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIDENCE_INTERVAL = Config.DEFAULT_CONFIDENCE_INTERVAL
DEFAULT_BASE_TEMPERATURE = Config.DEFAULT_BASE_TEMPERATURE
WARMING_TREND_PER_YEAR = Config.WARMING_TREND_PER_YEAR
SEASONAL_VARIATION_AMPLITUDE = Config.SEASONAL_VARIATION_AMPLITUDE
DEFAULT_VOLATILITY = Config.DEFAULT_VOLATILITY

class ModelType(str, Enum):
    """Types de modèles disponibles"""
    RANDOM_FOREST = "random_forest"
    FALLBACK = "fallback"

class PredictionFeatures(BaseModel):
    """Entrée attendue : dictionnaire de features déjà alignées sur le scaler"""
    features: Dict[str, float] = Field(..., description="Features numériques alignées sur le pipeline entraîné")

class BatchPredictionInput(BaseModel):
    """Structure d'entrée pour les prédictions par batch"""
    predictions: List[PredictionFeatures]
    model_name: str = ModelType.RANDOM_FOREST

class PredictionOutput(BaseModel):
    """Structure de sortie pour les prédictions multi-cibles"""
    predictions: Dict[str, float]
    model_used: str
    prediction_date: datetime
    input_features: Dict[str, Any]

class ModelInfo(BaseModel):
    """Information sur les modèles disponibles"""
    model_name: str
    model_type: str
    training_date: datetime
    target_names: List[str]
    feature_names: List[str]
    is_loaded: bool

class FallbackModel:
    """Modèle de fallback simple pour les prédictions"""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction basée sur une tendance linéaire simple"""
        if X.shape[0] == 0 or X.shape[1] < 2:
            raise ValueError("Features insuffisantes pour la prédiction")
            
        base_temp = DEFAULT_BASE_TEMPERATURE
        seasonal_variation = SEASONAL_VARIATION_AMPLITUDE * np.sin(2 * np.pi * X[:, 1] / 12)
        trend = (X[:, 0] - 2000) * WARMING_TREND_PER_YEAR
        
        return base_temp + seasonal_variation + trend

class ModelManager:
    """Gestionnaire des modèles ML"""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.pipeline = ClimateDataPipeline()
        self.feature_order: List[str] = []
        self.target_names: List[str] = []
        
    def load_models(self) -> None:
        """Chargement des modèles locaux (rf_model.pkl + scaler + pipeline)."""
        logger.info("🔄 Chargement des modèles locaux...")
        
        try:
            self._load_data_pipeline()
            self._load_local_rf_model()
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
        
        # Toujours créer un fallback si aucun modèle n'est chargé
        if not self.models:
            logger.warning("⚠️ Aucun modèle chargé, utilisation d'un modèle de fallback")
            self._create_fallback_model()
    
    def _load_data_pipeline(self) -> None:
        """Chargement du pipeline de données (scaler + feature/target names)."""
        if os.path.exists(Config.MODEL_PIPELINE_PATH):
            try:
                self.pipeline.load_pipeline(Config.MODEL_PIPELINE_PATH)
                self.feature_order = getattr(self.pipeline, "_feature_columns", [])
                self.target_names = getattr(self.pipeline, "_target_columns", [])
                # Use the correct feature order from config
                if hasattr(Config, 'FEATURE_COLUMNS'):
                    self.feature_order = Config.FEATURE_COLUMNS
                logger.info(f"✅ Pipeline de données chargé. Features: {len(self.feature_order)}, Targets: {self.target_names}")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de charger le pipeline: {e}")
    
    def _load_local_rf_model(self) -> None:
        """Chargement du modèle RandomForest local."""
        rf_path = Path(Config.MODELS_DIR) / "rf_model.pkl"
        if rf_path.exists():
            try:
                model = joblib.load(rf_path)
                self.models['random_forest'] = {
                    'model': model,
                    'type': 'sklearn',
                    'loaded_at': datetime.now()
                }
                logger.info("✅ Modèle RandomForest chargé depuis le disque")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de charger le modèle RandomForest: {e}")
    
    def _create_fallback_model(self) -> None:
        """Création d'un modèle de fallback simple"""        
        self.models['fallback'] = {
            'model': FallbackModel(),
            'type': 'fallback',
            'loaded_at': datetime.now()
        }
        logger.info("✅ Modèle de fallback créé")
    
    def predict(self, model_name: str, feature_array: np.ndarray) -> Dict[str, Any]:
        """Génération de prédictions multi-cibles."""
        if model_name not in self.models:
            available_models = list(self.models.keys())
            if available_models:
                model_name = available_models[0]
                logger.warning(f"Modèle demandé non trouvé, utilisation de {available_models[0]}")
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Aucun modèle disponible."
                )
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        try:
            preds = model.predict(feature_array)
            if preds.ndim == 1:
                preds = preds.reshape(1, -1)
            # Map predictions to target names
            targets = self.target_names or [f"target_{i}" for i in range(preds.shape[1])]
            prediction_dict = {name: float(preds[0, idx]) for idx, name in enumerate(targets)}
            
            return {
                'predictions': prediction_dict,
                'model_used': model_name,
                'model_type': model_info['type']
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur de prédiction: {str(e)}"
            )
    
    def prepare_features_array(self, features: Dict[str, float]) -> np.ndarray:
        """Prépare un array numpy ordonné selon les features du pipeline et applique le scaler."""
        if not self.feature_order:
            raise HTTPException(status_code=500, detail="Features du pipeline indisponibles")
        
        vector = np.array([[features.get(col, 0.0) for col in self.feature_order]])
        
        if self.pipeline.is_fitted:
            vector = self.pipeline.scaler.transform(vector)
        
        return vector

def fetch_current_weather() -> pd.DataFrame:
    """
    Récupère les données météo actuelles depuis l'API de prévision
    """
    logger.info("🌐 Récupération des données météo actuelles...")
    
    try:
        # Paramètres pour la prévision (aujourd'hui + demain)
        params = {
            "latitude": Config.MARRAKECH_LAT,
            "longitude": Config.MARRAKECH_LON,
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
            "timezone": "Africa/Casablanca",
            "forecast_days": 2  # Aujourd'hui + demain
        }
        
        # Requête API
        response = requests.get(Config.WEATHER_FORECAST_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Conversion en DataFrame
        df = pd.DataFrame({
            'time': data['hourly']['time'],
            'temperature_2m': data['hourly']['temperature_2m'],
            'apparent_temperature': data['hourly']['apparent_temperature'],
            'relative_humidity_2m': data['hourly']['relative_humidity_2m'],
            'precipitation': data['hourly']['precipitation'],
            'rain': data['hourly']['rain'],
            'snowfall': data['hourly']['snowfall'],
            'weathercode': data['hourly']['weathercode'],
            'windspeed_10m': data['hourly']['windspeed_10m'],
            'windgusts_10m': data['hourly']['windgusts_10m'],
            'winddirection_10m': data['hourly']['winddirection_10m'],
            'shortwave_radiation': data['hourly']['shortwave_radiation'],
            'et0_fao_evapotranspiration': data['hourly']['et0_fao_evapotranspiration']
        })
        
        # Conversion de la colonne time
        df['datetime'] = pd.to_datetime(df['time'])
        df['date'] = df['datetime'].dt.date
        
        # Agrégation quotidienne
        daily_df = df.groupby('date').agg(
            temperature_2m_max=('temperature_2m', 'max'),
            temperature_2m_min=('temperature_2m', 'min'),
            temperature_2m_mean=('temperature_2m', 'mean'),
            apparent_temperature_max=('apparent_temperature', 'max'),
            apparent_temperature_min=('apparent_temperature', 'min'),
            relative_humidity_2m=('relative_humidity_2m', 'mean'),
            precipitation_sum=('precipitation', 'sum'),
            rain_sum=('rain', 'sum'),
            snowfall_sum=('snowfall', 'sum'),
            precipitation_hours=('precipitation', lambda x: (x > 0).sum()),
            windspeed_10m_max=('windspeed_10m', 'max'),
            windgusts_10m_max=('windgusts_10m', 'max'),
            winddirection_10m_dominant=('winddirection_10m', lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean()),
            shortwave_radiation_sum=('shortwave_radiation', 'sum'),
            et0_fao_evapotranspiration=('et0_fao_evapotranspiration', 'sum'),
            weathercode=('weathercode', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        ).reset_index()
        
        # Ajouter les colonnes temporelles
        daily_df['datetime'] = pd.to_datetime(daily_df['date'])
        daily_df['year'] = daily_df['datetime'].dt.year
        daily_df['month'] = daily_df['datetime'].dt.month
        daily_df['day'] = daily_df['datetime'].dt.day
        daily_df['day_of_year'] = daily_df['datetime'].dt.dayofyear
        daily_df['season'] = daily_df['month'].apply(lambda m: 1 if m in [12,1,2] else 2 if m in [3,4,5] else 3 if m in [6,7,8] else 4)
        
        # Renommer date en time
        daily_df = daily_df.rename(columns={'date': 'time'})
        
        logger.info(f"✅ Données actuelles récupérées: {len(daily_df)} jours")
        return daily_df
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des données actuelles: {e}")
        raise

# Configuration de l'application
app = FastAPI(
    title=Config.API_TITLE,
    description="API de prédiction des températures climatiques basée sur des modèles ML",
    version=Config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Templates and static files
templates = Jinja2Templates(directory="src/templates")
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global historical data
historical_df = None

# Instance globale du gestionnaire
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'API"""
    global historical_df
    logger.info("🚀 Démarrage de l'API Climate Prediction")
    model_manager.load_models()
    historical_df = pd.read_csv("marrakech_weather_2018_2023_final.csv")
    historical_df['datetime'] = pd.to_datetime(historical_df['datetime'])
    logger.info("✅ API prête à recevoir des requêtes")

@app.get("/")
async def root():
    """Endpoint racine avec informations de l'API"""
    return {
        "message": Config.API_TITLE,
        "version": Config.API_VERSION,
        "status": "active",
        "available_endpoints": [
            "/predict",
            "/predict/batch",
            "/models",
            "/health",
            "/docs",
            "/web",
            "/dashboard"
        ]
    }

@app.get("/health")
async def health_check():
    """Vérification de la santé de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": Config.API_VERSION,
        "models_loaded": len(model_manager.models),
        "available_models": list(model_manager.models.keys()),
        "targets": model_manager.target_names,
        "features_expected": model_manager.feature_order
    }

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Liste des modèles disponibles"""
    models_info = []
    
    for name, info in model_manager.models.items():
        models_info.append(ModelInfo(
            model_name=name,
            model_type=info['type'],
            training_date=info['loaded_at'],
            target_names=model_manager.target_names,
            feature_names=model_manager.feature_order,
            is_loaded=True
        ))
    
    return models_info

# API v1 endpoints for tests (aliases)
@app.post("/api/v1/predict", response_model=PredictionOutput)
async def v1_predict(input_data: PredictionFeatures, model_name: str = ModelType.RANDOM_FOREST):
    return await predict_weather(input_data, model_name)

@app.post("/api/v1/batch_predict")
async def v1_batch_predict(input_data: BatchPredictionInput):
    return await predict_batch(input_data)

@app.get("/api/v1/models", response_model=List[ModelInfo])
async def v1_get_models():
    return await get_models()

@app.get("/api/v1/metrics")
async def v1_metrics():
    return {
        "uptime": 0,
        "models_loaded": len(model_manager.models),
        "available_models": list(model_manager.models.keys()),
        "version": Config.API_VERSION,
        "targets": model_manager.target_names,
        "features_expected": model_manager.feature_order
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_weather(
    input_data: PredictionFeatures,
    model_name: str = ModelType.RANDOM_FOREST
):
    """Prédiction multi-cibles à partir d'un vecteur de features complet."""
    try:
        logger.info(f"Prédiction demandée avec modèle {model_name}")
        
        feature_array = model_manager.prepare_features_array(input_data.features)
        result = model_manager.predict(model_name, feature_array)
        
        return PredictionOutput(
            predictions=result['predictions'],
            model_used=result['model_used'],
            prediction_date=datetime.now(),
            input_features=input_data.features
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(input_data: BatchPredictionInput):
    """Prédictions par batch à partir de vecteurs de features complets."""
    try:
        results = []
        
        for pred_input in input_data.predictions:
            feature_array = model_manager.prepare_features_array(pred_input.features)
            result = model_manager.predict(input_data.model_name, feature_array)
            
            results.append({
                "predictions": result['predictions'],
                "input": pred_input.features
            })
        
        return {
            "predictions": results,
            "model_used": input_data.model_name,
            "batch_size": len(results),
            "prediction_date": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors des prédictions batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Déclenchement d'un réentraînement des modèles en arrière-plan"""
    background_tasks.add_task(retrain_task)
    return {
        "message": "Réentraînement des modèles démarré en arrière-plan",
        "status": "started",
        "timestamp": datetime.now()
    }

async def retrain_task():
    """Tâche de réentraînement en arrière-plan"""
    try:
        logger.info("🔄 Démarrage du réentraînement...")
        
        # Ici vous pourriez relancer l'entraînement complet
        # pipeline = ClimateDataPipeline()
        # results = pipeline.run_full_pipeline()
        
        logger.info("✅ Réentraînement complété")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du réentraînement: {e}")

# Web UI endpoints
@app.get("/web", response_class=HTMLResponse)
async def web_home(request: Request):
    """Page d'accueil avec formulaire de prédiction"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/web/predict", response_class=HTMLResponse)
async def web_predict(
    request: Request,
    date: str = Form(...),
    temp_max: float = Form(...),
    temp_min: float = Form(...),
    temp_mean: float = Form(...),
    app_temp_max: float = Form(0),
    app_temp_min: float = Form(0),
    precip_sum: float = Form(0),
    rain_sum: float = Form(0),
    snowfall_sum: float = Form(0),
    precip_hours: int = Form(0),
    wind_speed_max: float = Form(0),
    wind_gusts_max: float = Form(0),
    wind_dir: float = Form(0),
    sw_rad_sum: float = Form(0),
    et0: float = Form(0),
    weathercode: int = Form(0),
    rel_humidity: float = Form(0)
):
    """Prédiction via formulaire web"""
    global historical_df
    
    dt = pd.to_datetime(date)
    new_row = {
        'time': date,
        'temperature_2m_max': temp_max,
        'temperature_2m_min': temp_min,
        'temperature_2m_mean': temp_mean,
        'apparent_temperature_max': app_temp_max,
        'apparent_temperature_min': app_temp_min,
        'precipitation_sum': precip_sum,
        'rain_sum': rain_sum,
        'snowfall_sum': snowfall_sum,
        'precipitation_hours': precip_hours,
        'windspeed_10m_max': wind_speed_max,
        'windgusts_10m_max': wind_gusts_max,
        'winddirection_10m_dominant': wind_dir,
        'shortwave_radiation_sum': sw_rad_sum,
        'et0_fao_evapotranspiration': et0,
        'weathercode': weathercode,
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'day_of_year': dt.dayofyear,
        'season': 1 if dt.month in [12,1,2] else 2 if dt.month in [3,4,5] else 3 if dt.month in [6,7,8] else 4,
        'datetime': dt,
        'relative_humidity_2m': rel_humidity
    }
    
    temp_df = historical_df.copy()
    temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Transform using pipeline
    transformed = model_manager.pipeline.transform(temp_df)
    last_features = transformed.iloc[-1][model_manager.feature_order].values.reshape(1, -1)
    if model_manager.pipeline.is_fitted:
        last_features = model_manager.pipeline.scaler.transform(last_features)
    result = model_manager.predict('random_forest', last_features)
    
    # Generate plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pd.to_datetime(historical_df['datetime']), historical_df['temperature_2m_mean'], label='Historical Mean Temp')
    ax.axhline(y=result['predictions'].get('temperature_2m_mean', temp_mean), color='r', linestyle='--', 
               label=f'Predicted Mean Temp: {result["predictions"].get("temperature_2m_mean", temp_mean):.2f}°C')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Historical Temperature and Prediction')
    ax.legend()
    plot_path = 'src/static/plot.png'
    fig.savefig(plot_path)
    plt.close()
    
    return templates.TemplateResponse("result.html", {
        "request": request, 
        "prediction": result['predictions'], 
        "plot_url": "/static/plot.png"
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Tableau de bord avec météo actuelle et prédictions"""
    try:
        # Récupérer les données actuelles
        current_weather = fetch_current_weather()
        
        # Prendre les données d'aujourd'hui
        today = datetime.now().date()
        today_data = current_weather[current_weather['time'] == today]
        
        if today_data.empty:
            today_data = current_weather.iloc[0]
        else:
            today_data = today_data.iloc[0]
        
        # Préparer les features pour la prédiction
        features = today_data.to_dict()
        features.pop('time', None)
        # Keep 'datetime' for feature engineering
        
        # Transformer et prédire
        temp_df = historical_df.copy()
        temp_df = pd.concat([temp_df, pd.DataFrame([features])], ignore_index=True)
        transformed = model_manager.pipeline.transform(temp_df)
        last_features = transformed.iloc[-1][model_manager.feature_order].values.reshape(1, -1)
        if model_manager.pipeline.is_fitted:
            last_features = model_manager.pipeline.scaler.transform(last_features)
        prediction = model_manager.predict('random_forest', last_features)
        
        # Générer le graphique
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Données historiques récentes
        recent_history = historical_df.tail(30)  # 30 derniers jours
        ax.plot(pd.to_datetime(recent_history['datetime']), recent_history['temperature_2m_mean'], 
                label='Historical Mean Temp', color='blue')
        
        # Aujourd'hui
        ax.axvline(x=pd.to_datetime(today), color='green', linestyle='--', label='Today')
        ax.scatter(pd.to_datetime(today), today_data['temperature_2m_mean'], 
                  color='orange', s=100, label=f'Today Forecast: {today_data["temperature_2m_mean"]:.1f}°C')
        
        # Prédiction du modèle
        predicted_temp = prediction['predictions'].get('temperature_2m_mean', today_data['temperature_2m_mean'])
        ax.scatter(pd.to_datetime(today), predicted_temp, 
                  color='red', s=100, marker='x', label=f'Model Prediction: {predicted_temp:.1f}°C')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Climate Dashboard: Historical Data, Today\'s Forecast & Model Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = 'src/static/dashboard_plot.png'
        fig.savefig(plot_path)
        plt.close()
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "today_data": today_data.to_dict(),
            "prediction": prediction['predictions'],
            "plot_url": "/static/dashboard_plot.png",
            "current_date": today.strftime("%Y-%m-%d")
        })
        
    except Exception as e:
        logger.error(f"Erreur dashboard: {e}")
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "error": str(e),
            "current_date": datetime.now().strftime("%Y-%m-%d")
        })

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Démarrage de l'API FastAPI")
    logger.info(f"📍 Adresse: {Config.API_HOST}:{Config.API_PORT}")
    logger.info(f"📚 Documentation: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    logger.info(f"🌐 Interface Web: http://{Config.API_HOST}:{Config.API_PORT}/web")
    logger.info(f"📊 Dashboard: http://{Config.API_HOST}:{Config.API_PORT}/dashboard")
    
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level=Config.LOG_LEVEL.lower()
    )
