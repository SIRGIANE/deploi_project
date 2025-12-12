"""
Module pour l'ingestion des données météo dans la base de données
"""
import logging
import pandas as pd
from sqlalchemy import create_engine
from src.config import Config
import os
import requests
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_engine():
    """Crée la connexion à la base de données météo"""
    # Ces variables devraient idéalement être dans des variables d'environnement
    user = os.getenv("POSTGRES_USER", "user")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    host = os.getenv("POSTGRES_HOST", "weather-db") # Nom du service docker
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "weather_data")
    
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)

def fetch_recent_weather_data(start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    """
    Récupère les données météo depuis une date de début jusqu'à maintenant
    """
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=7)
    
    logger.info(f"🌐 Récupération des données météo de {start_date.date()} à {end_date.date()}...")
    
    try:
        # Paramètres de l'API
        params = Config.WEATHER_API_PARAMS.copy()
        params.update({
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        })
        
        # Requête API
        response = requests.get(Config.WEATHER_API_BASE_URL, params=params)
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
        
        # Calcul des agrégats quotidiens
        df['date'] = df['datetime'].dt.date
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
        
        # Renommer date en time pour compatibilité
        daily_df = daily_df.rename(columns={'date': 'time'})
        
        logger.info(f"✅ Données récupérées: {len(daily_df)} jours")
        return daily_df
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des données récentes: {e}")
        raise

def ingest_weather_data(source_path: str = "marrakech_weather_2018_2023_final.csv", fetch_recent: bool = True):
    """
    Charge les données depuis un CSV et les insère dans la base de données
    Optionnellement récupère et ajoute les données récentes
    """
    logger.info("🚀 Démarrage de l'ingestion des données...")
    
    try:
        # 1. Lecture des données historiques
        logger.info(f"📥 Lecture des données depuis {source_path}")
        df = pd.read_csv(source_path)
        
        # Standardisation des colonnes
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'])
        
        # 2. Récupération des données récentes si demandé
        if fetch_recent:
            try:
                # Calculer la date de début comme le lendemain de la dernière date dans les données
                if 'datetime' in df.columns and not df.empty:
                    last_date = pd.to_datetime(df['datetime']).max()
                    start_date = last_date + timedelta(days=1)
                else:
                    start_date = datetime.now() - timedelta(days=7)  # Fallback
                
                end_date = datetime.now()
                if start_date >= end_date:
                    logger.info("📅 Aucune nouvelle donnée à récupérer (données déjà à jour)")
                else:
                    recent_df = fetch_recent_weather_data(start_date=start_date, end_date=end_date)
                    # Les colonnes sont déjà correctement nommées
                    # Ajouter les colonnes manquantes avec des valeurs par défaut
                    missing_columns = set(df.columns) - set(recent_df.columns)
                    for col in missing_columns:
                        if col not in ['datetime']:  # datetime sera ajouté après
                            recent_df[col] = np.nan  # Valeur manquante appropriée
                    
                    # Fusionner avec les données historiques
                    df = pd.concat([df, recent_df], ignore_index=True)
                    # Supprimer les doublons basés sur la date
                    df = df.drop_duplicates(subset=['datetime'], keep='last')
                    logger.info(f"📅 Données fusionnées: {len(df)} lignes totales")
                    
                    # Sauvegarder le CSV mis à jour
                    df.to_csv(source_path, index=False)
                    logger.info(f"💾 CSV mis à jour: {source_path}")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de récupérer les données récentes: {e}")
        
        # 3. Connexion à la DB
        engine = get_db_engine()
        
        # 4. Insertion (append pour ajouter aux données existantes)
        logger.info("💾 Sauvegarde dans la base de données PostgreSQL...")
        df.to_sql('weather_measurements', engine, if_exists='append', index=False)
        
        logger.info(f"✅ Ingestion terminée avec succès : {len(df)} lignes insérées.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'ingestion : {e}")
        raise

if __name__ == "__main__":
    # Mode de vérification : récupérer et afficher les données récentes
    print("🔍 MODE VÉRIFICATION - Récupération des données récentes...")
    
    try:
        recent_data = fetch_recent_weather_data(start_date=datetime.now() - timedelta(days=7))
        print(f"✅ {len(recent_data)} jours de données récupérés")
        print("\n📊 Aperçu des données récentes :")
        print(recent_data.head())
        print("\n📈 Statistiques :")
        print(recent_data.describe())
        
        # Sauvegarder pour inspection
        recent_data.to_csv("recent_weather_verification.csv", index=False)
        print("💾 Données sauvegardées dans 'recent_weather_verification.csv'")
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")
    
    # Mode normal d'ingestion
    print("\n🚀 Démarrage de l'ingestion normale...")
    ingest_weather_data()
