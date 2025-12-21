"""
Module pour charger et traiter les données météo de Marrakech
Remplace le module kaggle_weather_data.py pour utiliser le dataset local
NOUVEAU: Collecte quotidienne des données météo via API + stockage PostgreSQL
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarrakechWeatherDataLoader:
    """Chargeur de données météo de Marrakech depuis la base de données et API"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialise le chargeur de données
        
        Args:
            connection_string: URL de connexion à la DB (optionnel)
        """
        import os
        
        if connection_string:
            self.connection_string = connection_string
            self._db_info = "custom connection"
        else:
            # Valeurs par défaut correspondant au docker-compose
            user = os.getenv("POSTGRES_USER", "user")
            password = os.getenv("POSTGRES_PASSWORD", "password")
            
            # CORRECTION: Détection automatique améliorée
            # Dans Airflow Docker: utiliser weather-db ou airflow-postgres
            # Hors Docker (macOS local): utiliser localhost
            if os.path.exists("/.dockerenv"):
                # Dans Docker
                default_host = "weather-db"
                default_port = "5432"
            elif os.getenv("AIRFLOW_HOME"):
                # Airflow local mais pas Docker
                default_host = os.getenv("POSTGRES_HOST", "localhost")
                default_port = os.getenv("POSTGRES_PORT", "5433")
            else:
                # Mode local standard
                default_host = "localhost"
                default_port = "5433"
                
            host = os.getenv("POSTGRES_HOST", default_host)
            port = os.getenv("POSTGRES_PORT", default_port)
            db = os.getenv("POSTGRES_DB", "weather_data")
            
            self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"
            self._db_info = f"{host}:{port}/{db}"
            
        # Configuration API météo (Open-Meteo)
        self.api_base_url = "https://api.open-meteo.com/v1/forecast"
        self.historical_api_url = "https://archive-api.open-meteo.com/v1/archive"
        self.marrakech_lat = 31.6295
        self.marrakech_lon = -7.9811
        
        # Fichier de données cumulatives local
        self.cumulative_data_file = "data/cumulative_weather_data.csv"
        self.historical_data_file = "marrakech_weather_2018_2023_final.csv"
        Path("data").mkdir(exist_ok=True)
        
        # Table PostgreSQL
        self.table_name = "weather_data"
            
        logger.info(f"📂 Chargeur DB + API initialisé")
        logger.info(f"   🔗 PostgreSQL: {self._db_info}")
    
    def _get_db_engine(self):
        """Crée une connexion à la base de données PostgreSQL"""
        try:
            from sqlalchemy import create_engine
            engine = create_engine(self.connection_string)
            return engine
        except Exception as e:
            logger.warning(f"⚠️ Impossible de créer le moteur DB: {e}")
            return None
    
    def init_database(self) -> bool:
        """
        Initialise la table PostgreSQL si elle n'existe pas
        
        Returns:
            bool: True si succès
        """
        try:
            engine = self._get_db_engine()
            if engine is None:
                return False
            
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS weather_data (
                id SERIAL PRIMARY KEY,
                time DATE UNIQUE NOT NULL,
                temperature_2m_max FLOAT,
                temperature_2m_min FLOAT,
                temperature_2m_mean FLOAT,
                apparent_temperature_max FLOAT,
                apparent_temperature_min FLOAT,
                precipitation_sum FLOAT,
                rain_sum FLOAT,
                snowfall_sum FLOAT,
                precipitation_hours FLOAT,
                windspeed_10m_max FLOAT,
                windgusts_10m_max FLOAT,
                winddirection_10m_dominant FLOAT,
                shortwave_radiation_sum FLOAT,
                et0_fao_evapotranspiration FLOAT,
                weathercode FLOAT,
                humidity FLOAT,
                year FLOAT,
                month FLOAT,
                day FLOAT,
                day_of_year FLOAT,
                season FLOAT,
                datetime DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_weather_time ON weather_data(time);
            CREATE INDEX IF NOT EXISTS idx_weather_datetime ON weather_data(datetime);
            """
            
            # Utiliser engine.begin() pour auto-commit (compatible SQLAlchemy 2.x)
            with engine.begin() as conn:
                from sqlalchemy import text
                conn.execute(text(create_table_sql))
            
            logger.info("✅ Table PostgreSQL 'weather_data' initialisée")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de la DB: {e}")
            return False
    
    def load_historical_data_to_db(self) -> bool:
        """
        Charge les données historiques dans PostgreSQL si la table est vide
        
        Returns:
            bool: True si succès
        """
        try:
            engine = self._get_db_engine()
            if engine is None:
                logger.warning("⚠️ Pas de connexion DB, skip du chargement historique")
                return False
            
            # Vérifier si la table contient déjà des données
            with engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}"))
                count = result.scalar()
            
            if count > 0:
                logger.info(f"ℹ️ La table contient déjà {count} lignes, skip du chargement historique")
                return True
            
            # Charger les données historiques depuis le fichier CSV
            if not Path(self.historical_data_file).exists():
                logger.warning(f"⚠️ Fichier historique non trouvé: {self.historical_data_file}")
                return False
            
            logger.info(f"📥 Chargement des données historiques depuis {self.historical_data_file}...")
            df = pd.read_csv(self.historical_data_file)
            
            # Préparer les colonnes
            if 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'])
                df['time'] = pd.to_datetime(df['time']).dt.date
            
            # Insérer dans PostgreSQL
            df.to_sql(self.table_name, engine, if_exists='append', index=False)
            
            logger.info(f"✅ {len(df)} lignes historiques chargées dans PostgreSQL")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement historique: {e}")
            return False
    
    def store_to_postgres(self, df: pd.DataFrame) -> bool:
        """
        Stocke les données dans PostgreSQL (avec upsert pour éviter les doublons)
        
        Args:
            df: DataFrame à stocker
            
        Returns:
            bool: True si succès
        """
        try:
            engine = self._get_db_engine()
            if engine is None:
                logger.warning("⚠️ Pas de connexion DB, stockage uniquement en CSV")
                return False
            
            # S'assurer que la table existe
            self.init_database()
            
            # Préparer les données
            df_to_store = df.copy()
            if 'time' in df_to_store.columns:
                df_to_store['time'] = pd.to_datetime(df_to_store['time']).dt.date
            if 'datetime' in df_to_store.columns:
                df_to_store['datetime'] = pd.to_datetime(df_to_store['datetime']).dt.date
            
            # Upsert: insérer ou mettre à jour
            from sqlalchemy import text
            
            inserted_count = 0
            with engine.connect() as conn:
                for _, row in df_to_store.iterrows():
                    # Vérifier si la date existe déjà
                    date_val = row.get('time', row.get('datetime'))
                    check_sql = text(f"SELECT id FROM {self.table_name} WHERE time = :date_val")
                    result = conn.execute(check_sql, {"date_val": date_val})
                    existing = result.fetchone()
                    
                    if existing is None:
                        # Insérer la nouvelle ligne
                        columns = [col for col in df_to_store.columns if col in [
                            'time', 'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                            'apparent_temperature_max', 'apparent_temperature_min', 'precipitation_sum',
                            'rain_sum', 'snowfall_sum', 'precipitation_hours', 'windspeed_10m_max',
                            'windgusts_10m_max', 'winddirection_10m_dominant', 'shortwave_radiation_sum',
                            'et0_fao_evapotranspiration', 'weathercode', 'humidity', 'year', 'month',
                            'day', 'day_of_year', 'season', 'datetime'
                        ]]
                        
                        cols_str = ', '.join(columns)
                        vals_str = ', '.join([f':{col}' for col in columns])
                        insert_sql = text(f"INSERT INTO {self.table_name} ({cols_str}) VALUES ({vals_str})")
                        
                        values = {col: row[col] for col in columns if col in row.index}
                        conn.execute(insert_sql, values)
                        inserted_count += 1
                
                conn.commit()
            
            logger.info(f"✅ {inserted_count} nouvelles lignes stockées dans PostgreSQL")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du stockage PostgreSQL: {e}")
            return False
    
    def get_data_from_postgres(self) -> Optional[pd.DataFrame]:
        """
        Récupère toutes les données depuis PostgreSQL
        
        Returns:
            pd.DataFrame ou None si erreur
        """
        try:
            engine = self._get_db_engine()
            if engine is None:
                return None
            
            query = f"SELECT * FROM {self.table_name} ORDER BY time"
            df = pd.read_sql(query, engine)
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                logger.info(f"📊 {len(df)} lignes récupérées depuis PostgreSQL")
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de la lecture PostgreSQL: {e}")
            return None
    
    def fetch_today_weather_data(self) -> pd.DataFrame:
        """
        Récupère les données météo d'aujourd'hui via l'API Open-Meteo
        
        Returns:
            pd.DataFrame: Données météo d'aujourd'hui (pas de valeurs manquantes)
        """
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        # Récupérer les 7 derniers jours pour avoir des données complètes
        start_date = today - timedelta(days=7)
        
        logger.info(f"🌦️  Récupération des données météo du {start_date} au {today}")
        
        try:
            # === DONNÉES HISTORIQUES (jusqu'à hier) ===
            historical_params = {
                "latitude": self.marrakech_lat,
                "longitude": self.marrakech_lon,
                "start_date": str(start_date),
                "end_date": str(yesterday),
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "temperature_2m_mean",
                    "apparent_temperature_max",
                    "apparent_temperature_min",
                    "precipitation_sum",
                    "rain_sum",
                    "snowfall_sum",
                    "precipitation_hours",
                    "windspeed_10m_max",
                    "windgusts_10m_max",
                    "winddirection_10m_dominant",
                    "shortwave_radiation_sum",
                    "et0_fao_evapotranspiration",
                    "weathercode"
                ],
                "timezone": "Africa/Casablanca"
            }
            
            historical_response = requests.get(self.historical_api_url, params=historical_params, timeout=15)
            historical_response.raise_for_status()
            historical_data = historical_response.json()
            
            historical_df = pd.DataFrame()
            if 'daily' in historical_data and historical_data['daily']:
                historical_df = pd.DataFrame(historical_data['daily'])
                historical_df['time'] = pd.to_datetime(historical_df['time'])
                logger.info(f"✅ Données historiques récupérées: {len(historical_df)} jours")
            
            # === DONNÉES DE PRÉVISION (aujourd'hui) ===
            forecast_params = {
                "latitude": self.marrakech_lat,
                "longitude": self.marrakech_lon,
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "temperature_2m_mean",
                    "apparent_temperature_max",
                    "apparent_temperature_min",
                    "precipitation_sum",
                    "rain_sum",
                    "snowfall_sum",
                    "precipitation_hours",
                    "windspeed_10m_max",
                    "windgusts_10m_max",
                    "winddirection_10m_dominant",
                    "shortwave_radiation_sum",
                    "et0_fao_evapotranspiration",
                    "weathercode"
                ],
                "timezone": "Africa/Casablanca",
                "forecast_days": 1  # Seulement aujourd'hui
            }
            
            forecast_response = requests.get(self.api_base_url, params=forecast_params, timeout=15)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            forecast_df = pd.DataFrame()
            if 'daily' in forecast_data and forecast_data['daily']:
                forecast_df = pd.DataFrame(forecast_data['daily'])
                forecast_df['time'] = pd.to_datetime(forecast_df['time'])
                logger.info(f"✅ Données de prévision récupérées: {len(forecast_df)} jours")
            
            # === COMBINAISON DES DONNÉES ===
            if historical_df.empty and forecast_df.empty:
                raise ValueError("Aucune donnée récupérée des APIs")
            
            # Combiner historique et prévision
            if not historical_df.empty and not forecast_df.empty:
                combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
            elif not historical_df.empty:
                combined_df = historical_df
            else:
                combined_df = forecast_df
            
            # Supprimer les doublons par date
            combined_df = combined_df.drop_duplicates(subset='time', keep='last')
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            
            df = combined_df.copy()
            
            if df.empty:
                raise ValueError("DataFrame vide après combinaison")
            
            df['time'] = pd.to_datetime(df['time'])
            df['datetime'] = df['time'].dt.strftime('%Y-%m-%d')
            
            # Ajouter les colonnes dérivées
            df['year'] = df['time'].dt.year.astype(float)
            df['month'] = df['time'].dt.month.astype(float)
            df['day'] = df['time'].dt.day.astype(float)
            df['day_of_year'] = df['time'].dt.dayofyear.astype(float)
            
            # Saison (1=hiver, 2=printemps, 3=été, 4=automne)
            def get_season(month):
                if month in [12, 1, 2]:
                    return 1.0
                elif month in [3, 4, 5]:
                    return 2.0
                elif month in [6, 7, 8]:
                    return 3.0
                else:
                    return 4.0
            
            df['season'] = df['time'].dt.month.apply(get_season)
            
            # Colonnes numériques à vérifier
            numeric_cols = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                           'apparent_temperature_max', 'apparent_temperature_min',
                           'precipitation_sum', 'rain_sum', 'snowfall_sum', 'precipitation_hours',
                           'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant',
                           'shortwave_radiation_sum', 'et0_fao_evapotranspiration', 'weathercode']
            
            existing_numeric = [col for col in numeric_cols if col in df.columns]
            
            # Pour les valeurs manquantes, interpoler
            for col in existing_numeric:
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].interpolate(method='linear')
                    df[col] = df[col].ffill().bfill()
            
            # Convertir time en string pour le CSV
            df['time'] = df['time'].dt.strftime('%Y-%m-%d')
            
            logger.info(f"✅ Données météo récupérées: {len(df)} jours")
            logger.info(f"   📅 Période: {df['time'].iloc[0]} à {df['time'].iloc[-1]}")
            if 'temperature_2m_mean' in df.columns:
                logger.info(f"   🌡️ Temp aujourd'hui: {df['temperature_2m_mean'].iloc[-1]:.1f}°C")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des données météo: {e}")
            raise
    
    def store_daily_data(self, new_data: pd.DataFrame) -> bool:
        """
        Stocke les nouvelles données quotidiennes dans le fichier CSV ET PostgreSQL
        
        Args:
            new_data: DataFrame avec les nouvelles données
            
        Returns:
            bool: True si stockage réussi
        """
        try:
            # === STOCKAGE CSV ===
            if Path(self.cumulative_data_file).exists():
                existing_data = pd.read_csv(self.cumulative_data_file)
                existing_data['datetime'] = pd.to_datetime(existing_data['datetime'])
                logger.info(f"📂 Données CSV existantes: {len(existing_data)} lignes")
            else:
                # Si pas de fichier cumulatif, charger l'historique
                if Path(self.historical_data_file).exists():
                    existing_data = pd.read_csv(self.historical_data_file)
                    if 'time' in existing_data.columns:
                        existing_data['datetime'] = pd.to_datetime(existing_data['time'])
                    logger.info(f"📂 Chargement initial depuis l'historique: {len(existing_data)} lignes")
                else:
                    existing_data = pd.DataFrame()
                    logger.info("📂 Création du fichier de données cumulatives")
            
            # Convertir datetime
            new_data['datetime'] = pd.to_datetime(new_data['datetime'])
            
            if not existing_data.empty:
                existing_dates = set(existing_data['datetime'].dt.date)
                new_dates = new_data['datetime'].dt.date
                new_data_filtered = new_data[~new_data['datetime'].dt.date.isin(existing_dates)]
            else:
                new_data_filtered = new_data
            
            if new_data_filtered.empty:
                logger.info("ℹ️  Aucune nouvelle donnée à ajouter (dates déjà existantes)")
            else:
                # Combiner les données
                if existing_data.empty:
                    combined_data = new_data_filtered
                else:
                    combined_data = pd.concat([existing_data, new_data_filtered], ignore_index=True)
                
                # Trier par date
                combined_data = combined_data.sort_values('datetime').reset_index(drop=True)
                
                # Sauvegarder CSV
                combined_data.to_csv(self.cumulative_data_file, index=False)
                
                logger.info(f"✅ CSV: +{len(new_data_filtered)} nouvelles lignes")
                logger.info(f"   📊 Total CSV: {len(combined_data)} lignes")
            
            # === STOCKAGE POSTGRESQL ===
            postgres_success = self.store_to_postgres(new_data)
            if postgres_success:
                logger.info("✅ PostgreSQL: Données synchronisées")
            else:
                logger.warning("⚠️ PostgreSQL: Stockage échoué (CSV uniquement)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du stockage: {e}")
            return False
    
    def sync_csv_to_postgres(self) -> bool:
        """
        Synchronise le fichier CSV vers PostgreSQL
        
        Returns:
            bool: True si succès
        """
        try:
            if not Path(self.cumulative_data_file).exists():
                logger.warning("⚠️ Fichier CSV non trouvé pour synchronisation")
                return False
            
            df = pd.read_csv(self.cumulative_data_file)
            logger.info(f"📥 Synchronisation de {len(df)} lignes vers PostgreSQL...")
            
            success = self.store_to_postgres(df)
            
            if success:
                logger.info("✅ Synchronisation CSV → PostgreSQL terminée")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la synchronisation: {e}")
            return False
    
    def get_cumulative_data(self) -> pd.DataFrame:
        """
        Récupère toutes les données cumulatives (historiques + récentes)
        Priorité: PostgreSQL > CSV > Fichier historique
        
        Returns:
            pd.DataFrame: Données complètes sans valeurs manquantes
        """
        try:
            # 1. Essayer PostgreSQL d'abord
            df = self.get_data_from_postgres()
            if df is not None and not df.empty:
                logger.info(f"📊 Données depuis PostgreSQL: {len(df)} lignes")
                return df
            
            # 2. Sinon, données cumulatives CSV
            if Path(self.cumulative_data_file).exists():
                df = pd.read_csv(self.cumulative_data_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                logger.info(f"📊 Données depuis CSV cumulatif: {len(df)} lignes")
                return df
            
            # 3. Fallback sur le fichier historique original
            if Path(self.historical_data_file).exists():
                df = pd.read_csv(self.historical_data_file)
                if 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'])
                logger.info(f"📊 Données depuis fichier historique: {len(df)} lignes")
                return df
            
            raise FileNotFoundError("Aucun fichier de données trouvé")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des données cumulatives: {e}")
            raise
    
    def collect_and_store_today_data(self) -> Dict[str, Any]:
        """
        Collecte les données d'aujourd'hui et les stocke (CSV + PostgreSQL)
        
        Returns:
            dict: Résumé de l'opération
        """
        logger.info("🚀 COLLECTE QUOTIDIENNE DES DONNÉES MÉTÉO")
        logger.info("=" * 50)
        
        try:
            # Initialiser PostgreSQL si nécessaire
            self.init_database()
            
            # Charger les données historiques dans PostgreSQL si vide
            self.load_historical_data_to_db()
            
            # Récupérer les données récentes via l'API
            today_data = self.fetch_today_weather_data()
            
            if today_data.empty:
                raise ValueError("Aucune donnée récupérée de l'API")
            
            # Les stocker (CSV + PostgreSQL)
            success = self.store_daily_data(today_data)
            
            if not success:
                raise ValueError("Échec du stockage des données")
            
            # Statistiques
            cumulative_data = self.get_cumulative_data()
            
            # Données du jour le plus récent
            latest_record = today_data.iloc[-1]
            today_weather = {
                'date': str(latest_record.get('datetime', latest_record.get('time', ''))),
                'temperature_mean': float(latest_record.get('temperature_2m_mean', 20.0)),
                'temperature_max': float(latest_record.get('temperature_2m_max', 25.0)),
                'temperature_min': float(latest_record.get('temperature_2m_min', 15.0)),
                'precipitation': float(latest_record.get('precipitation_sum', 0.0)),
                'windspeed': float(latest_record.get('windspeed_10m_max', 15.0))
            }
            
            result = {
                'success': success,
                'new_records': len(today_data),
                'total_records': len(cumulative_data),
                'date_range': {
                    'start': str(cumulative_data['datetime'].min())[:10],
                    'end': str(cumulative_data['datetime'].max())[:10]
                },
                'today_weather': today_weather,
                'collection_time': datetime.now().isoformat(),
                'data_quality': 'complete',
                'storage': {
                    'csv': True,
                    'postgresql': True
                }
            }
            
            logger.info("✅ COLLECTE TERMINÉE")
            logger.info(f"   📅 Nouvelles données: {result['new_records']} lignes")
            logger.info(f"   📊 Total cumulé: {result['total_records']} lignes")
            logger.info(f"   🌡️ Temp aujourd'hui: {today_weather['temperature_mean']:.1f}°C")
            logger.info(f"   💾 Stockage: CSV ✓ | PostgreSQL ✓")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la collecte quotidienne: {e}")
            raise
    
    def get_days_since_last_training(self) -> int:
        """
        Calcule le nombre de jours depuis le dernier entraînement
        
        Returns:
            int: Nombre de jours
        """
        try:
            # Vérifier le fichier de résultats d'entraînement
            results_file = "results/weather_training_results.json"
            if Path(results_file).exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                last_training = results.get('training_completed')
                if last_training:
                    # Gérer différents formats de date
                    try:
                        last_dt = datetime.fromisoformat(last_training.replace('Z', '+00:00'))
                    except:
                        last_dt = pd.to_datetime(last_training)
                    
                    days_since = (datetime.now() - last_dt).days
                    logger.info(f"📅 Dernier entraînement: {last_dt.date()} ({days_since} jours)")
                    return days_since
            
            logger.info("📅 Aucun entraînement précédent trouvé")
            return 999  # Force le retraining si aucun historique
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors du calcul des jours: {e}")
            return 999  # En cas d'erreur, forcer le retraining
    
    def should_trigger_retraining(self, threshold_days: int = 7) -> bool:
        """
        Détermine si un retraining doit être déclenché
        
        Args:
            threshold_days: Seuil en jours pour déclencher le retraining
            
        Returns:
            bool: True si retraining nécessaire
        """
        days_since = self.get_days_since_last_training()
        should_retrain = days_since >= threshold_days
        
        logger.info(f"🤖 Retraining nécessaire: {'OUI' if should_retrain else 'NON'}")
        logger.info(f"   📅 Jours écoulés: {days_since}/{threshold_days}")
        
        return should_retrain

    def update_last_training_date(self) -> bool:
        """
        Met à jour la date du dernier entraînement dans le fichier de résultats
        
        Returns:
            bool: True si succès
        """
        try:
            results_file = "results/weather_training_results.json"
            Path("results").mkdir(exist_ok=True)
            
            # Charger les résultats existants ou créer un nouveau dict
            if Path(results_file).exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
            else:
                results = {}
            
            # Mettre à jour la date
            results['training_completed'] = datetime.now().isoformat()
            results['last_training_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Sauvegarder
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Date de training mise à jour: {results['last_training_date']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la mise à jour de la date: {e}")
            return False

    def load_weather_data(self) -> pd.DataFrame:
        """
        Charge les données météo (version mise à jour pour utiliser les données cumulatives)
        
        Returns:
            pd.DataFrame: Données météo brutes
        """
        logger.info("=" * 70)
        logger.info("📥 CHARGEMENT DES DONNÉES MÉTÉO COMPLÈTES")
        logger.info("=" * 70)
        
        try:
            # Charger les données cumulatives
            df = self.get_cumulative_data()
            
            logger.info(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Conversion datetime si nécessaire
            if 'datetime' not in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'])
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise

    def preprocess_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Préprocessing des données météo
        
        Args:
            df: DataFrame brut
            
        Returns:
            pd.DataFrame: Données préprocessées
        """
        logger.info("=" * 70)
        logger.info("🔧 PREPROCESSING DES DONNÉES")
        logger.info("=" * 70)
        
        df_clean = df.copy()
        
        # 1. Gestion de la colonne datetime
        if 'time' in df_clean.columns and 'datetime' not in df_clean.columns:
            df_clean['datetime'] = pd.to_datetime(df_clean['time'])
            logger.info("✅ Colonne datetime créée")
        
        # 2. Tri par date
        if 'datetime' in df_clean.columns:
            df_clean = df_clean.sort_values('datetime').reset_index(drop=True)
            logger.info("✅ Données triées par date")
        
        # 3. Gestion des valeurs manquantes
        initial_missing = df_clean.isnull().sum().sum()
        
        # Colonnes numériques
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Interpolation linéaire pour les séries temporelles
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        
        final_missing = df_clean.isnull().sum().sum()
        logger.info(f"✅ Valeurs manquantes: {initial_missing} → {final_missing}")
        
        # 4. Détection et traitement des outliers (méthode IQR)
        outliers_count = 0
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outliers_count += outliers_mask.sum()
            
            # Remplacement par la médiane
            if outliers_mask.sum() > 0:
                median_val = df_clean[col].median()
                df_clean.loc[outliers_mask, col] = median_val
        
        logger.info(f"✅ Outliers traités: {outliers_count} valeurs")
        
        # 5. Suppression des doublons
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"✅ Doublons supprimés: {duplicates}")
        
        logger.info(f"📊 Forme finale: {df_clean.shape}")
        
        return df_clean
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features avancées pour le ML
        
        Args:
            df: DataFrame préprocessé
            
        Returns:
            pd.DataFrame: Données avec features enrichies
        """
        logger.info("=" * 70)
        logger.info("🎯 CRÉATION DES FEATURES")
        logger.info("=" * 70)
        
        df_features = df.copy()
        
        # 1. Features temporelles
        if 'datetime' in df_features.columns:
            # Assurer que datetime est bien en format datetime
            df_features['datetime'] = pd.to_datetime(df_features['datetime'])
            
            df_features['Year'] = df_features['datetime'].dt.year
            df_features['Month'] = df_features['datetime'].dt.month
            df_features['Day'] = df_features['datetime'].dt.day
            df_features['DayOfYear'] = df_features['datetime'].dt.dayofyear
            df_features['WeekOfYear'] = df_features['datetime'].dt.isocalendar().week
            df_features['Quarter'] = df_features['datetime'].dt.quarter
            
            # Features cycliques
            df_features['Month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
            df_features['Month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
            df_features['DayOfYear_sin'] = np.sin(2 * np.pi * df_features['DayOfYear'] / 365)
            df_features['DayOfYear_cos'] = np.cos(2 * np.pi * df_features['DayOfYear'] / 365)
            
            logger.info("✅ Features temporelles créées (11 features)")
        
        # 2. Features de température (si disponibles)
        temp_cols = [col for col in df_features.columns if 'temp' in col.lower()]
        
        if temp_cols:
            main_temp_col = temp_cols[0]  # Utiliser la première colonne de température
            logger.info(f"🌡️  Colonne de température principale: {main_temp_col}")
            
            # Lag features
            for lag in [1, 3, 7, 14, 30]:
                df_features[f'Temp_lag_{lag}'] = df_features[main_temp_col].shift(lag)
            logger.info("✅ Lag features créées (5 features)")
            
            # Moving averages
            for window in [3, 7, 14, 30]:
                df_features[f'Temp_ma_{window}'] = df_features[main_temp_col].rolling(
                    window=window, min_periods=1
                ).mean()
            logger.info("✅ Moving averages créées (4 features)")
            
            # Tendance et volatilité
            df_features['Temp_trend_30d'] = df_features[main_temp_col].rolling(
                window=30, min_periods=1
            ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            
            df_features['Temp_volatility_7d'] = df_features[main_temp_col].rolling(
                window=7, min_periods=1
            ).std()
            logger.info("✅ Tendance et volatilité créées (2 features)")
            
            # Différences
            df_features['Temp_diff_1d'] = df_features[main_temp_col].diff()
            df_features['Temp_diff_7d'] = df_features[main_temp_col].diff(7)
            logger.info("✅ Différences créées (2 features)")
        
        # 3. Features de précipitations (si disponibles)
        precip_cols = [col for col in df_features.columns if 'precip' in col.lower() or 'rain' in col.lower()]
        if precip_cols:
            main_precip_col = precip_cols[0]
            df_features['Precip_cumsum_7d'] = df_features[main_precip_col].rolling(window=7).sum()
            df_features['Precip_cumsum_30d'] = df_features[main_precip_col].rolling(window=30).sum()
            logger.info("✅ Features de précipitations créées (2 features)")
        
        # 4. Features de vent (si disponibles)
        wind_cols = [col for col in df_features.columns if 'wind' in col.lower()]
        if wind_cols:
            for wind_col in wind_cols[:2]:  # Limiter à 2 colonnes de vent
                df_features[f'{wind_col}_ma_7'] = df_features[wind_col].rolling(window=7).mean()
            logger.info(f"✅ Features de vent créées ({min(2, len(wind_cols))} features)")
        
        # Suppression des NaN créés par les features
        initial_rows = len(df_features)
        df_features = df_features.dropna()
        removed_rows = initial_rows - len(df_features)
        
        if removed_rows > 0:
            logger.info(f"🧹 Lignes avec NaN supprimées: {removed_rows}")
        
        logger.info(f"📊 Forme finale avec features: {df_features.shape}")
        logger.info(f"✨ Total de colonnes: {len(df_features.columns)}")
        
        return df_features
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Génère un résumé statistique des données
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            dict: Résumé statistique
        """
        summary = {
            'shape': df.shape,
            'date_range': (df['datetime'].min(), df['datetime'].max()) if 'datetime' in df.columns else None,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return summary
