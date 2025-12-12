"""
Module pour charger et traiter les données météo de Marrakech
Remplace le module kaggle_weather_data.py pour utiliser le dataset local
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarrakechWeatherDataLoader:
    """Chargeur de données météo de Marrakech depuis la base de données"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialise le chargeur de données
        
        Args:
            connection_string: URL de connexion à la DB (optionnel)
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            # Valeurs par défaut correspondant au docker-compose
            import os
            user = os.getenv("POSTGRES_USER", "user")
            password = os.getenv("POSTGRES_PASSWORD", "password")
            host = os.getenv("POSTGRES_HOST", "weather-db")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "weather_data")
            self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"
            
        logger.info(f"📂 Chargeur DB initialisé")
    
    def load_weather_data(self) -> pd.DataFrame:
        """
        Charge les données météo depuis la base de données
        
        Returns:
            pd.DataFrame: Données météo brutes
        """
        logger.info("=" * 70)
        logger.info("📥 CHARGEMENT DES DONNÉES DEPUIS LA DB")
        logger.info("=" * 70)
        
        try:
            from sqlalchemy import create_engine
            engine = create_engine(self.connection_string)
            
            query = "SELECT * FROM weather_measurements ORDER BY datetime"
            df = pd.read_sql(query, engine)
            
            logger.info(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Conversion datetime si nécessaire
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement DB, tentative de fallback CSV: {e}")
            # Fallback sur le CSV si la DB n'est pas dispo (pour le dev local)
            try:
                return pd.read_csv("marrakech_weather_2018_2023_final.csv")
            except:
                raise e

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
