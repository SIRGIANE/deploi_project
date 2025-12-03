"""
Pipeline de données pour le projet Climate MLOps
Automatise le chargement, nettoyage et préparation des données climatiques
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClimateDataPipeline:
    """Pipeline de traitement des données climatiques"""
    
    def __init__(self, data_path: str = "/root/.cache/kagglehub/datasets/berkeleyearth/climate-change-earth-surface-temperature-data/versions/2"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_data(self) -> pd.DataFrame:
        """Chargement des données climatiques depuis Kaggle"""
        logger.info("🔄 Chargement des données climatiques...")
        
        try:
            global_temp = pd.read_csv(f"{self.data_path}/GlobalTemperatures.csv")
            logger.info(f"✅ Données chargées: {global_temp.shape}")
            return global_temp
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validation de la qualité des données"""
        logger.info("🔍 Validation des données...")
        
        required_columns = ['dt', 'LandAverageTemperature']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ Colonnes manquantes: {missing_cols}")
            return False
            
        # Vérification du pourcentage de valeurs manquantes
        missing_pct = df['LandAverageTemperature'].isna().mean()
        if missing_pct > 0.5:
            logger.warning(f"⚠️ Taux élevé de valeurs manquantes: {missing_pct:.2%}")
            
        # Vérification de la plage de dates
        df['dt'] = pd.to_datetime(df['dt'])
        date_range = df['dt'].max() - df['dt'].min()
        logger.info(f"📅 Plage de dates: {date_range.days} jours")
        
        logger.info("✅ Validation des données terminée")
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage et préparation des données"""
        logger.info("🧹 Nettoyage des données...")
        
        df_clean = df.copy()
        
        # Conversion des dates
        df_clean['dt'] = pd.to_datetime(df_clean['dt'])
        
        # Extraction des features temporelles
        df_clean['Year'] = df_clean['dt'].dt.year
        df_clean['Month'] = df_clean['dt'].dt.month
        df_clean['Day'] = df_clean['dt'].dt.day
        df_clean['Quarter'] = df_clean['dt'].dt.quarter
        df_clean['DayOfYear'] = df_clean['dt'].dt.dayofyear
        df_clean['WeekOfYear'] = df_clean['dt'].dt.isocalendar().week
        
        # Suppression des valeurs manquantes
        initial_size = len(df_clean)
        df_clean = df_clean.dropna(subset=['LandAverageTemperature'])
        final_size = len(df_clean)
        
        logger.info(f"🗑️ Suppression de {initial_size - final_size} lignes avec valeurs manquantes")
        
        # Tri par date
        df_clean = df_clean.sort_values('dt').reset_index(drop=True)
        
        logger.info(f"✅ Données nettoyées: {df_clean.shape}")
        return df_clean
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Création des features avancées pour ML"""
        logger.info("🔧 Création des features...")
        
        df_features = df.copy()
        
        # Features cycliques (saisonnalité)
        df_features['Month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
        df_features['Month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
        df_features['DayOfYear_sin'] = np.sin(2 * np.pi * df_features['DayOfYear'] / 365)
        df_features['DayOfYear_cos'] = np.cos(2 * np.pi * df_features['DayOfYear'] / 365)
        
        # Features de lag (valeurs passées)
        for lag in [1, 3, 6, 12]:
            df_features[f'Temp_lag_{lag}'] = df_features['LandAverageTemperature'].shift(lag)
        
        # Moyennes mobiles
        for window in [3, 6, 12]:
            df_features[f'Temp_ma_{window}'] = df_features['LandAverageTemperature'].rolling(window=window).mean()
        
        # Tendance (pente sur les derniers 12 mois)
        df_features['Temp_trend_12m'] = df_features['LandAverageTemperature'].rolling(window=12).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 12 else np.nan
        )
        
        # Volatilité (écart-type sur les derniers 6 mois)
        df_features['Temp_volatility_6m'] = df_features['LandAverageTemperature'].rolling(window=6).std()
        
        # Différences
        df_features['Temp_diff_1m'] = df_features['LandAverageTemperature'].diff(1)
        df_features['Temp_diff_12m'] = df_features['LandAverageTemperature'].diff(12)
        
        # Suppression des lignes avec des NaN dus aux lags et moyennes mobiles
        df_features = df_features.dropna()
        
        feature_cols = len(df_features.columns) - len(df.columns)
        logger.info(f"✅ {feature_cols} nouvelles features créées")
        
        return df_features
    
    def prepare_ml_data(self, df: pd.DataFrame, split_date: str = '2010-01-01') -> Dict[str, Any]:
        """Préparation des données pour l'entraînement ML"""
        logger.info("📊 Préparation des données ML...")
        
        feature_columns = [
            'Year', 'Month', 'Quarter', 'DayOfYear', 'WeekOfYear',
            'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos',
            'Temp_lag_1', 'Temp_lag_3', 'Temp_lag_6', 'Temp_lag_12',
            'Temp_ma_3', 'Temp_ma_6', 'Temp_ma_12',
            'Temp_trend_12m', 'Temp_volatility_6m',
            'Temp_diff_1m', 'Temp_diff_12m'
        ]
        
        target_column = 'LandAverageTemperature'
        
        # Division temporelle
        train_data = df[df['dt'] < split_date].copy()
        test_data = df[df['dt'] >= split_date].copy()
        
        # Matrices X et y
        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        # Normalisation
        if not self.is_fitted:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.is_fitted = True
        else:
            X_train_scaled = self.scaler.transform(X_train)
            
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"🚂 Données d'entraînement: {X_train_scaled.shape}")
        logger.info(f"🧪 Données de test: {X_test_scaled.shape}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'feature_names': feature_columns,
            'train_dates': train_data['dt'].values,
            'test_dates': test_data['dt'].values
        }
    
    def save_pipeline(self, filepath: str = 'models/data_pipeline.joblib'):
        """Sauvegarde du pipeline (scaler notamment)"""
        logger.info(f"💾 Sauvegarde du pipeline: {filepath}")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        pipeline_data = {
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'data_path': self.data_path
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info("✅ Pipeline sauvegardé")
    
    def load_pipeline(self, filepath: str = 'models/data_pipeline.joblib'):
        """Chargement du pipeline"""
        logger.info(f"📂 Chargement du pipeline: {filepath}")
        
        pipeline_data = joblib.load(filepath)
        self.scaler = pipeline_data['scaler']
        self.is_fitted = pipeline_data['is_fitted']
        self.data_path = pipeline_data['data_path']
        
        logger.info("✅ Pipeline chargé")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Exécution complète du pipeline"""
        logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET")
        
        # Étapes du pipeline
        raw_data = self.load_data()
        
        if not self.validate_data(raw_data):
            raise ValueError("Validation des données échouée")
        
        clean_data = self.clean_data(raw_data)
        featured_data = self.create_features(clean_data)
        ml_data = self.prepare_ml_data(featured_data)
        
        # Sauvegarde
        self.save_pipeline()
        
        logger.info("🎉 PIPELINE TERMINÉ AVEC SUCCÈS")
        
        return {
            'raw_shape': raw_data.shape,
            'clean_shape': clean_data.shape,
            'featured_shape': featured_data.shape,
            'ml_data': ml_data,
            'pipeline_saved': True
        }

def main():
    """Fonction principale pour tester le pipeline"""
    pipeline = ClimateDataPipeline()
    
    try:
        results = pipeline.run_full_pipeline()
        
        print("📊 RÉSUMÉ DU PIPELINE:")
        print(f"   Données brutes: {results['raw_shape']}")
        print(f"   Données nettoyées: {results['clean_shape']}")
        print(f"   Avec features: {results['featured_shape']}")
        print(f"   Train/Test préparés: ✅")
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {e}")
        raise

if __name__ == "__main__":
    main()