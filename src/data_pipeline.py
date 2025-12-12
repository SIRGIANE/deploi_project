"""
Pipeline de données pour le projet Climate MLOps avec Dataset Marrakech Weather
Pipeline en 3 étapes: raw → processed → features
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import du loader Marrakech Weather
try:
    from src.marrakech_data_loader import MarrakechWeatherDataLoader
    from src.config import Config
except ImportError:
    from marrakech_data_loader import MarrakechWeatherDataLoader
    from config import Config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RAW_PATH = "data/raw"
DEFAULT_PROCESSED_PATH = "data/processed"
DEFAULT_FEATURES_PATH = "data/features"
DEFAULT_SPLIT_RATIO = 0.8
MIN_DATA_POINTS = 100  # Réduit pour les données journalières de Marrakech

class WeatherDataPipeline:
    """Pipeline de traitement des données météorologiques en 3 étapes"""
    
    def __init__(self, 
                 raw_path: str = DEFAULT_RAW_PATH,
                 processed_path: str = DEFAULT_PROCESSED_PATH,
                 features_path: str = DEFAULT_FEATURES_PATH,
                 data_file: str = None):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.features_path = Path(features_path)
        
        # Création des dossiers
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        self._feature_columns: List[str] = []
        self._target_columns: List[str] = []
        
        # Utilisation du fichier de données Marrakech
        if data_file is None:
            data_file = Config.get_data_file_path()
        
        self.marrakech_loader = MarrakechWeatherDataLoader(str(data_file))
        
    # ============================================================================
    # ÉTAPE 1: CHARGEMENT DES DONNÉES LOCALES → data/raw/
    # ============================================================================
    
    def step1_download_raw_data(self) -> pd.DataFrame:
        """
        ÉTAPE 1: Charge les données depuis le fichier local et les stocke dans data/raw/
        
        Returns:
            pd.DataFrame: Données brutes chargées
        """
        logger.info("=" * 70)
        logger.info("ÉTAPE 1: CHARGEMENT DES DONNÉES MÉTÉO DE MARRAKECH")
        logger.info("=" * 70)
        
        try:
            # Chargement depuis le fichier local
            weather_df = self.marrakech_loader.load_weather_data()
            
            # Sauvegarde dans data/raw/
            raw_file = self.raw_path / "weather_data_raw.csv"
            weather_df.to_csv(raw_file, index=False)
            logger.info(f"✅ Données brutes sauvegardées: {raw_file}")
            logger.info(f"   📊 Shape: {weather_df.shape}")
            logger.info(f"   📋 Colonnes: {list(weather_df.columns[:10])}...")
            
            return weather_df
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise
    
    # ============================================================================
    # ÉTAPE 2: PREPROCESSING DES DONNÉES → data/processed/
    # ============================================================================
    
    def step2_preprocess_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        ÉTAPE 2: Applique le preprocessing et stocke dans data/processed/
        
        Args:
            df: DataFrame brut (ou charge depuis data/raw/ si None)
            
        Returns:
            pd.DataFrame: Données préprocessées
        """
        logger.info("=" * 70)
        logger.info("ÉTAPE 2: PREPROCESSING DES DONNÉES")
        logger.info("=" * 70)
        
        try:
            # Chargement depuis raw si nécessaire
            if df is None:
                raw_file = self.raw_path / "weather_data_raw.csv"
                if not raw_file.exists():
                    logger.warning("⚠️ Fichier raw non trouvé, chargement...")
                    df = self.step1_download_raw_data()
                else:
                    logger.info(f"📂 Chargement depuis: {raw_file}")
                    df = pd.read_csv(raw_file)
            
            # Préprocessing
            df_processed = self.marrakech_loader.preprocess_weather_data(df)
            
            # Sauvegarde dans data/processed/
            processed_file = self.processed_path / "weather_data_processed.csv"
            df_processed.to_csv(processed_file, index=False)
            logger.info(f"✅ Données préprocessées sauvegardées: {processed_file}")
            logger.info(f"   📊 Shape: {df_processed.shape}")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du preprocessing: {e}")
            raise
    
    # ============================================================================
    # ÉTAPE 3: CRÉATION DES FEATURES → data/features/
    # ============================================================================
    
    def step3_create_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        ÉTAPE 3: Crée les features avancées et stocke dans data/features/
        
        Args:
            df: DataFrame préprocessé (ou charge depuis data/processed/ si None)
            
        Returns:
            pd.DataFrame: Données avec features enrichies
        """
        logger.info("=" * 70)
        logger.info("ÉTAPE 3: CRÉATION DES FEATURES")
        logger.info("=" * 70)
        
        try:
            # Chargement depuis processed si nécessaire
            if df is None:
                processed_file = self.processed_path / "weather_data_processed.csv"
                if not processed_file.exists():
                    logger.warning("⚠️ Fichier processed non trouvé, preprocessing...")
                    df = self.step2_preprocess_data()
                else:
                    logger.info(f"📂 Chargement depuis: {processed_file}")
                    df = pd.read_csv(processed_file)
                    # Reconversion de datetime si nécessaire
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Création des features
            df_features = self.marrakech_loader.create_weather_features(df)
            
            # Sauvegarde dans data/features/
            features_file = self.features_path / "weather_data_features.csv"
            df_features.to_csv(features_file, index=False)
            logger.info(f"✅ Features sauvegardées: {features_file}")
            logger.info(f"   📊 Shape: {df_features.shape}")
            
            return df_features
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création des features: {e}")
            raise
    
    # ============================================================================
    # VALIDATION DES DONNÉES
    # ============================================================================
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validation de la qualité des données"""
        logger.info("🔍 Validation des données...")
        
        errors = []
        
        try:
            # Vérification de la taille minimale
            if len(df) < MIN_DATA_POINTS:
                errors.append(f"Données insuffisantes: {len(df)} < {MIN_DATA_POINTS}")
            
            # Vérification des colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 5:
                errors.append(f"Trop peu de features numériques: {len(numeric_cols)}")
            
            # Vérification du pourcentage de valeurs manquantes
            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if missing_pct > 0.3:
                errors.append(f"Taux élevé de valeurs manquantes: {missing_pct:.2%}")
            
        except Exception as e:
            errors.append(f"Erreur lors de la validation: {str(e)}")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("✅ Validation des données réussie")
        else:
            logger.error(f"❌ Validation échouée: {errors}")
            
        return is_valid, errors
    
    # ============================================================================
    # PRÉPARATION DES DONNÉES POUR ML
    # ============================================================================
    
    def _detect_targets(self, df: pd.DataFrame) -> List[str]:
        """Détecte plusieurs cibles météo (temp, humidité, vent, précipitations, pression)."""
        targets = []
        
        excluded_suffixes = ['_lag', '_ma', '_volatility', '_diff', '_trend', '_encoded', '_rolling', '_sin', '_cos']
        
        for col in df.columns:
            if any(col.endswith(suffix) or suffix in col for suffix in excluded_suffixes):
                continue
                
            col_lower = col.lower()
            
            if 'temp' in col_lower or 'temperature' in col_lower:
                if pd.api.types.is_numeric_dtype(df[col]):
                    targets.append(col)
                    continue
            
            if 'humid' in col_lower or 'humidity' in col_lower:
                if pd.api.types.is_numeric_dtype(df[col]):
                    targets.append(col)
                    continue
            
            if 'wind' in col_lower and ('speed' in col_lower or 'bearing' in col_lower):
                if pd.api.types.is_numeric_dtype(df[col]):
                    targets.append(col)
                    continue
            
            if 'pressure' in col_lower or 'press' in col_lower:
                if pd.api.types.is_numeric_dtype(df[col]):
                    targets.append(col)
                    continue
            
            if ('precip' in col_lower or 'rain' in col_lower) and '_encoded' not in col_lower:
                if pd.api.types.is_numeric_dtype(df[col]):
                    targets.append(col)
        
        return list(dict.fromkeys(targets))

    def prepare_ml_data(self, df: pd.DataFrame = None, 
                       split_ratio: float = DEFAULT_SPLIT_RATIO,
                       target_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Préparation des données pour l'entraînement ML
        Charge depuis data/features/ si df n'est pas fourni
        """
        logger.info("=" * 70)
        logger.info("PRÉPARATION DES DONNÉES POUR ML")
        logger.info("=" * 70)
        
        try:
            # Chargement depuis features si nécessaire
            if df is None:
                features_file = self.features_path / "weather_data_features.csv"
                if not features_file.exists():
                    logger.warning("⚠️ Fichier features non trouvé, création...")
                    df = self.step3_create_features()
                else:
                    logger.info(f"📂 Chargement depuis: {features_file}")
                    df = pd.read_csv(features_file)
            
            # Sélection automatique des cibles météo
            if target_columns is None:
                target_columns = self._detect_targets(df)
            if not target_columns:
                raise ValueError("Aucune variable météo trouvée pour la prédiction")
                
            logger.info(f"🎯 Variables cibles: {target_columns}")
            
            # Sélection des features (exclure les cibles et colonnes non-numériques)
            feature_columns = Config.FEATURE_COLUMNS
            
            # Nettoyage final
            clean_df = df[target_columns + feature_columns].dropna()
            
            if len(clean_df) < MIN_DATA_POINTS:
                raise ValueError(f"Données insuffisantes après nettoyage: {len(clean_df)}")
            
            # Division temporelle (80% train, 20% test)
            split_idx = int(len(clean_df) * split_ratio)
            train_df = clean_df.iloc[:split_idx]
            test_df = clean_df.iloc[split_idx:]
            
            # Matrices X et y
            X_train = train_df[feature_columns]
            y_train = train_df[target_columns]
            X_test = test_df[feature_columns]
            y_test = test_df[target_columns]
            
            # Normalisation
            if not self.is_fitted:
                X_train_scaled = self.scaler.fit_transform(X_train)
                self.is_fitted = True
                logger.info("✅ Scaler ajusté sur les données d'entraînement")
            else:
                X_train_scaled = self.scaler.transform(X_train)
                
            X_test_scaled = self.scaler.transform(X_test)
            
            # Sauvegarde des matrices dans data/features/
            np.save(self.features_path / "X_train.npy", X_train_scaled)
            np.save(self.features_path / "X_test.npy", X_test_scaled)
            np.save(self.features_path / "y_train.npy", y_train.values)
            np.save(self.features_path / "y_test.npy", y_test.values)
            logger.info(f"💾 Matrices ML sauvegardées dans: {self.features_path}")
            
            self._feature_columns = feature_columns
            self._target_columns = target_columns
            
            logger.info(f"🚂 Données d'entraînement: {X_train_scaled.shape}")
            logger.info(f"🧪 Données de test: {X_test_scaled.shape}")
            
            return {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train.values,
                'y_test': y_test.values,
                'feature_names': feature_columns,
                'target_names': target_columns,
                'train_dates': train_df.index.values if 'datetime' in train_df else None,
                'test_dates': test_df.index.values if 'datetime' in test_df else None,
                'scaler': self.scaler
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la préparation ML: {e}")
            raise
    
    # ============================================================================
    # SAUVEGARDE ET CHARGEMENT DU PIPELINE
    # ============================================================================
    
    def save_pipeline(self, filepath: str = 'models/data_pipeline.joblib') -> None:
        """Sauvegarde du pipeline"""
        logger.info(f"💾 Sauvegarde du pipeline: {filepath}")
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            pipeline_data = {
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'raw_path': str(self.raw_path),
                'processed_path': str(self.processed_path),
                'features_path': str(self.features_path),
                'feature_columns': self._feature_columns,
                'target_columns': self._target_columns,
                'label_encoders': self.label_encoders,
                'saved_at': datetime.now()
            }
            
            joblib.dump(pipeline_data, filepath)
            logger.info("✅ Pipeline sauvegardé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
            raise
    
    def load_pipeline(self, filepath: str = 'models/data_pipeline.joblib') -> None:
        """Chargement du pipeline"""
        logger.info(f"📂 Chargement du pipeline: {filepath}")
        
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Fichier pipeline non trouvé: {filepath}")
                
            pipeline_data = joblib.load(filepath)
            
            self.scaler = pipeline_data['scaler']
            self.is_fitted = pipeline_data['is_fitted']
            self._feature_columns = pipeline_data.get('feature_columns', [])
            self._target_columns = pipeline_data.get('target_columns', [])
            self.label_encoders = pipeline_data.get('label_encoders', {})
            
            saved_at = pipeline_data.get('saved_at', 'Unknown')
            logger.info(f"✅ Pipeline chargé (sauvegardé le: {saved_at})")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the pipeline preprocessing and feature engineering"""
        logger.info("🔄 Transformation des nouvelles données...")
        
        try:
            # Preprocess
            df_processed = self.marrakech_loader.preprocess_weather_data(df)
            # Feature engineering
            df_features = self.marrakech_loader.create_weather_features(df_processed)
            
            logger.info(f"✅ Transformation terminée: {df_features.shape}")
            return df_features
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la transformation: {e}")
            raise
            
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Charge et prépare les données pour la validation (wrapper pour Airflow).
        Exécute les étapes 1 à 3 si nécessaire.
        """
        logger.info("🔄 Chargement et préparation des données pour validation...")
        try:
            # Tenter de charger les features existantes
            features_file = self.features_path / "weather_data_features.csv"
            if features_file.exists():
                logger.info(f"📂 Chargement des features existantes: {features_file}")
                return pd.read_csv(features_file)
            
            # Sinon, exécuter le pipeline jusqu'à la création des features
            logger.info("⚠️ Features non trouvées, exécution du pipeline...")
            raw_data = self.step1_download_raw_data()
            processed_data = self.step2_preprocess_data(raw_data)
            features_data = self.step3_create_features(processed_data)
            
            return features_data
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la préparation des données: {e}")
            raise
    
    # ============================================================================
    # PIPELINE COMPLET
    # ============================================================================
    
    def run_full_pipeline(self, split_ratio: float = DEFAULT_SPLIT_RATIO) -> Dict[str, Any]:
        """Exécution complète du pipeline en 3 étapes"""
        logger.info("\n" + "=" * 70)
        logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET (3 ÉTAPES)")
        logger.info("=" * 70 + "\n")
        
        try:
            # ÉTAPE 1: Téléchargement depuis Kaggle → data/raw/
            raw_data = self.step1_download_raw_data()
            
            # ÉTAPE 2: Preprocessing → data/processed/
            processed_data = self.step2_preprocess_data(raw_data)
            
            # ÉTAPE 3: Création des features → data/features/
            features_data = self.step3_create_features(processed_data)
            
            # Validation
            is_valid, errors = self.validate_data(features_data)
            if not is_valid:
                raise ValueError(f"Validation échouée: {errors}")
            
            # Préparation ML
            ml_data = self.prepare_ml_data(features_data, split_ratio)
            
            # Sauvegarde du pipeline
            self.save_pipeline()
            
            # Statistiques finales
            stats = {
                'raw_shape': raw_data.shape,
                'processed_shape': processed_data.shape,
                'features_shape': features_data.shape,
                'train_shape': ml_data['X_train'].shape,
                'test_shape': ml_data['X_test'].shape,
                'feature_count': len(ml_data['feature_names']),
                'target': ml_data['target_names'],
                'pipeline_saved': True,
                'raw_file': str(self.raw_path / "weather_data_raw.csv"),
                'processed_file': str(self.processed_path / "weather_data_processed.csv"),
                'features_file': str(self.features_path / "weather_data_features.csv")
            }
            
            logger.info("\n" + "=" * 70)
            logger.info("🎉 PIPELINE COMPLET TERMINÉ AVEC SUCCÈS")
            logger.info("=" * 70)
            logger.info(f"📊 ÉTAPE 1 (Raw):       {stats['raw_shape']}")
            logger.info(f"📊 ÉTAPE 2 (Processed): {stats['processed_shape']}")
            logger.info(f"📊 ÉTAPE 3 (Features):  {stats['features_shape']}")
            logger.info(f"📂 Fichiers créés:")
            logger.info(f"   - {stats['raw_file']}")
            logger.info(f"   - {stats['processed_file']}")
            logger.info(f"   - {stats['features_file']}")
            logger.info("=" * 70 + "\n")
            
            return {
                'ml_data': ml_data,
                'stats': stats,
                'raw_data': raw_data,
                'processed_data': processed_data,
                'features_data': features_data
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur dans le pipeline complet: {e}")
            raise

# Alias pour compatibilité avec l'ancien code
ClimateDataPipeline = WeatherDataPipeline
DataPipeline = WeatherDataPipeline

def main():
    """Fonction principale pour tester le pipeline"""
    try:
        pipeline = WeatherDataPipeline()
        results = pipeline.run_full_pipeline()
        
        stats = results['stats']
        
        print("\n📊 RÉSUMÉ DU PIPELINE MÉTÉO:")
        print(f"   📥 Données brutes: {stats['raw_shape']}")
        print(f"   📊 Données préprocessées: {stats['processed_shape']}")
        print(f"   📊 Données avec features: {stats['features_shape']}")
        print(f"   🚂 Entraînement: {stats['train_shape']}")
        print(f"   🧪 Test: {stats['test_shape']}")
        print(f"   🎯 Variables cibles: {stats['target']}")
        print(f"   📊 Features: {stats['feature_count']}")
        print(f"   💾 Pipeline sauvegardé: {'✅' if stats['pipeline_saved'] else '❌'}")
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {e}")
        raise

if __name__ == "__main__":
    main()