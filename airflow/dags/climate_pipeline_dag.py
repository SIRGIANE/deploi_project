from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
import logging
import sys
import os

# Ajouter le chemin du workspace - CORRECTION pour macOS
sys.path.insert(0, '/workspace')
print(f"📂 Chemin du projet ajouté: /workspace")

import pandas as pd
import mlflow

# Configuration du logging
logger = logging.getLogger(__name__)

# Configuration du DAG
default_args = {
    'owner': 'climate-mlops',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'catchup': False,
}

dag = DAG(
    'climate_data_pipeline',
    default_args=default_args,
    description='Pipeline de données météorologiques - Collecte quotidienne, Training hebdomadaire',
    schedule_interval='0 6 * * *',  # Exécution quotidienne à 6h du matin
    catchup=False,
    tags=['weather', 'ml', 'climate'],
)

# ============================================================================
# FONCTIONS DES TÂCHES
# ============================================================================

def task_collect_today_data(**context):
    """Collecter les données météo d'aujourd'hui depuis Open-Meteo API"""
    try:
        logger.info("🌦️ [COLLECTE] Collecte des données météo d'aujourd'hui...")
        
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        
        mlflow.set_experiment("Climate_Marrakech_Airflow")
        with mlflow.start_run(run_name="daily_data_collection"):
            loader = MarrakechWeatherDataLoader()
            result = loader.collect_and_store_today_data()
            
            if result['success']:
                # Logger les métriques
                mlflow.log_metric("new_records", result['new_records'])
                mlflow.log_metric("total_records", result['total_records'])
                mlflow.log_metric("today_temp_mean", result['today_weather']['temperature_mean'])
                
                logger.info(f"✅ Données collectées: {result['new_records']} nouveaux enregistrements")
                logger.info(f"📊 Total: {result['total_records']} enregistrements")
                logger.info(f"🌡️ Température aujourd'hui: {result['today_weather']['temperature_mean']:.1f}°C")
                
                # Stocker le résultat pour la décision de training
                context['ti'].xcom_push(key='collection_result', value=result)
                
                return result
            else:
                raise Exception("Échec de la collecte des données")
                
    except Exception as e:
        logger.error(f"❌ Erreur lors de la collecte: {e}")
        raise

def check_training_needed(**context):
    """Vérifie si le re-training est nécessaire (tous les 7 jours)"""
    try:
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        
        loader = MarrakechWeatherDataLoader()
        days_since_training = loader.get_days_since_last_training()
        
        logger.info(f"📅 Jours depuis le dernier training: {days_since_training}")
        
        if days_since_training >= 7:
            logger.info("✅ Re-training nécessaire (>= 7 jours)")
            return 'do_training'
        else:
            logger.info(f"⏭️ Pas de re-training nécessaire. Prochain dans {7 - days_since_training} jours")
            return 'skip_training'
            
    except Exception as e:
        logger.warning(f"⚠️ Impossible de vérifier: {e}. Training par défaut.")
        return 'do_training'

def task_load_data(**context):
    """Charger les données brutes"""
    try:
        logger.info("🔄 [STEP 1] Chargement des données brutes...")
        
        from src.data_pipeline import WeatherDataPipeline
        
        mlflow.set_experiment("Climate_Marrakech_Airflow")
        with mlflow.start_run(run_name="step1_load_data"):
            pipeline = WeatherDataPipeline()
            raw_data = pipeline.step1_download_raw_data()
            
            mlflow.log_metric("raw_data_rows", len(raw_data))
            mlflow.log_metric("raw_data_cols", len(raw_data.columns))
            
            logger.info(f"✅ Données chargées: {len(raw_data)} lignes")
            return {'status': 'success', 'rows': len(raw_data)}
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement: {e}")
        raise

def task_preprocess_data(**context):
    """Prétraiter les données"""
    try:
        logger.info("🔄 [STEP 2] Prétraitement des données...")
        
        from src.data_pipeline import WeatherDataPipeline
        
        with mlflow.start_run(run_name="step2_preprocess"):
            pipeline = WeatherDataPipeline()
            processed_data = pipeline.step2_preprocess_data()
            
            mlflow.log_metric("processed_data_rows", len(processed_data))
            logger.info(f"✅ Données prétraitées: {len(processed_data)} lignes")
            return {'status': 'success', 'rows': len(processed_data)}
    except Exception as e:
        logger.error(f"❌ Erreur lors du prétraitement: {e}")
        raise

def task_create_features(**context):
    """Créer les features pour le ML"""
    try:
        logger.info("🔄 [STEP 3] Création des features...")
        
        from src.data_pipeline import WeatherDataPipeline
        
        with mlflow.start_run(run_name="step3_create_features"):
            pipeline = WeatherDataPipeline()
            features_data = pipeline.step3_create_features()
            
            mlflow.log_metric("features_rows", len(features_data))
            mlflow.log_metric("features_cols", len(features_data.columns))
            
            logger.info(f"✅ Features créées: {len(features_data)} lignes")
            return {'status': 'success', 'rows': len(features_data)}
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création des features: {e}")
        raise

def task_train_model(**context):
    """Entraîner le modèle ML"""
    try:
        logger.info("🔄 [STEP 4] Entraînement du modèle...")
        
        from src.train_model import train_random_forest_model
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        
        with mlflow.start_run(run_name="step4_train_model"):
            metrics = train_random_forest_model()
            mlflow.log_metrics(metrics)
            
            # Mettre à jour la date du dernier training
            loader = MarrakechWeatherDataLoader()
            loader.update_last_training_date()
            
            logger.info(f"✅ Modèle entraîné avec métriques: {metrics}")
            return {'status': 'success', 'metrics': metrics}
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        raise

# ============================================================================
# CRÉATION DES TÂCHES
# ============================================================================

# Tâche 1: Collecte quotidienne des données
task_collect = PythonOperator(
    task_id='collect_today_data',
    python_callable=task_collect_today_data,
    dag=dag,
)

# Tâche 2: Décision - Training nécessaire ?
task_check_training = BranchPythonOperator(
    task_id='check_training_needed',
    python_callable=check_training_needed,
    dag=dag,
)

# Branche A: Faire le training
task_load = PythonOperator(
    task_id='do_training',
    python_callable=task_load_data,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=task_preprocess_data,
    dag=dag,
)

task_features = PythonOperator(
    task_id='create_features',
    python_callable=task_create_features,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=task_train_model,
    dag=dag,
)

task_training_done = BashOperator(
    task_id='training_completed',
    bash_command='echo "✅ Training hebdomadaire terminé!"',
    dag=dag,
)

# Branche B: Skip le training
task_skip = EmptyOperator(
    task_id='skip_training',
    dag=dag,
)

task_skip_done = BashOperator(
    task_id='collection_only_completed',
    bash_command='echo "✅ Collecte quotidienne terminée (pas de training)"',
    dag=dag,
)

# Tâche finale
task_end = EmptyOperator(
    task_id='end',
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# ============================================================================
# DÉFINITION DES DÉPENDANCES
# ============================================================================

# Flux principal
task_collect >> task_check_training

# Branche Training (tous les 7 jours)
task_check_training >> task_load >> task_preprocess >> task_features >> task_train >> task_training_done >> task_end

# Branche Skip (collecte seulement)
task_check_training >> task_skip >> task_skip_done >> task_end
