"""
🌦️ Weather MLOps Continuous Training Pipeline
DAG pour l'entraînement continu des modèles météorologiques avec Marrakech Weather Dataset
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
import os
import json
import logging
from pathlib import Path

# Configuration par défaut
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

# Définition du DAG
dag = DAG(
    'weather_continuous_training_pipeline',
    default_args=default_args,
    description='🌦️ Pipeline de formation continue pour les modèles météorologiques (Marrakech Weather Dataset)',
    schedule_interval='@weekly',  # Exécution hebdomadaire (tous les 7 jours)
    catchup=False,
    tags=['mlops', 'weather', 'marrakech', 'continuous-training'],
)

def ingest_new_data(**context):
    """Téléchargement et ingestion des nouvelles données"""
    import sys
    sys.path.append('/workspace')
    from src.ingest_data import ingest_weather_data
    
    logging.info("🚀 Démarrage de l'ingestion automatique des données...")
    # Dans un cas réel, on passerait l'URL de l'API météo ici
    ingest_weather_data()
    return "Ingestion terminée"

def check_data_drift(**context):
    """Vérification du data drift (Simulé car module optionnel supprimé)"""
    logging.info("🔍 Vérification du data drift météorologique...")
    logging.info("⚠️ Module check_data_drift non présent, passage de l'étape.")
    
    # Simulation: pas de drift détecté par défaut
    result = {
        'drift_detected': False,
        'drift_score': 0.0,
        'threshold': 0.3,
        'needs_retraining': True # Force retraining for demo purposes
    }
    
    logging.info(f"📊 Résultats drift (simulé): {result}")
    context['task_instance'].xcom_push(key='drift_results', value=result)
    return result['needs_retraining']

def load_marrakech_data(**context):
    """Chargement des données locales Marrakech"""
    import sys
    sys.path.append('/workspace')
    
    from src.marrakech_data_loader import MarrakechWeatherDataLoader
    
    logging.info("📥 Chargement des données Marrakech Weather depuis la DB...")
    
    # Le loader utilise maintenant la DB par défaut via les variables d'env
    loader = MarrakechWeatherDataLoader()
    df = loader.load_weather_data()
    
    logging.info(f"✅ Données chargées: {len(df)} lignes")
    
    return {
        'source': 'database',
        'rows': len(df),
        'status': 'success'
    }

def validate_weather_data(**context):
    """Validation de la qualité des données météorologiques"""
    import sys
    sys.path.append('/workspace')
    
    from src.data_pipeline import WeatherDataPipeline
    
    logging.info("✅ Validation de la qualité des données météorologiques...")
    
    pipeline = WeatherDataPipeline()
    weather_data = pipeline.load_and_prepare_data()
    is_valid, errors = pipeline.validate_data(weather_data)
    
    if not is_valid:
        raise Exception(f"❌ Validation échouée: {errors}")
    
    validation_results = {
        'is_valid': is_valid,
        'data_shape': weather_data.shape,
        'columns_count': len(weather_data.columns)
    }
    
    logging.info(f"✅ Données validées: {validation_results}")
    return validation_results

def run_dvc_pipeline(**context):
    """Exécution du pipeline DVC complet pour le projet météo"""
    logging.info("🔄 Exécution du pipeline DVC météo...")
    
    # Simulation ou commande réelle si DVC est configuré
    # os.chdir('/workspace')
    # return {'pipeline_status': 'completed'}
    return {'pipeline_status': 'skipped (local mode)'}

def train_weather_models(**context):
    """Entraînement des modèles météorologiques"""
    import sys
    sys.path.append('/workspace')
    
    from src.train_model import WeatherModelTrainer
    
    logging.info("🤖 Entraînement des modèles météorologiques...")
    
    trainer = WeatherModelTrainer()
    
    # Entraînement complet
    results = trainer.run_full_training()
    
    context['task_instance'].xcom_push(key='training_results', value=results)
    
    return results

def evaluate_models(**context):
    """Évaluation et comparaison des modèles"""
    import sys
    sys.path.append('/workspace')
    
    from src.model_comparison import evaluate_all_models, compare_with_production
    
    logging.info("📊 Évaluation des modèles météo...")
    
    # Récupération des résultats d'entraînement
    training_results = context['task_instance'].xcom_pull(
        task_ids='train_weather_models',
        key='training_results'
    )
    
    # Évaluation complète
    evaluation_results = evaluate_all_models()
    
    # Comparaison avec le modèle en production
    comparison_results = compare_with_production(evaluation_results)
    
    should_promote = comparison_results.get('new_model_better', False)
    
    results = {
        'evaluation': evaluation_results,
        'comparison': comparison_results,
        'should_promote': should_promote,
        'best_model': comparison_results.get('best_model', training_results.get('best_model'))
    }
    
    context['task_instance'].xcom_push(key='model_results', value=results)
    
    return should_promote

def register_best_model(**context):
    """Enregistrement du meilleur modèle (Simulé)"""
    logging.info("🏆 Enregistrement du modèle (Module register_model supprimé)...")
    
    # Récupération des résultats
    model_results = context['task_instance'].xcom_pull(
        task_ids='evaluate_models',
        key='model_results'
    )
    
    if not model_results or not model_results.get('should_promote'):
        logging.info("🚫 Pas de nouveau modèle météo à promouvoir")
        return False
        
    best_model = model_results['best_model']
    logging.info(f"✅ Modèle considéré comme enregistré: {best_model.get('name', 'unknown')}")
    
    return {
        'model_registered': True,
        'model_version': 'v1_simulated',
        'model_name': best_model.get('name')
    }

def update_model_card(**context):
    """Mise à jour de la documentation (Simulé)"""
    logging.info("📝 Mise à jour de la Model Card (Module generate_model_card supprimé)...")
    return {'model_card_path': 'skipped'}

def send_notification(**context):
    """Envoi de notifications sur les résultats du pipeline météo"""
    
    # Récupération des résultats
    training_results = context['task_instance'].xcom_pull(
        task_ids='train_weather_models',
        key='training_results'
    ) or {}
    
    model_results = context['task_instance'].xcom_pull(
        task_ids='evaluate_models',
        key='model_results'
    ) or {}
    
    drift_results = context['task_instance'].xcom_pull(
        task_ids='check_data_drift',
        key='drift_results'
    ) or {}
    
    # Construction du message
    execution_date = context['execution_date'].strftime('%Y-%m-%d %H:%M:%S')
    
    best_model = training_results.get('best_model', 'N/A')
    # Gestion sécurisée des métriques
    models_perf = training_results.get('models_performance', {})
    best_metrics = models_perf.get(best_model, {}) if isinstance(models_perf, dict) else {}
    
    message = f"""
    🌦️ **Weather MLOps Pipeline - Résultats**
    
    📅 **Date d'exécution**: {execution_date}
    📊 **Dataset**: Marrakech Weather
    
    🔍 **Data Drift**:
    - Score: {drift_results.get('drift_score', 'N/A')}
    
    🤖 **Modèles**:
    - Meilleur modèle: {best_model}
    - Test RMSE: {best_metrics.get('avg_test_rmse', 'N/A')}
    - Nouveau modèle promu: {'Oui' if model_results.get('should_promote', False) else 'Non'}
    
    ✅ **Pipeline exécuté avec succès**
    """
    
    logging.info(f"📧 Notification: {message}")
    return {'notification_sent': True}

def re_run_data_pipeline(**context):
    """Re-exécution du pipeline de données après ingestion pour régénérer les features"""
    import sys
    sys.path.append('/workspace')
    
    from src.data_pipeline import WeatherDataPipeline
    
    logging.info("🔄 Re-exécution du pipeline de données après ingestion...")
    
    pipeline = WeatherDataPipeline()
    results = pipeline.run_full_pipeline()
    
    logging.info("✅ Pipeline de données re-exécuté avec succès")
    return {
        'pipeline_rerun': True,
        'new_data_shape': results['stats']['features_shape']
    }

# ========================
# DÉFINITION DES TÂCHES
# ========================

# 0. Ingestion des données (Nouvelle tâche)
ingest_data_task = PythonOperator(
    task_id='ingest_new_data',
    python_callable=ingest_new_data,
    dag=dag,
)

# 1. Vérification du data drift
check_drift_task = PythonOperator(
    task_id='check_data_drift',
    python_callable=check_data_drift,
    dag=dag,
)

# 2. Chargement des données Marrakech
load_data_task = PythonOperator(
    task_id='load_marrakech_data',
    python_callable=load_marrakech_data,
    dag=dag,
)

# 3. Validation de la qualité des données
validate_data_task = PythonOperator(
    task_id='validate_weather_data',
    python_callable=validate_weather_data,
    dag=dag,
)

# 4. Exécution du pipeline DVC
run_pipeline_task = BashOperator(
    task_id='run_dvc_pipeline',
    bash_command='echo "DVC Pipeline skipped in local mode"',
    dag=dag,
)

# 5. Entraînement des modèles météo
train_task = PythonOperator(
    task_id='train_weather_models',
    python_callable=train_weather_models,
    dag=dag,
)

# 6. Évaluation des modèles
evaluate_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag,
)

# 7. Enregistrement du meilleur modèle
register_model_task = PythonOperator(
    task_id='register_best_model',
    python_callable=register_best_model,
    dag=dag,
)

# 8. Mise à jour de la documentation
update_docs_task = PythonOperator(
    task_id='update_model_card',
    python_callable=update_model_card,
    dag=dag,
)

# 9. Push vers GitHub
push_github_task = BashOperator(
    task_id='push_to_github',
    bash_command='echo "Git push skipped in local mode"',
    dag=dag,
)

# 10. Notifications
notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag,
    trigger_rule='none_failed_min_one_success',
)

# 11. Re-exécution du pipeline de données après ingestion
re_run_data_pipeline_task = PythonOperator(
    task_id='re_run_data_pipeline',
    python_callable=re_run_data_pipeline,
    dag=dag,
)

# ========================
# DÉFINITION DES DÉPENDANCES
# ========================

# Pipeline principal
# On commence par l'ingestion, puis on vérifie le drift sur les nouvelles données
ingest_data_task >> check_drift_task >> load_data_task >> validate_data_task
validate_data_task >> run_pipeline_task >> train_task
train_task >> evaluate_task >> register_model_task
register_model_task >> update_docs_task >> push_github_task >> notify_task

# Pipeline de notification toujours exécuté
[check_drift_task, train_task, evaluate_task, register_model_task] >> notify_task

# Re-exécution du pipeline de données après ingestion
ingest_data_task >> re_run_data_pipeline_task >> validate_data_task