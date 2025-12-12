"""
📊 Module de comparaison de modèles
Compare les performances des nouveaux modèles avec ceux en production
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

logger = logging.getLogger(__name__)

class ModelComparator:
    """Comparateur de performance entre modèles"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.reports_dir = Path("reports/model_comparison")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        
    def load_production_model_info(self):
        """Chargement des informations du modèle en production"""
        try:
            # Recherche du modèle en production dans MLflow
            client = mlflow.tracking.MlflowClient()
            
            # Chercher le modèle avec le tag "production"
            production_models = []
            
            for experiment in client.list_experiments():
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.stage = 'production'"
                )
                
                for run in runs:
                    production_models.append({
                        'run_id': run.info.run_id,
                        'experiment_id': run.info.experiment_id,
                        'metrics': run.data.metrics,
                        'tags': run.data.tags,
                        'start_time': run.info.start_time
                    })
            
            if not production_models:
                logger.warning("⚠️ Aucun modèle en production trouvé")
                return None
            
            # Prendre le plus récent
            latest_prod = max(production_models, key=lambda x: x['start_time'])
            
            logger.info(f"📋 Modèle en production trouvé: {latest_prod['run_id']}")
            return latest_prod
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle production: {e}")
            return None
    
    def load_current_model_metrics(self):
        """Chargement des métriques du modèle actuel"""
        try:
            # Chargement depuis le fichier de métriques DVC
            metrics_files = [
                "metrics.json",
                "evaluation_metrics.json"
            ]
            
            current_metrics = {}
            
            for metrics_file in metrics_files:
                if Path(metrics_file).exists():
                    with open(metrics_file, 'r') as f:
                        file_metrics = json.load(f)
                        current_metrics.update(file_metrics)
            
            # Recherche du run MLflow le plus récent
            client = mlflow.tracking.MlflowClient()
            
            # Obtenir l'expérience par défaut
            experiment = client.get_experiment_by_name("Default")
            if experiment is None:
                experiments = client.list_experiments()
                if experiments:
                    experiment = experiments[0]
            
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if runs:
                    latest_run = runs[0]
                    mlflow_metrics = latest_run.data.metrics
                    current_metrics.update(mlflow_metrics)
                    
                    current_model_info = {
                        'run_id': latest_run.info.run_id,
                        'metrics': current_metrics,
                        'tags': latest_run.data.tags,
                        'start_time': latest_run.info.start_time
                    }
                    
                    logger.info(f"📊 Métriques modèle actuel chargées: {len(current_metrics)} métriques")
                    return current_model_info
            
            # Si pas de MLflow, utiliser seulement les fichiers locaux
            if current_metrics:
                return {
                    'run_id': 'local_run',
                    'metrics': current_metrics,
                    'tags': {},
                    'start_time': datetime.now().timestamp() * 1000
                }
            
            logger.warning("⚠️ Aucune métrique actuelle trouvée")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement métriques actuelles: {e}")
            return None
    
    def evaluate_all_models(self):
        """Évaluation de tous les modèles disponibles"""
        try:
            logger.info("📊 Évaluation de tous les modèles...")
            
            # Chargement des données de test
            test_features_path = "data/features/X_test.npy"
            test_targets_path = "data/features/y_test.npy"
            
            if not (Path(test_features_path).exists() and Path(test_targets_path).exists()):
                logger.error("❌ Données de test manquantes")
                return {}
            
            X_test = np.load(test_features_path)
            y_test = np.load(test_targets_path)
            
            model_results = {}
            
            # Évaluation des modèles disponibles
            model_files = {
                'random_forest': 'models/rf_model.pkl',
                'lstm': 'models/lstm_model.h5',
                'scaler': 'models/scaler.pkl'
            }
            
            # Chargement du scaler
            # Note: X_test.npy contient déjà des données scalées par le pipeline
            # On ne doit pas les rescaler ici
            X_test_scaled = X_test
            
            # Évaluation Random Forest
            if Path(model_files['random_forest']).exists():
                try:
                    rf_model = joblib.load(model_files['random_forest'])
                    rf_predictions = rf_model.predict(X_test_scaled)
                    
                    rf_metrics = self.calculate_metrics(y_test, rf_predictions, 'random_forest')
                    model_results['random_forest'] = rf_metrics
                    
                except Exception as e:
                    logger.error(f"❌ Erreur évaluation Random Forest: {e}")
            
            # Évaluation LSTM (si TensorFlow/Keras disponible)
            if Path(model_files['lstm']).exists():
                try:
                    import tensorflow as tf
                    from tensorflow import keras
                    
                    lstm_model = keras.models.load_model(model_files['lstm'])
                    
                    # Reshape pour LSTM si nécessaire
                    if len(X_test_scaled.shape) == 2:
                        X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
                    else:
                        X_test_lstm = X_test_scaled
                    
                    lstm_predictions = lstm_model.predict(X_test_lstm).flatten()
                    
                    lstm_metrics = self.calculate_metrics(y_test, lstm_predictions, 'lstm')
                    model_results['lstm'] = lstm_metrics
                    
                except Exception as e:
                    logger.error(f"❌ Erreur évaluation LSTM: {e}")
            
            logger.info(f"✅ {len(model_results)} modèles évalués")
            return model_results
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation modèles: {e}")
            return {}
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calcul des métriques de performance"""
        try:
            metrics = {
                'model_name': model_name,
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                'predictions_count': len(y_pred),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Métriques additionnelles
            residuals = y_true - y_pred
            metrics.update({
                'mean_residual': np.mean(residuals),
                'std_residual': np.std(residuals),
                'max_error': np.max(np.abs(residuals)),
                'median_error': np.median(np.abs(residuals))
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul métriques {model_name}: {e}")
            return {'model_name': model_name, 'error': str(e)}
    
    def compare_with_production(self, current_evaluations):
        """Comparaison avec le modèle en production"""
        try:
            logger.info("🔄 Comparaison avec le modèle en production...")
            
            production_model = self.load_production_model_info()
            
            if not production_model:
                # Pas de modèle en production, promouvoir le meilleur actuel
                best_current = self.select_best_model(current_evaluations)
                
                return {
                    'new_model_better': True,
                    'best_model': best_current,
                    'production_model': None,
                    'comparison_reason': 'no_production_model',
                    'recommendation': 'deploy_best_current'
                }
            
            # Comparaison des métriques
            production_metrics = production_model['metrics']
            best_current = self.select_best_model(current_evaluations)
            
            if not best_current:
                return {
                    'new_model_better': False,
                    'best_model': None,
                    'production_model': production_model,
                    'comparison_reason': 'no_valid_current_model',
                    'recommendation': 'keep_production'
                }
            
            # Critères de comparaison (RMSE principal)
            improvement_threshold = 0.05  # 5% d'amélioration minimum
            
            prod_rmse = production_metrics.get('rmse', float('inf'))
            current_rmse = best_current.get('rmse', float('inf'))
            
            improvement = (prod_rmse - current_rmse) / prod_rmse if prod_rmse > 0 else 0
            
            # Critères additionnels
            criteria_met = 0
            total_criteria = 3
            
            # 1. RMSE amélioration
            if improvement >= improvement_threshold:
                criteria_met += 1
            
            # 2. R² amélioration
            prod_r2 = production_metrics.get('r2', 0)
            current_r2 = best_current.get('r2', 0)
            if current_r2 > prod_r2 * 1.02:  # 2% d'amélioration
                criteria_met += 1
            
            # 3. MAE amélioration
            prod_mae = production_metrics.get('mae', float('inf'))
            current_mae = best_current.get('mae', float('inf'))
            if current_mae < prod_mae * 0.98:  # 2% d'amélioration
                criteria_met += 1
            
            should_promote = criteria_met >= 2  # Au moins 2 critères sur 3
            
            comparison_results = {
                'new_model_better': should_promote,
                'best_model': best_current,
                'production_model': production_model,
                'improvement_percentage': improvement * 100,
                'criteria_met': f"{criteria_met}/{total_criteria}",
                'detailed_comparison': {
                    'rmse': {'production': prod_rmse, 'current': current_rmse, 'improvement': improvement},
                    'r2': {'production': prod_r2, 'current': current_r2},
                    'mae': {'production': prod_mae, 'current': current_mae}
                },
                'recommendation': 'promote' if should_promote else 'keep_production'
            }
            
            # Sauvegarde du rapport de comparaison
            self.save_comparison_report(comparison_results)
            
            logger.info(f"🏆 Nouveau modèle {'recommandé' if should_promote else 'pas recommandé'} "
                       f"(amélioration: {improvement*100:.2f}%)")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"❌ Erreur comparaison avec production: {e}")
            return {
                'new_model_better': False,
                'error': str(e),
                'recommendation': 'keep_production'
            }
    
    def select_best_model(self, evaluations):
        """Sélection du meilleur modèle basé sur les métriques"""
        if not evaluations:
            return None
        
        # Critère principal: RMSE le plus bas
        best_model = None
        best_rmse = float('inf')
        
        for model_name, metrics in evaluations.items():
            if 'error' in metrics:
                continue
                
            model_rmse = metrics.get('rmse', float('inf'))
            
            if model_rmse < best_rmse:
                best_rmse = model_rmse
                best_model = metrics.copy()
                best_model['name'] = model_name
                best_model['path'] = f"models/{model_name}_model.pkl"
        
        return best_model
    
    def save_comparison_report(self, results):
        """Sauvegarde du rapport de comparaison"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.reports_dir / f"model_comparison_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Rapport de comparaison sauvegardé: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde rapport comparaison: {e}")

# Fonctions pour Airflow
def evaluate_all_models():
    """Fonction d'évaluation appelée par Airflow"""
    comparator = ModelComparator()
    return comparator.evaluate_all_models()

def compare_with_production(evaluation_results):
    """Fonction de comparaison appelée par Airflow"""
    comparator = ModelComparator()
    return comparator.compare_with_production(evaluation_results)

if __name__ == "__main__":
    # Test local
    logging.basicConfig(level=logging.INFO)
    
    comparator = ModelComparator()
    
    # Évaluation
    evaluations = comparator.evaluate_all_models()
    print(f"📊 Évaluations: {list(evaluations.keys())}")
    
    # Comparaison
    comparison = comparator.compare_with_production(evaluations)
    print(f"🏆 Recommandation: {comparison['recommendation']}")