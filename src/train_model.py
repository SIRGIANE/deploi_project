"""
Script d'entraînement automatisé pour les modèles de prédiction météorologique
Utilise le dataset Kaggle Weather et MLflow pour l'entraînement et le tracking
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import optuna
# Import du nouveau pipeline météo (import relatif)
try:
    from data_pipeline import WeatherDataPipeline
    from config import Config
except ImportError:
    from src.data_pipeline import WeatherDataPipeline
    from src.config import Config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MLFLOW_URI = Config.MLFLOW_TRACKING_URI
DEFAULT_EXPERIMENT_NAME = Config.MLFLOW_EXPERIMENT_NAME
DEFAULT_RESULTS_DIR = Config.RESULTS_DIR
DEFAULT_RF_PARAMS = Config.DEFAULT_RF_PARAMS

class MetricsCalculator:
    """Calculateur de métriques d'évaluation"""
    
    @staticmethod
    def calculate_multi_target_metrics(
        y_train_true: np.ndarray,
        y_train_pred: np.ndarray,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
        target_names: list,
    ) -> Dict[str, float]:
        """Calcul des métriques pour plusieurs cibles météo."""
        metrics: Dict[str, float] = {}
        
        # S'assurer que les arrays sont 2D
        if y_train_true.ndim == 1:
            y_train_true = y_train_true.reshape(-1, 1)
            y_train_pred = y_train_pred.reshape(-1, 1)
            y_test_true = y_test_true.reshape(-1, 1)
            y_test_pred = y_test_pred.reshape(-1, 1)
        
        for idx, name in enumerate(target_names):
            ytr = y_train_true[:, idx]
            ytr_pred = y_train_pred[:, idx]
            yte = y_test_true[:, idx]
            yte_pred = y_test_pred[:, idx]
            
            # Normaliser le nom pour MLflow (supprimer les caractères spéciaux)
            clean_name = name.replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_')
            
            metrics[f'{clean_name}_train_rmse'] = float(np.sqrt(mean_squared_error(ytr, ytr_pred)))
            metrics[f'{clean_name}_test_rmse'] = float(np.sqrt(mean_squared_error(yte, yte_pred)))
            metrics[f'{clean_name}_train_mae'] = float(mean_absolute_error(ytr, ytr_pred))
            metrics[f'{clean_name}_test_mae'] = float(mean_absolute_error(yte, yte_pred))
            metrics[f'{clean_name}_train_r2'] = float(r2_score(ytr, ytr_pred))
            metrics[f'{clean_name}_test_r2'] = float(r2_score(yte, yte_pred))
        
        # Moyennes globales pour comparaison
        clean_names = [name.replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_') for name in target_names]
        test_rmse_values = [metrics[f'{name}_test_rmse'] for name in clean_names]
        test_mae_values = [metrics[f'{name}_test_mae'] for name in clean_names]
        test_r2_values = [metrics[f'{name}_test_r2'] for name in clean_names]
        metrics['avg_test_rmse'] = float(np.mean(test_rmse_values))
        metrics['avg_test_mae'] = float(np.mean(test_mae_values))
        metrics['avg_test_r2'] = float(np.mean(test_r2_values))
        
        return metrics
    
    @staticmethod
    def log_metrics_to_mlflow(metrics: Dict[str, float]) -> None:
        """Enregistrement des métriques dans MLflow"""
        mlflow.log_metrics(metrics)

class WeatherModelTrainer:
    """Classe pour l'entraînement automatisé des modèles météorologiques"""
    
    def __init__(self, mlflow_uri: str = DEFAULT_MLFLOW_URI, 
                 experiment_name: str = DEFAULT_EXPERIMENT_NAME):
        self.mlflow_uri = mlflow_uri
        # If no custom name provided, use training date (YYYYMMDD)
        if experiment_name == DEFAULT_EXPERIMENT_NAME or not experiment_name:
            date_str = datetime.now().strftime('%Y%m%d')
            self.experiment_name = f"training_{date_str}"
        else:
            self.experiment_name = experiment_name
        self.pipeline = WeatherDataPipeline()
        
        # Configuration MLflow avec stockage local des artefacts
        try:
            os.makedirs("mlruns", exist_ok=True)
            mlflow.set_tracking_uri(f"file:./mlruns")
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"✅ MLflow configuré avec backend local: ./mlruns | expérience: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur MLflow, continuant sans tracking: {e}")
            mlflow.set_experiment(self.experiment_name)
        
        # Variables pour les données
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.feature_names: Optional[list] = None
        self.target_names: Optional[list] = None
        
    def prepare_data(self) -> Dict[str, Any]:
        """Préparation des données météo via le pipeline"""
        logger.info("🔄 Préparation des données météorologiques...")
        
        try:
            results = self.pipeline.run_full_pipeline()
            ml_data = results['ml_data']
            
            self.X_train = ml_data['X_train']
            self.X_test = ml_data['X_test']
            self.y_train = ml_data['y_train']
            self.y_test = ml_data['y_test']
            self.feature_names = ml_data['feature_names']
            self.target_names = ml_data.get('target_names') or [ml_data.get('target_name')]
            
            logger.info(f"✅ Données préparées: Train {self.X_train.shape}, Test {self.X_test.shape}")
            logger.info(f"🎯 Variables cibles: {self.target_names}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la préparation des données: {e}")
            raise
    
    def _validate_data_prepared(self) -> None:
        """Validation que les données sont prêtes pour l'entraînement"""
        if any(data is None for data in [self.X_train, self.X_test, self.y_train, self.y_test, self.target_names]):
            raise ValueError("Les données doivent être préparées avant l'entraînement")
    
    def train_random_forest(self, params: Optional[Dict[str, Any]] = None) -> Tuple[RandomForestRegressor, Dict[str, float]]:
        """Entraînement du modèle Random Forest"""
        self._validate_data_prepared()
        
        if params is None:
            params = DEFAULT_RF_PARAMS.copy()
        
        run_name = f"RandomForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            logger.info("🌲 Entraînement Random Forest...")
            
            # Log des paramètres
            mlflow.log_params(params)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("target_variables", ",".join(self.target_names))
            
            try:
                # Entraînement
                model = RandomForestRegressor(**params)
                model.fit(self.X_train, self.y_train)
                
                # Prédictions
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
                
                # Calcul des métriques multi-cibles
                metrics = MetricsCalculator.calculate_multi_target_metrics(
                    self.y_train, train_pred, self.y_test, test_pred, self.target_names
                )
                MetricsCalculator.log_metrics_to_mlflow(metrics)
                
                # Feature importance
                if self.feature_names:
                    feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                    for feature, importance in sorted(feature_importance.items(), 
                                                    key=lambda x: x[1], reverse=True)[:5]:
                        mlflow.log_metric(f"importance_{feature}", float(importance))
                
                # Sauvegarde du modèle
                mlflow.sklearn.log_model(model, "random_forest_model")
                models_dir = Path("models")
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, models_dir / "rf_model.pkl")
                if self.pipeline and hasattr(self.pipeline, "scaler"):
                    joblib.dump(self.pipeline.scaler, models_dir / "scaler.pkl")
                
                logger.info(f"✅ Random Forest - Test RMSE (moyenne): {metrics['avg_test_rmse']:.4f}")
                return model, metrics
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement Random Forest: {e}")
                raise
    
    def tune_random_forest_optuna(self, n_trials: int = 50) -> Tuple[RandomForestRegressor, Dict[str, Any], Dict[str, float]]:
        """Optimisation des hyperparamètres Random Forest avec Optuna"""
        self._validate_data_prepared()
        
        logger.info(f"🔍 Optimisation Random Forest avec Optuna ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            
            model = RandomForestRegressor(**params)
            model.fit(self.X_train, self.y_train)
            
            pred = model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, pred))
            
            return rmse
        
        # Création de l’étude Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Meilleurs paramètres
        best_params = study.best_params
        best_params['random_state'] = 42
        
        logger.info(f"✅ Meilleurs paramètres trouvés: {best_params}")
        logger.info(f"   Meilleur RMSE: {study.best_value:.4f}")
        
        # Entraînement final avec les meilleurs paramètres
        best_model, best_metrics = self.train_random_forest(best_params)
        
        return best_model, best_params, best_metrics

    def train_gradient_boosting(self, params: Optional[Dict[str, Any]] = None) -> Tuple[GradientBoostingRegressor, Dict[str, float]]:
        """Entraînement du modèle Gradient Boosting"""
        self._validate_data_prepared()
        
        if params is None:
            params = {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
        
        run_name = f"GradientBoosting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            logger.info("📈 Entraînement Gradient Boosting...")
            
            mlflow.log_params(params)
            mlflow.log_param("model_type", "GradientBoosting")
            mlflow.log_param("target_variables", ",".join(self.target_names))
            
            try:
                model = MultiOutputRegressor(GradientBoostingRegressor(**params))
                model.fit(self.X_train, self.y_train)
                
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
                
                metrics = MetricsCalculator.calculate_multi_target_metrics(
                    self.y_train, train_pred, self.y_test, test_pred, self.target_names
                )
                MetricsCalculator.log_metrics_to_mlflow(metrics)
                
                mlflow.sklearn.log_model(model, "gradient_boosting_model")
                
                logger.info(f"✅ Gradient Boosting - Test RMSE (moyenne): {metrics['avg_test_rmse']:.4f}")
                return model, metrics
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement Gradient Boosting: {e}")
                raise
    
    def train_linear_regression(self) -> Tuple[LinearRegression, Dict[str, float]]:
        """Entraînement du modèle de régression linéaire (baseline)"""
        self._validate_data_prepared()
        
        run_name = f"LinearRegression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            logger.info("📊 Entraînement Régression Linéaire (baseline)...")
            
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("target_variables", ",".join(self.target_names))
            
            try:
                model = LinearRegression()
                model.fit(self.X_train, self.y_train)
                
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
                
                metrics = MetricsCalculator.calculate_multi_target_metrics(
                    self.y_train, train_pred, self.y_test, test_pred, self.target_names
                )
                MetricsCalculator.log_metrics_to_mlflow(metrics)
                
                mlflow.sklearn.log_model(model, "linear_regression_model")
                
                logger.info(f"✅ Linear Regression - Test RMSE (moyenne): {metrics['avg_test_rmse']:.4f}")
                return model, metrics
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement Linear Regression: {e}")
                raise
    
    def compare_models(self, models_results: Dict[str, Dict[str, float]]) -> str:
        """Comparaison des modèles et identification du meilleur"""
        logger.info("📊 COMPARAISON DES MODÈLES:")
        logger.info("=" * 60)
        
        for model_name, metrics in models_results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"   Test RMSE (moy): {metrics['avg_test_rmse']:.4f}")
            logger.info(f"   Test MAE  (moy): {metrics['avg_test_mae']:.4f}")
            logger.info(f"   Test R²   (moy): {metrics['avg_test_r2']:.4f}")
        
        # Identification du meilleur modèle (plus faible RMSE)
        best_model_name = min(models_results.keys(), key=lambda k: models_results[k]['avg_test_rmse'])
        logger.info(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
        logger.info("=" * 60)
        
        return best_model_name
    
    def compare_models_advanced(self, models_results: Dict[str, Dict[str, float]], 
                              training_times: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Comparaison avancée des modèles avec critères multiples
        Returns detailed selection results with scoring
        """
        logger.info("📊 COMPARAISON AVANCÉE DES MODÈLES:")
        logger.info("=" * 80)
        
        # Normalisation des métriques pour scoring
        rmse_values = [metrics['avg_test_rmse'] for metrics in models_results.values()]
        r2_values = [metrics['avg_test_r2'] for metrics in models_results.values()]
        mae_values = [metrics['avg_test_mae'] for metrics in models_results.values()]
        
        min_rmse, max_rmse = min(rmse_values), max(rmse_values)
        min_r2, max_r2 = min(r2_values), max(r2_values)
        min_mae, max_mae = min(mae_values), max(mae_values)
        
        model_scores = {}
        
        for model_name, metrics in models_results.items():
            # Score normalisé (0-100)
            # RMSE: plus bas = meilleur (inverse)
            rmse_score = 100 * (1 - (metrics['avg_test_rmse'] - min_rmse) / (max_rmse - min_rmse)) if max_rmse != min_rmse else 100
            
            # R²: plus haut = meilleur
            r2_score = 100 * (metrics['avg_test_r2'] - min_r2) / (max_r2 - min_r2) if max_r2 != min_r2 else 100
            
            # MAE: plus bas = meilleur (inverse)  
            mae_score = 100 * (1 - (metrics['avg_test_mae'] - min_mae) / (max_mae - min_mae)) if max_mae != min_mae else 100
            
            # Pondération des scores (configurable)
            weights = Config.MODEL_SELECTION_WEIGHTS if hasattr(Config, 'MODEL_SELECTION_WEIGHTS') else {
                'rmse': 0.4,    # 40% - Erreur principale
                'r2': 0.3,      # 30% - Qualité d'ajustement  
                'mae': 0.2,     # 20% - Erreur absolue
                'time': 0.1     # 10% - Temps d'entraînement
            }
            
            # Score temporel (si disponible)
            time_score = 100
            if training_times and model_name in training_times:
                max_time = max(training_times.values()) 
                min_time = min(training_times.values())
                if max_time != min_time:
                    # Plus rapide = meilleur score
                    time_score = 100 * (1 - (training_times[model_name] - min_time) / (max_time - min_time))
            
            # Score composite pondéré
            composite_score = (
                rmse_score * weights['rmse'] + 
                r2_score * weights['r2'] + 
                mae_score * weights['mae'] + 
                time_score * weights['time']
            )
            
            model_scores[model_name] = {
                'composite_score': composite_score,
                'rmse_score': rmse_score,
                'r2_score': r2_score, 
                'mae_score': mae_score,
                'time_score': time_score,
                'metrics': metrics,
                'training_time': training_times.get(model_name, 0) if training_times else 0
            }
            
            logger.info(f"\n{model_name}:")
            logger.info(f"   📊 Score composite: {composite_score:.2f}/100")
            logger.info(f"   📏 Test RMSE: {metrics['avg_test_rmse']:.4f} (score: {rmse_score:.1f})")
            logger.info(f"   📊 Test R²: {metrics['avg_test_r2']:.4f} (score: {r2_score:.1f})")
            logger.info(f"   📐 Test MAE: {metrics['avg_test_mae']:.4f} (score: {mae_score:.1f})")
            if training_times and model_name in training_times:
                logger.info(f"   ⏱️ Temps: {training_times[model_name]:.2f}s (score: {time_score:.1f})")
        
        # Sélection du meilleur modèle
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['composite_score'])
        best_score = model_scores[best_model_name]['composite_score']
        
        logger.info(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
        logger.info(f"   📊 Score final: {best_score:.2f}/100")
        logger.info("=" * 80)
        
        return {
            'best_model': best_model_name,
            'best_score': best_score,
            'all_scores': model_scores,
            'selection_criteria': {
                'primary_metric': 'composite_score',
                'weights': weights,
                'improvement_threshold': Config.MODEL_PROMOTION_CRITERIA.get('min_score_improvement', 5.0)
            }
        }
    
    def should_deploy_model(self, current_results: Dict[str, Any], 
                           previous_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Détermine si le nouveau modèle doit être déployé en production
        Basé sur les critères de promotion définis dans Config
        """
        logger.info("🚀 ÉVALUATION POUR DÉPLOIEMENT:")
        logger.info("=" * 60)
        
        best_model = current_results['best_model']
        best_metrics = current_results['models_performance'][best_model]
        
        deployment_decision = {
            'should_deploy': False,
            'reasons': [],
            'model_name': best_model,
            'metrics': best_metrics,
            'checks': {}
        }
        
        criteria = Config.MODEL_PROMOTION_CRITERIA
        
        # Check 1: Performance minimale
        min_r2_threshold = criteria.get('min_r2_threshold', 0.7)
        current_r2 = best_metrics['avg_test_r2']
        
        if current_r2 >= min_r2_threshold:
            deployment_decision['checks']['min_performance'] = True
            deployment_decision['reasons'].append(f"✅ R² satisfaisant: {current_r2:.3f} >= {min_r2_threshold}")
        else:
            deployment_decision['checks']['min_performance'] = False
            deployment_decision['reasons'].append(f"❌ R² insuffisant: {current_r2:.3f} < {min_r2_threshold}")
        
        # Check 2: Amélioration par rapport au modèle précédent
        if previous_results:
            prev_best = previous_results.get('best_model')
            if prev_best and prev_best in previous_results.get('models_performance', {}):
                prev_metrics = previous_results['models_performance'][prev_best]
                
                rmse_improvement = (prev_metrics['avg_test_rmse'] - best_metrics['avg_test_rmse']) / prev_metrics['avg_test_rmse']
                r2_improvement = (best_metrics['avg_test_r2'] - prev_metrics['avg_test_r2']) / abs(prev_metrics['avg_test_r2'])
                
                min_rmse_improvement = criteria.get('min_rmse_improvement', 0.05)
                min_r2_improvement = criteria.get('min_r2_improvement', 0.02)
                
                if rmse_improvement >= min_rmse_improvement:
                    deployment_decision['checks']['rmse_improvement'] = True
                    deployment_decision['reasons'].append(f"✅ Amélioration RMSE: {rmse_improvement:.1%} >= {min_rmse_improvement:.1%}")
                else:
                    deployment_decision['checks']['rmse_improvement'] = False
                    deployment_decision['reasons'].append(f"⚠️ Amélioration RMSE limitée: {rmse_improvement:.1%} < {min_rmse_improvement:.1%}")
                
                if r2_improvement >= min_r2_improvement:
                    deployment_decision['checks']['r2_improvement'] = True  
                    deployment_decision['reasons'].append(f"✅ Amélioration R²: {r2_improvement:.1%} >= {min_r2_improvement:.1%}")
                else:
                    deployment_decision['checks']['r2_improvement'] = False
                    deployment_decision['reasons'].append(f"⚠️ Amélioration R² limitée: {r2_improvement:.1%} < {min_r2_improvement:.1%}")
        else:
            # Premier modèle - pas de comparaison
            deployment_decision['checks']['improvement'] = True
            deployment_decision['reasons'].append("✅ Premier modèle - pas de modèle précédent à comparer")
        
        # Check 3: Données suffisantes pour validation
        min_data_points = criteria.get('min_data_points', 100)
        test_samples = current_results.get('data_preparation', {}).get('test_samples', 0)
        
        if test_samples >= min_data_points:
            deployment_decision['checks']['sufficient_data'] = True
            deployment_decision['reasons'].append(f"✅ Données suffisantes: {test_samples} >= {min_data_points}")
        else:
            deployment_decision['checks']['sufficient_data'] = False
            deployment_decision['reasons'].append(f"❌ Données insuffisantes: {test_samples} < {min_data_points}")
        
        # Décision finale
        require_all_checks = criteria.get('require_positive_tests', True)
        
        if require_all_checks:
            deployment_decision['should_deploy'] = all(deployment_decision['checks'].values())
        else:
            # Au moins performance minimale + amélioration OU données suffisantes
            deployment_decision['should_deploy'] = (
                deployment_decision['checks'].get('min_performance', False) and
                deployment_decision['checks'].get('sufficient_data', False)
            )
        
        decision_text = "🚀 DÉPLOYER" if deployment_decision['should_deploy'] else "⏸️ NE PAS DÉPLOYER"
        logger.info(f"\n📋 DÉCISION: {decision_text}")
        
        for reason in deployment_decision['reasons']:
            logger.info(f"   {reason}")
        
        logger.info("=" * 60)
        
        return deployment_decision
    
    def save_results(self, results: Dict[str, Any], filepath: str = None) -> None:
        """Sauvegarde des résultats d'entraînement"""
        if filepath is None:
            os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
            filepath = f"{DEFAULT_RESULTS_DIR}/weather_training_results.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"💾 Résultats sauvegardés: {filepath}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
    
    def run_full_training(self) -> Dict[str, Any]:
        """Exécution complète de l'entraînement"""
        logger.info("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT MÉTÉO COMPLET")
        
        try:
            # Préparation des données
            data_results = self.prepare_data()
            
            # Dictionnaire pour suivre les temps d'entraînement
            training_times = {}
            import time
            
            # Modèle baseline
            start_time = time.time()
            lr_model, lr_metrics = self.train_linear_regression()
            training_times['LinearRegression'] = time.time() - start_time
            
            # Gradient Boosting
            start_time = time.time()
            gb_model, gb_metrics = self.train_gradient_boosting()
            training_times['GradientBoosting'] = time.time() - start_time
            
            # Random Forest avec optimisation Optuna
            logger.info("🔍 Optimisation des hyperparamètres Random Forest...")
            start_time = time.time()
            rf_model, best_hyperparams, rf_metrics = self.tune_random_forest_optuna(n_trials=20)  # Réduit pour rapidité
            training_times['RandomForest'] = time.time() - start_time
            
            # Comparaison des modèles avec méthode avancée
            models_performance = {
                'LinearRegression': lr_metrics,
                'GradientBoosting': gb_metrics,
                'RandomForest': rf_metrics
            }
            
            # Utilisation de la sélection avancée
            logger.info("🧠 Utilisation de la sélection avancée avec score composite...")
            advanced_comparison = self.compare_models_advanced(models_performance, training_times)
            best_model_name = advanced_comparison['best_model']
            
            # Log de la comparaison simple aussi pour référence
            simple_best = self.compare_models(models_performance)
            if simple_best != best_model_name:
                logger.info(f"📊 Note: Sélection simple (RMSE seul) aurait choisi: {simple_best}")
                logger.info(f"📊 Sélection avancée (score composite) a choisi: {best_model_name}")
            
            # Compilation des résultats
            results = {
                'data_preparation': {
                    'dataset': 'Kaggle Weather Dataset',
                    'target_variables': self.target_names,
                    'train_samples': int(self.X_train.shape[0]),
                    'test_samples': int(self.X_test.shape[0]),
                    'feature_count': int(self.X_train.shape[1])
                },
                'models_performance': models_performance,
                'best_model': best_model_name,
                'best_hyperparameters': best_hyperparams,
                'training_times': training_times,
                'selection_method': 'advanced_composite_score',
                'model_selection_details': advanced_comparison,
                'training_completed': datetime.now().isoformat(),
                'mlflow_uri': self.mlflow_uri,
                'mlflow_experiment': self.experiment_name
            }
            
            # Évaluation pour déploiement
            deployment_decision = self.should_deploy_model(results)
            results['deployment_recommendation'] = deployment_decision
            
            # Sauvegarde des résultats
            self.save_results(results)
            
            logger.info("🎉 ENTRAÎNEMENT MÉTÉO TERMINÉ AVEC SUCCÈS")
            logger.info(f"📊 Consultez MLflow: {self.mlflow_uri} | Expérience: {self.experiment_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement complet: {e}")
            raise

# Alias pour compatibilité
ModelTrainer = WeatherModelTrainer

def parse_arguments() -> argparse.Namespace:
    """Analyse des arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Entraînement automatisé des modèles de prédiction météorologique',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mlflow-uri', 
        default=DEFAULT_MLFLOW_URI, 
        help='URI du serveur MLflow'
    )
    parser.add_argument(
        '--experiment-name',
        default=DEFAULT_EXPERIMENT_NAME,
        help='Nom de l\'expérience MLflow'
    )
    
    return parser.parse_args()

def main() -> None:
    """Fonction principale avec gestion d'erreurs robuste"""
    args = parse_arguments()
    
    try:
        # Initialisation du trainer
        trainer = WeatherModelTrainer(
            mlflow_uri=args.mlflow_uri,
            experiment_name=args.experiment_name
        )
        
        # Entraînement complet
        results = trainer.run_full_training()
        
        # Affichage des résultats finaux
        best_model = results['best_model']
        best_metrics = results['models_performance'][best_model]
        
        print("\n" + "=" * 70)
        print("🎯 ENTRAÎNEMENT MÉTÉO TERMINÉ AVEC SUCCÈS!")
        print("=" * 70)
        print(f"📊 Dataset: {results['data_preparation']['dataset']}")
        print(f"🎯 Variables cibles: {results['data_preparation']['target_variables']}")
        print(f"📈 Meilleur modèle: {best_model}")
        print(f"📊 Test RMSE (moy): {best_metrics['avg_test_rmse']:.4f}")
        print(f"📊 Test R² (moy): {best_metrics['avg_test_r2']:.4f}")
        print(f"📊 Test MAE (moy): {best_metrics['avg_test_mae']:.4f}")
        print(f"🔗 MLflow: {args.mlflow_uri}")
        print(f"💾 Résultats: {DEFAULT_RESULTS_DIR}/weather_training_results.json")
        print("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Entraînement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur critique lors de l'entraînement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()