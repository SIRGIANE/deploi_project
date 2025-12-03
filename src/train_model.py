"""
Script d'entraînement automatisé pour les modèles climatiques
Utilise le pipeline de données et MLflow pour l'entraînement et le tracking
"""

import os
import sys
import logging
from datetime import datetime
import argparse
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import optuna

# Import du pipeline local
from data_pipeline import ClimateDataPipeline

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Classe pour l'entraînement automatisé des modèles"""
    
    def __init__(self, mlflow_uri="http://localhost:5050"):
        self.mlflow_uri = mlflow_uri
        self.pipeline = ClimateDataPipeline()
        self.experiment_name = "Climate_Temperature_Prediction"
        
        # Configuration MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)
        
    def prepare_data(self):
        """Préparation des données via le pipeline"""
        logger.info("🔄 Préparation des données...")
        
        results = self.pipeline.run_full_pipeline()
        ml_data = results['ml_data']
        
        self.X_train = ml_data['X_train']
        self.X_test = ml_data['X_test']
        self.y_train = ml_data['y_train']
        self.y_test = ml_data['y_test']
        self.feature_names = ml_data['feature_names']
        
        logger.info(f"✅ Données préparées: Train {self.X_train.shape}, Test {self.X_test.shape}")
        
        return results
    
    def train_random_forest(self, params=None):
        """Entraînement du modèle Random Forest"""
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        
        with mlflow.start_run(run_name=f"RandomForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info("🌲 Entraînement Random Forest...")
            
            # Log des paramètres
            mlflow.log_params(params)
            mlflow.log_param("model_type", "RandomForest")
            
            # Entraînement
            model = RandomForestRegressor(**params)
            model.fit(self.X_train, self.y_train)
            
            # Prédictions et métriques
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            metrics = self._calculate_metrics(self.y_train, train_pred, self.y_test, test_pred)
            mlflow.log_metrics(metrics)
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            # Sauvegarde du modèle
            mlflow.sklearn.log_model(
                model, 
                "random_forest_model",
                registered_model_name="RandomForest_Climate"
            )
            
            logger.info(f"✅ Random Forest - Test RMSE: {metrics['test_rmse']:.4f}")
            return model, metrics
    
    def train_linear_regression(self):
        """Entraînement du modèle de régression linéaire (baseline)"""
        with mlflow.start_run(run_name=f"LinearRegression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info("📈 Entraînement Régression Linéaire...")
            
            mlflow.log_param("model_type", "LinearRegression")
            
            # Entraînement
            model = LinearRegression()
            model.fit(self.X_train, self.y_train)
            
            # Prédictions et métriques
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            metrics = self._calculate_metrics(self.y_train, train_pred, self.y_test, test_pred)
            mlflow.log_metrics(metrics)
            
            # Sauvegarde du modèle
            mlflow.sklearn.log_model(
                model,
                "linear_regression_model", 
                registered_model_name="LinearRegression_Climate"
            )
            
            logger.info(f"✅ Linear Regression - Test RMSE: {metrics['test_rmse']:.4f}")
            return model, metrics
    
    def optimize_random_forest(self, n_trials=50):
        """Optimisation des hyperparamètres avec Optuna"""
        logger.info(f"🔍 Optimisation des hyperparamètres ({n_trials} essais)...")
        
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
            test_pred = model.predict(self.X_test)
            
            return mean_squared_error(self.y_test, test_pred, squared=False)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        logger.info(f"🏆 Meilleurs paramètres: {best_params}")
        
        # Entraînement avec les meilleurs paramètres
        best_model, best_metrics = self.train_random_forest(best_params)
        
        # Log des résultats d'optimisation
        with mlflow.start_run(run_name=f"Optuna_Optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(best_params)
            mlflow.log_metrics({
                'best_rmse': study.best_value,
                'n_trials': n_trials
            })
            mlflow.log_dict(best_params, "best_hyperparameters.json")
        
        return best_model, best_metrics, best_params
    
    def _calculate_metrics(self, y_train_true, y_train_pred, y_test_true, y_test_pred):
        """Calcul des métriques d'évaluation"""
        return {
            'train_rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_true, y_test_pred)),
            'train_mae': mean_absolute_error(y_train_true, y_train_pred),
            'test_mae': mean_absolute_error(y_test_true, y_test_pred),
            'train_r2': r2_score(y_train_true, y_train_pred),
            'test_r2': r2_score(y_test_true, y_test_pred)
        }
    
    def run_full_training(self, optimize=True, n_trials=50):
        """Exécution complète de l'entraînement"""
        logger.info("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT COMPLET")
        
        # Préparation des données
        data_results = self.prepare_data()
        
        # Modèle baseline
        lr_model, lr_metrics = self.train_linear_regression()
        
        # Random Forest
        if optimize:
            rf_model, rf_metrics, best_params = self.optimize_random_forest(n_trials)
        else:
            rf_model, rf_metrics = self.train_random_forest()
            best_params = None
        
        # Comparaison des modèles
        comparison = {
            'LinearRegression': lr_metrics,
            'RandomForest': rf_metrics
        }
        
        # Identification du meilleur modèle
        best_model_name = min(comparison.keys(), key=lambda k: comparison[k]['test_rmse'])
        
        logger.info("📊 RÉSUMÉ DES RÉSULTATS:")
        logger.info("=" * 50)
        for model_name, metrics in comparison.items():
            logger.info(f"{model_name}:")
            logger.info(f"   Test RMSE: {metrics['test_rmse']:.4f}")
            logger.info(f"   Test R²: {metrics['test_r2']:.4f}")
        
        logger.info(f"🏆 MEILLEUR MODÈLE: {best_model_name}")
        
        results = {
            'data_preparation': data_results,
            'models_performance': comparison,
            'best_model': best_model_name,
            'best_hyperparameters': best_params,
            'training_completed': datetime.now().isoformat()
        }
        
        # Sauvegarde des résultats
        os.makedirs('results', exist_ok=True)
        with open('results/training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        logger.info(f"📊 Consultez MLflow: {self.mlflow_uri}")
        
        return results

def main():
    """Fonction principale avec arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Entraînement des modèles climatiques')
    parser.add_argument('--optimize', action='store_true', help='Optimiser les hyperparamètres')
    parser.add_argument('--trials', type=int, default=50, help='Nombre d\'essais pour l\'optimisation')
    parser.add_argument('--mlflow-uri', default='http://localhost:5050', help='URI MLflow')
    
    args = parser.parse_args()
    
    try:
        trainer = ModelTrainer(mlflow_uri=args.mlflow_uri)
        results = trainer.run_full_training(optimize=args.optimize, n_trials=args.trials)
        
        print("\n🎯 ENTRAÎNEMENT RÉUSSI!")
        print(f"📈 Meilleur modèle: {results['best_model']}")
        print(f"📊 RMSE: {results['models_performance'][results['best_model']]['test_rmse']:.4f}")
        print(f"🔗 MLflow: {args.mlflow_uri}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()