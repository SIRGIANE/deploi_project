"""
Script d'évaluation avancée des modèles
Calcule des métriques détaillées et génère des rapports de comparaison
"""

import os
import sys
import logging
from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score
)
from scipy import stats
import mlflow
import mlflow.sklearn

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Classe pour l'évaluation complète des modèles"""
    
    def __init__(self, mlflow_uri="http://localhost:5050"):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.metrics_history = {}
        
    def compute_advanced_metrics(self, y_true, y_pred, model_name):
        """Calcul des métriques avancées pour la comparaison"""
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Résidus
        residuals = y_true - y_pred
        
        metrics = {
            'Model': model_name,
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'Median_AE': median_absolute_error(y_true, y_pred),
            'Explained_Variance': explained_variance_score(y_true, y_pred),
            'Max_Error': np.max(np.abs(residuals)),
            'Min_Error': np.min(np.abs(residuals)),
            'Std_Residuals': np.std(residuals),
            'Skewness_Residuals': stats.skew(residuals),
            'Kurtosis_Residuals': stats.kurtosis(residuals),
            'Mean_Residuals': np.mean(residuals),
            'RMSE_std': np.std(np.abs(residuals)),
            'Median_AE': median_absolute_error(y_true, y_pred),
            'Q95_Error': np.percentile(np.abs(residuals), 95),
            'Q99_Error': np.percentile(np.abs(residuals), 99)
        }
        
        return metrics
    
    def compare_models(self, models_results):
        """Comparaison de plusieurs modèles"""
        logger.info("📊 Comparaison des modèles...")
        
        all_metrics = []
        
        for model_name, y_true, y_pred in models_results:
            metrics = self.compute_advanced_metrics(y_true, y_pred, model_name)
            all_metrics.append(metrics)
        
        metrics_df = pd.DataFrame(all_metrics)
        
        # Sauvegarde en JSON
        metrics_json = metrics_df.to_json(orient='records', indent=2)
        with open('model_metrics_comparison.json', 'w') as f:
            f.write(metrics_json)
        
        logger.info("✅ Métriques sauvegardées dans model_metrics_comparison.json")
        
        return metrics_df
    
    def generate_comparison_report(self, metrics_df, output_path='reports/model_comparison.html'):
        """Génération d'un rapport HTML de comparaison"""
        logger.info("📝 Génération du rapport de comparaison...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Création du rapport HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #ecf0f1; }}
                .metric-highlight {{ background-color: #f39c12; font-weight: bold; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; }}
            </style>
        </head>
        <body>
            <h1>🎯 Model Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>📊 Métriques Principales</h2>
                {metrics_df[['Model', 'RMSE', 'MAE', 'R²', 'MAPE']].to_html(classes='data')}
            </div>
            
            <div class="section">
                <h2>📈 Métriques Avancées</h2>
                {metrics_df[['Model', 'Explained_Variance', 'Max_Error', 'Min_Error', 'Std_Residuals']].to_html(classes='data')}
            </div>
            
            <div class="section">
                <h2>🏆 Meilleur Modèle par Métrique</h2>
                <ul>
                    <li><strong>RMSE :</strong> {metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']} ({metrics_df['RMSE'].min():.4f})</li>
                    <li><strong>MAE :</strong> {metrics_df.loc[metrics_df['MAE'].idxmin(), 'Model']} ({metrics_df['MAE'].min():.4f})</li>
                    <li><strong>R² :</strong> {metrics_df.loc[metrics_df['R²'].idxmax(), 'Model']} ({metrics_df['R²'].max():.4f})</li>
                    <li><strong>MAPE :</strong> {metrics_df.loc[metrics_df['MAPE'].idxmin(), 'Model']} ({metrics_df['MAPE'].min():.4f}%)</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"✅ Rapport généré: {output_path}")
        
    def plot_model_comparison(self, metrics_df, output_path='reports/model_comparison.png'):
        """Génération de graphiques de comparaison"""
        logger.info("📊 Génération des graphiques...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Comparison - Advanced Metrics', fontsize=16, fontweight='bold')
        
        # 1. RMSE vs MAE
        axes[0, 0].scatter(metrics_df['RMSE'], metrics_df['MAE'], s=200, alpha=0.6, c=range(len(metrics_df)))
        for i, model in enumerate(metrics_df['Model']):
            axes[0, 0].annotate(model, (metrics_df['RMSE'].iloc[i], metrics_df['MAE'].iloc[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 0].set_xlabel('RMSE (°C)', fontweight='bold')
        axes[0, 0].set_ylabel('MAE (°C)', fontweight='bold')
        axes[0, 0].set_title('RMSE vs MAE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R² Score
        colors = ['#2ecc71' if x > 0.98 else '#f39c12' if x > 0.95 else '#e74c3c' for x in metrics_df['R²']]
        axes[0, 1].barh(metrics_df['Model'], metrics_df['R²'], color=colors)
        axes[0, 1].set_xlabel('R² Score', fontweight='bold')
        axes[0, 1].set_title('R² Score par Modèle')
        axes[0, 1].set_xlim([0.97, 1.0])
        for i, v in enumerate(metrics_df['R²']):
            axes[0, 1].text(v - 0.003, i, f'{v:.4f}', ha='right', va='center', fontweight='bold')
        
        # 3. MAPE
        axes[0, 2].barh(metrics_df['Model'], metrics_df['MAPE'], color=['#3498db', '#e74c3c', '#2ecc71'][:len(metrics_df)])
        axes[0, 2].set_xlabel('MAPE (%)', fontweight='bold')
        axes[0, 2].set_title('Mean Absolute Percentage Error')
        
        # 4. Error Distribution
        axes[1, 0].boxplot([metrics_df['RMSE'], metrics_df['MAE'], metrics_df['Max_Error']], 
                          labels=['RMSE', 'MAE', 'Max Error'])
        axes[1, 0].set_ylabel('Error (°C)', fontweight='bold')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. Max vs Min Error
        x_pos = np.arange(len(metrics_df))
        width = 0.35
        axes[1, 1].bar(x_pos - width/2, metrics_df['Max_Error'], width, label='Max Error', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, metrics_df['Min_Error'], width, label='Min Error', alpha=0.8)
        axes[1, 1].set_ylabel('Error (°C)', fontweight='bold')
        axes[1, 1].set_title('Max vs Min Error')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics_df['Model'], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Explained Variance
        axes[1, 2].barh(metrics_df['Model'], metrics_df['Explained_Variance'], 
                       color=['#27ae60', '#3498db', '#e67e22'][:len(metrics_df)])
        axes[1, 2].set_xlabel('Explained Variance', fontweight='bold')
        axes[1, 2].set_title('Variance Expliquée')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Graphiques sauvegardés: {output_path}")
        plt.close()
    
    def log_to_mlflow(self, metrics_df):
        """Enregistrement des métriques dans MLflow"""
        logger.info("📤 Enregistrement dans MLflow...")
        
        with mlflow.start_run(run_name=f"Model_Evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            for _, row in metrics_df.iterrows():
                for col in metrics_df.columns:
                    if col != 'Model' and isinstance(row[col], (int, float)):
                        mlflow.log_metric(f"{row['Model']}_{col}", row[col])
            
            # Log du fichier JSON
            mlflow.log_artifact('model_metrics_comparison.json')
            
            logger.info("✅ Métriques enregistrées dans MLflow")

def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage de l'évaluation des modèles")
    
    try:
        # Chemins des fichiers
        model_path = Path('models/rf_model.pkl')
        X_test_path = Path('data/features/X_test.npy')
        y_test_path = Path('data/features/y_test.npy')
        scaler_path = Path('models/scaler.pkl')
        output_path = Path('reports/model_comparison/evaluation_metrics.json')
        
        # Vérification de l'existence des fichiers
        if not all(p.exists() for p in [model_path, X_test_path, y_test_path]):
            logger.error("❌ Fichiers manquants pour l'évaluation")
            sys.exit(1)
            
        # Chargement des données et du modèle
        import joblib
        model = joblib.load(model_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        # Chargement du scaler si nécessaire (si le modèle attend des données non scalées mais que X_test est scalé, ou inversement)
        # Dans ce pipeline, X_test est déjà scalé lors de la préparation
        
        logger.info(f"✅ Modèle et données chargés. Test shape: {X_test.shape}")
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Instanciation de l'évaluateur
        evaluator = ModelEvaluator()
        
        # Calcul des métriques
        # Note: y_test peut être multi-output
        metrics_list = []
        
        # Si y_test est 2D
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            for i in range(y_test.shape[1]):
                target_metrics = evaluator.compute_advanced_metrics(
                    y_test[:, i], 
                    y_pred[:, i] if y_pred.ndim > 1 else y_pred, 
                    f"RandomForest_Target_{i+1}"
                )
                metrics_list.append(target_metrics)
        else:
            metrics = evaluator.compute_advanced_metrics(y_test, y_pred, "RandomForest")
            metrics_list.append(metrics)
            
        metrics_df = pd.DataFrame(metrics_list)
        
        # Sauvegarde des métriques pour DVC
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format simple pour DVC metrics
        dvc_metrics = {
            "avg_rmse": float(metrics_df['RMSE'].mean()),
            "avg_mae": float(metrics_df['MAE'].mean()),
            "avg_r2": float(metrics_df['R²'].mean())
        }
        
        with open(output_path, 'w') as f:
            json.dump(dvc_metrics, f, indent=2)
            
        # Génération du rapport complet
        evaluator.generate_comparison_report(metrics_df)
        evaluator.plot_model_comparison(metrics_df)
        
        # Log MLflow
        evaluator.log_to_mlflow(metrics_df)
        
        logger.info(f"✅ Évaluation terminée. Métriques sauvegardées dans {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'évaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
