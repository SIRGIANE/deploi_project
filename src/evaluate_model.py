"""
Script d'évaluation des modèles météorologiques entraînés
Charge les modèles depuis MLflow et génère des rapports détaillés
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from config import Config
except ImportError:
    from src.config import Config

class ModelEvaluator:
    """Évaluateur de modèles météorologiques"""
    
    def __init__(self, mlflow_uri: str = "file:./mlruns"):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Répertoires de sortie
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_latest_training_results(self) -> Optional[Dict[str, Any]]:
        """Charge les derniers résultats d'entraînement"""
        results_file = self.results_dir / "weather_training_results.json"
        
        if not results_file.exists():
            logger.error(f"❌ Fichier de résultats introuvable: {results_file}")
            return None
            
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"✅ Résultats d'entraînement chargés: {results_file}")
            return results
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des résultats: {e}")
            return None
    
    def evaluate_latest_models(self) -> Dict[str, Any]:
        """Évalue les derniers modèles entraînés"""
        logger.info("📊 ÉVALUATION DES MODÈLES MÉTÉO")
        logger.info("=" * 50)
        
        # Charger les résultats d'entraînement
        training_results = self.load_latest_training_results()
        if not training_results:
            raise ValueError("Impossible de charger les résultats d'entraînement")
        
        best_model_name = training_results['best_model']
        models_performance = training_results['models_performance']
        
        logger.info(f"🏆 Meilleur modèle identifié: {best_model_name}")
        
        # Évaluation détaillée
        evaluation_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'performance_summary': self._generate_performance_summary(models_performance),
            'model_comparison': self._compare_models_detailed(models_performance),
            'deployment_ready': training_results.get('deployment_recommendation', {}).get('should_deploy', False),
            'data_quality_check': self._check_data_quality(training_results),
            'recommendations': self._generate_recommendations(training_results)
        }
        
        # Génération des rapports
        self._generate_model_card(evaluation_results, training_results)
        self._generate_comparison_report(evaluation_results)
        
        # Sauvegarde des résultats d'évaluation
        eval_file = self.results_dir / "model_evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info(f"✅ Évaluation terminée. Rapport sauvé: {eval_file}")
        return evaluation_results
    
    def _generate_performance_summary(self, models_performance: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Génère un résumé des performances"""
        summary = {}
        
        for model_name, metrics in models_performance.items():
            summary[model_name] = {
                'test_rmse': round(metrics['avg_test_rmse'], 4),
                'test_r2': round(metrics['avg_test_r2'], 4),
                'test_mae': round(metrics['avg_test_mae'], 4),
                'performance_grade': self._grade_performance(metrics['avg_test_r2'])
            }
        
        return summary
    
    def _grade_performance(self, r2_score: float) -> str:
        """Attribue une note de performance basée sur le R²"""
        if r2_score >= 0.95:
            return "A+ (Excellent)"
        elif r2_score >= 0.90:
            return "A (Très bon)"
        elif r2_score >= 0.80:
            return "B (Bon)"
        elif r2_score >= 0.70:
            return "C (Acceptable)"
        else:
            return "D (Insuffisant)"
    
    def _compare_models_detailed(self, models_performance: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Comparaison détaillée des modèles"""
        comparison = {
            'ranking_by_rmse': [],
            'ranking_by_r2': [],
            'performance_gaps': {}
        }
        
        # Classement par RMSE (plus bas = meilleur)
        rmse_ranking = sorted(models_performance.items(), key=lambda x: x[1]['avg_test_rmse'])
        comparison['ranking_by_rmse'] = [
            {'model': name, 'rmse': metrics['avg_test_rmse']} 
            for name, metrics in rmse_ranking
        ]
        
        # Classement par R² (plus haut = meilleur)
        r2_ranking = sorted(models_performance.items(), key=lambda x: x[1]['avg_test_r2'], reverse=True)
        comparison['ranking_by_r2'] = [
            {'model': name, 'r2': metrics['avg_test_r2']} 
            for name, metrics in r2_ranking
        ]
        
        # Calcul des écarts de performance
        best_rmse = rmse_ranking[0][1]['avg_test_rmse']
        best_r2 = r2_ranking[0][1]['avg_test_r2']
        
        for model_name, metrics in models_performance.items():
            rmse_gap = ((metrics['avg_test_rmse'] - best_rmse) / best_rmse * 100) if best_rmse > 0 else 0
            r2_gap = ((best_r2 - metrics['avg_test_r2']) / best_r2 * 100) if best_r2 > 0 else 0
            
            comparison['performance_gaps'][model_name] = {
                'rmse_gap_percent': round(rmse_gap, 2),
                'r2_gap_percent': round(r2_gap, 2)
            }
        
        return comparison
    
    def _check_data_quality(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Vérification de la qualité des données"""
        data_prep = training_results.get('data_preparation', {})
        
        quality_check = {
            'total_samples': data_prep.get('train_samples', 0) + data_prep.get('test_samples', 0),
            'train_test_ratio': round(data_prep.get('train_samples', 0) / data_prep.get('test_samples', 1), 2),
            'feature_count': data_prep.get('feature_count', 0),
            'target_variables': data_prep.get('target_variables', []),
            'data_quality_score': 'Good'  # Simplifiée pour cet exemple
        }
        
        # Évaluation de la qualité
        if quality_check['total_samples'] < 1000:
            quality_check['data_quality_score'] = 'Limited'
        elif quality_check['train_test_ratio'] < 3 or quality_check['train_test_ratio'] > 5:
            quality_check['data_quality_score'] = 'Fair'
        
        return quality_check
    
    def _generate_recommendations(self, training_results: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur les résultats"""
        recommendations = []
        
        best_model = training_results['best_model']
        best_metrics = training_results['models_performance'][best_model]
        deployment_decision = training_results.get('deployment_recommendation', {})
        
        # Recommandations basées sur les performances
        if best_metrics['avg_test_r2'] >= 0.95:
            recommendations.append("🏆 Excellentes performances - Modèle prêt pour la production")
        elif best_metrics['avg_test_r2'] >= 0.90:
            recommendations.append("✅ Bonnes performances - Déploiement recommandé")
        else:
            recommendations.append("⚠️ Performances limitées - Considérer plus de données ou features")
        
        # Recommandations basées sur le déploiement
        if deployment_decision.get('should_deploy', False):
            recommendations.append("🚀 Déploiement automatique approuvé")
        else:
            reasons = deployment_decision.get('reasons', [])
            if reasons:
                recommendations.append(f"⏸️ Déploiement en attente: {'; '.join(reasons)}")
        
        # Recommandations techniques
        if best_model == 'LinearRegression':
            recommendations.append("💡 Relation linéaire détectée - Modèle simple mais efficace")
        elif best_model in ['RandomForest', 'GradientBoosting']:
            recommendations.append("🌲 Modèle complexe sélectionné - Surveiller le surapprentissage")
        
        return recommendations
    
    def _generate_model_card(self, evaluation_results: Dict[str, Any], training_results: Dict[str, Any]) -> None:
        """Génère une carte de modèle détaillée"""
        model_card_path = self.reports_dir / "model_card.md"
        
        best_model = evaluation_results['best_model']
        best_metrics = evaluation_results['performance_summary'][best_model]
        
        model_card_content = f"""# Model Card - Climate MLOps

## Informations générales
- **Modèle**: {best_model}
- **Version**: {datetime.now().strftime('%Y.%m.%d')}
- **Date d'évaluation**: {evaluation_results['evaluation_timestamp']}
- **Statut**: {'✅ Prêt pour déploiement' if evaluation_results['deployment_ready'] else '⏸️ En attente'}

## Performances
- **RMSE Test**: {best_metrics['test_rmse']}°C
- **R² Test**: {best_metrics['test_r2']} ({best_metrics['performance_grade']})
- **MAE Test**: {best_metrics['test_mae']}°C

## Dataset
- **Source**: {training_results['data_preparation']['dataset']}
- **Échantillons total**: {evaluation_results['data_quality_check']['total_samples']}
- **Variables cibles**: {', '.join(training_results['data_preparation']['target_variables'])}
- **Features**: {evaluation_results['data_quality_check']['feature_count']}

## Comparaison des modèles
"""
        
        for model_name, summary in evaluation_results['performance_summary'].items():
            indicator = "🏆" if model_name == best_model else "  "
            model_card_content += f"- {indicator} **{model_name}**: RMSE={summary['test_rmse']}, R²={summary['test_r2']} ({summary['performance_grade']})\n"
        
        model_card_content += f"""
## Recommandations
"""
        for rec in evaluation_results['recommendations']:
            model_card_content += f"- {rec}\n"
        
        model_card_content += f"""
## Métriques techniques
- **URI MLflow**: {training_results.get('mlflow_uri', 'N/A')}
- **Expérience**: {training_results.get('mlflow_experiment', 'N/A')}
- **Méthode de sélection**: {training_results.get('selection_method', 'N/A')}

*Généré automatiquement par Climate MLOps Pipeline*
"""
        
        with open(model_card_path, 'w') as f:
            f.write(model_card_content)
        
        logger.info(f"📄 Carte de modèle générée: {model_card_path}")
    
    def _generate_comparison_report(self, evaluation_results: Dict[str, Any]) -> None:
        """Génère un rapport de comparaison JSON pour les artefacts"""
        comparison_path = Path("model_metrics_comparison.json")
        
        comparison_data = {
            'evaluation_timestamp': evaluation_results['evaluation_timestamp'],
            'best_model': evaluation_results['best_model'],
            'models_ranking': evaluation_results['model_comparison']['ranking_by_rmse'],
            'performance_summary': evaluation_results['performance_summary'],
            'deployment_recommendation': evaluation_results['deployment_ready']
        }
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"📊 Rapport de comparaison généré: {comparison_path}")

def main():
    """Fonction principale d'évaluation"""
    try:
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_latest_models()
        
        print("\n" + "=" * 60)
        print("📊 ÉVALUATION DES MODÈLES TERMINÉE")
        print("=" * 60)
        print(f"🏆 Meilleur modèle: {evaluation_results['best_model']}")
        print(f"🚀 Prêt pour déploiement: {'Oui' if evaluation_results['deployment_ready'] else 'Non'}")
        print(f"📄 Rapports générés dans: ./reports/")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'évaluation: {e}")
        raise

if __name__ == "__main__":
    main()