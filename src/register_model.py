"""
Script d'enregistrement et de promotion des modèles météorologiques
Gère l'enregistrement des modèles dans MLflow Model Registry et leur promotion
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

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

class ModelRegistry:
    """Gestionnaire d'enregistrement et de promotion des modèles"""
    
    def __init__(self, mlflow_uri: str = "file:./mlruns"):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient()
        
        # Répertoires
        self.results_dir = Path("results")
        self.models_dir = Path("models")
        
    def load_latest_results(self) -> Optional[Dict[str, Any]]:
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
    
    def register_best_model(self) -> Dict[str, Any]:
        """Enregistre le meilleur modèle dans MLflow Model Registry"""
        logger.info("📦 ENREGISTREMENT DU MEILLEUR MODÈLE")
        logger.info("=" * 50)
        
        # Charger les résultats
        training_results = self.load_latest_results()
        if not training_results:
            raise ValueError("Impossible de charger les résultats d'entraînement")
        
        best_model_name = training_results['best_model']
        deployment_decision = training_results.get('deployment_recommendation', {})
        
        logger.info(f"🏆 Modèle à enregistrer: {best_model_name}")
        
        # Configuration du modèle
        model_config = {
            'model_name': f"weather-{best_model_name.lower()}",
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'description': f"Modèle météorologique {best_model_name} - {datetime.now().strftime('%Y-%m-%d')}",
            'metrics': training_results['models_performance'][best_model_name],
            'deployment_ready': deployment_decision.get('should_deploy', False)
        }
        
        # Enregistrement du modèle
        registration_result = self._register_model_version(
            model_config, 
            training_results
        )
        
        # Promotion si approuvé pour le déploiement
        if model_config['deployment_ready']:
            promotion_result = self._promote_to_production(registration_result)
            registration_result.update(promotion_result)
        
        # Sauvegarde des informations d'enregistrement
        self._save_registration_info(registration_result)
        
        logger.info("✅ Enregistrement du modèle terminé")
        return registration_result
    
    def _register_model_version(self, model_config: Dict[str, Any], 
                               training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enregistre une nouvelle version du modèle"""
        
        model_name = model_config['model_name']
        
        try:
            # Créer le modèle s'il n'existe pas
            try:
                self.client.get_registered_model(model_name)
                logger.info(f"📦 Modèle existant trouvé: {model_name}")
            except Exception:
                self.client.create_registered_model(
                    model_name,
                    description=f"Modèle de prédiction météorologique - {training_results['best_model']}"
                )
                logger.info(f"📦 Nouveau modèle créé: {model_name}")
            
            # Trouver le run MLflow du meilleur modèle
            experiment_name = training_results.get('mlflow_experiment', 'Default')
            best_run_id = self._find_best_model_run(experiment_name, training_results['best_model'])
            
            if not best_run_id:
                raise ValueError(f"Run MLflow introuvable pour {training_results['best_model']}")
            
            # Enregistrer la version du modèle
            model_uri = f"runs:/{best_run_id}/{training_results['best_model'].lower()}_model"
            
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=best_run_id,
                description=model_config['description']
            )
            
            # Ajouter des tags métadata
            self._add_model_metadata(model_name, model_version.version, model_config, training_results)
            
            logger.info(f"✅ Version {model_version.version} enregistrée pour {model_name}")
            
            return {
                'model_name': model_name,
                'version': model_version.version,
                'run_id': best_run_id,
                'model_uri': model_uri,
                'registration_timestamp': datetime.now().isoformat(),
                'status': 'registered'
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enregistrement: {e}")
            raise
    
    def _find_best_model_run(self, experiment_name: str, model_type: str) -> Optional[str]:
        """Trouve le run ID du meilleur modèle dans l'expérience"""
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.warning(f"Expérience '{experiment_name}' introuvable")
                return None
            
            # Rechercher les runs de ce type de modèle
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.model_type = '{model_type}'",
                order_by=["metrics.avg_test_rmse ASC"],  # Meilleur RMSE en premier
                max_results=1
            )
            
            if runs:
                best_run = runs[0]
                logger.info(f"🔍 Run du meilleur modèle trouvé: {best_run.info.run_id}")
                return best_run.info.run_id
            else:
                logger.warning(f"Aucun run trouvé pour le modèle {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche du run: {e}")
            return None
    
    def _add_model_metadata(self, model_name: str, version: str, 
                           model_config: Dict[str, Any], training_results: Dict[str, Any]) -> None:
        """Ajoute des métadonnées au modèle enregistré"""
        try:
            # Tags de performance
            metrics = model_config['metrics']
            tags = {
                'rmse': str(round(metrics['avg_test_rmse'], 4)),
                'r2_score': str(round(metrics['avg_test_r2'], 4)),
                'mae': str(round(metrics['avg_test_mae'], 4)),
                'model_type': training_results['best_model'],
                'training_date': datetime.now().strftime('%Y-%m-%d'),
                'deployment_ready': str(model_config['deployment_ready']),
                'data_version': training_results.get('data_preparation', {}).get('dataset', 'unknown')
            }
            
            # Ajouter les tags
            for key, value in tags.items():
                self.client.set_model_version_tag(model_name, version, key, value)
            
            logger.info(f"🏷️ Métadonnées ajoutées au modèle {model_name} v{version}")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de l'ajout des métadonnées: {e}")
    
    def _promote_to_production(self, registration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Promeut le modèle vers la production"""
        logger.info("🚀 PROMOTION VERS LA PRODUCTION")
        
        model_name = registration_result['model_name']
        version = registration_result['version']
        
        try:
            # Archiver l'ancienne version en production (si elle existe)
            current_production_versions = self.client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            
            for old_version in current_production_versions:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=old_version.version,
                    stage="Archived",
                    archive_existing_versions=False
                )
                logger.info(f"📦 Version {old_version.version} archivée")
            
            # Promouvoir la nouvelle version
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=False
            )
            
            logger.info(f"🚀 Version {version} promue en Production")
            
            return {
                'promotion_status': 'success',
                'production_version': version,
                'promotion_timestamp': datetime.now().isoformat(),
                'archived_versions': [v.version for v in current_production_versions]
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la promotion: {e}")
            return {
                'promotion_status': 'failed',
                'error': str(e)
            }
    
    def _save_registration_info(self, registration_result: Dict[str, Any]) -> None:
        """Sauvegarde les informations d'enregistrement"""
        registration_file = self.results_dir / "model_registration.json"
        
        try:
            with open(registration_file, 'w') as f:
                json.dump(registration_result, f, indent=2, default=str)
            logger.info(f"💾 Informations d'enregistrement sauvées: {registration_file}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
    
    def list_registered_models(self) -> Dict[str, Any]:
        """Liste tous les modèles enregistrés"""
        logger.info("📋 MODÈLES ENREGISTRÉS")
        logger.info("=" * 40)
        
        try:
            registered_models = self.client.search_registered_models()
            
            models_info = {}
            
            for model in registered_models:
                model_name = model.name
                versions = self.client.get_latest_versions(model_name)
                
                models_info[model_name] = {
                    'description': model.description,
                    'versions': []
                }
                
                for version in versions:
                    version_info = {
                        'version': version.version,
                        'stage': version.current_stage,
                        'creation_timestamp': version.creation_timestamp,
                        'tags': dict(version.tags) if version.tags else {}
                    }
                    models_info[model_name]['versions'].append(version_info)
                
                logger.info(f"📦 {model_name}: {len(versions)} versions")
                for version in versions:
                    stage_emoji = "🚀" if version.current_stage == "Production" else "🔄"
                    logger.info(f"   {stage_emoji} v{version.version} ({version.current_stage})")
            
            return models_info
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la liste des modèles: {e}")
            return {}

def main():
    """Fonction principale d'enregistrement"""
    try:
        registry = ModelRegistry()
        
        # Enregistrer le meilleur modèle
        registration_result = registry.register_best_model()
        
        # Lister les modèles enregistrés
        models_info = registry.list_registered_models()
        
        print("\n" + "=" * 60)
        print("📦 ENREGISTREMENT DE MODÈLE TERMINÉ")
        print("=" * 60)
        print(f"🏆 Modèle enregistré: {registration_result['model_name']}")
        print(f"📋 Version: {registration_result['version']}")
        print(f"🚀 Statut: {registration_result['status']}")
        
        if registration_result.get('promotion_status') == 'success':
            print(f"🚀 Promotion: Version {registration_result['production_version']} en Production")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'enregistrement: {e}")
        raise

if __name__ == "__main__":
    main()