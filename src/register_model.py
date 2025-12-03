"""
Script d'enregistrement des modèles dans MLflow Model Registry
Permet le versioning et le déploiement des modèles
"""

import os
import sys
import logging
from datetime import datetime
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Gestionnaire du registre de modèles MLflow"""
    
    def __init__(self, mlflow_uri="http://localhost:5050"):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.experiment_name = "Climate_Temperature_Prediction"
        
    def register_sklearn_model(self, model, X_sample, y_sample, model_name="RandomForest_Climate", 
                               description="Random Forest model for climate temperature prediction"):
        """Enregistrement d'un modèle sklearn"""
        logger.info(f"📝 Enregistrement du modèle: {model_name}")
        
        try:
            # Inférence de la signature du modèle
            signature = infer_signature(X_sample, model.predict(X_sample))
            
            # Log du modèle dans MLflow
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name,
                signature=signature,
                description=description,
                metadata={
                    "framework": "scikit-learn",
                    "task": "regression",
                    "target": "LandAverageTemperature",
                    "registered_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Modèle enregistré: {model_name}")
            logger.info(f"   URI: {model_info.model_uri}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enregistrement: {e}")
            raise
    
    def register_tensorflow_model(self, model, X_sample, model_name="LSTM_Climate",
                                  description="LSTM model for climate temperature prediction"):
        """Enregistrement d'un modèle TensorFlow/Keras"""
        logger.info(f"📝 Enregistrement du modèle: {model_name}")
        
        try:
            # Inférence de la signature
            y_sample = model.predict(X_sample[:5])
            signature = infer_signature(X_sample[:5], y_sample)
            
            # Log du modèle dans MLflow
            model_info = mlflow.tensorflow.log_model(
                model=model,
                artifact_path="model",
                registered_model_name=model_name,
                signature=signature,
                description=description,
                metadata={
                    "framework": "tensorflow",
                    "task": "regression",
                    "target": "LandAverageTemperature",
                    "registered_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Modèle enregistré: {model_name}")
            logger.info(f"   URI: {model_info.model_uri}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enregistrement: {e}")
            raise
    
    def promote_model_to_production(self, model_name, version):
        """Promotion d'une version de modèle en production"""
        logger.info(f"🚀 Promotion du modèle {model_name} (v{version}) en production")
        
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Mise à jour du stage du modèle
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            
            logger.info(f"✅ Modèle promu en production")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la promotion: {e}")
            raise
    
    def list_registered_models(self):
        """Liste tous les modèles enregistrés"""
        logger.info("📋 Liste des modèles enregistrés:")
        
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.list_registered_models()
            
            for model in models:
                logger.info(f"  - {model.name} (versions: {len(model.latest_versions)})")
                for version in model.latest_versions:
                    logger.info(f"      v{version.version}: {version.current_stage}")
            
            return models
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération: {e}")
            raise
    
    def get_production_model(self, model_name):
        """Récupération du modèle en production"""
        logger.info(f"📥 Récupération du modèle en production: {model_name}")
        
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Récupération des versions du modèle
            versions = client.get_latest_versions(model_name, stages=["Production"])
            
            if not versions:
                logger.warning(f"⚠️ Aucun modèle en production pour {model_name}")
                return None
            
            # Récupération du modèle
            version = versions[0]
            model_uri = f"models:/{model_name}/Production"
            
            # Détermination du type de modèle et chargement
            if "sklearn" in version.source:
                model = mlflow.sklearn.load_model(model_uri)
            elif "tensorflow" in version.source:
                model = mlflow.tensorflow.load_model(model_uri)
            else:
                raise ValueError(f"Type de modèle non supporté: {version.source}")
            
            logger.info(f"✅ Modèle chargé: {model_name} v{version.version}")
            
            return model, version
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def create_model_card(self, model_name, version, metrics, hyperparameters, 
                          output_path="reports/model_card.md"):
        """Création d'une carte de modèle (Model Card)"""
        logger.info(f"📄 Création de la carte du modèle: {model_name}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        model_card = f"""
# Model Card: {model_name}

## Overview
- **Model Name:** {model_name}
- **Version:** {version}
- **Task:** Temperature Prediction
- **Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Details
### Model Architecture
- **Framework:** Scikit-learn / TensorFlow
- **Type:** Regression
- **Input Features:** 20 features climatiques avancées
- **Output:** Température moyenne mensuelle (°C)

### Hyperparameters
```json
{json.dumps(hyperparameters, indent=2)}
```

## Performance Metrics
### Test Metrics
| Metric | Value |
|--------|-------|
| RMSE | {metrics.get('test_rmse', 'N/A'):.4f} |
| MAE | {metrics.get('test_mae', 'N/A'):.4f} |
| R² | {metrics.get('test_r2', 'N/A'):.4f} |
| MAPE | {metrics.get('test_mape', 'N/A'):.2f}% |

### Train Metrics
| Metric | Value |
|--------|-------|
| RMSE | {metrics.get('train_rmse', 'N/A'):.4f} |
| MAE | {metrics.get('train_mae', 'N/A'):.4f} |
| R² | {metrics.get('train_r2', 'N/A'):.4f} |

## Data
### Input Features (20)
1. Year - Année
2. Month - Mois
3. Quarter - Trimestre
4. DayOfYear - Jour de l'année
5. WeekOfYear - Semaine de l'année
6-7. Month_sin, Month_cos - Features cycliques du mois
8-9. DayOfYear_sin, DayOfYear_cos - Features cycliques du jour
10-13. Temp_lag_1/3/6/12 - Valeurs décalées
14-16. Temp_ma_3/6/12 - Moyennes mobiles
17. Temp_trend_12m - Tendance sur 12 mois
18. Temp_volatility_6m - Volatilité sur 6 mois
19-20. Temp_diff_1m/12m - Différences temporelles

### Training Data
- **Source:** Berkeley Earth Climate Change Dataset
- **Period:** 1750-2010
- **Size:** {metrics.get('train_size', 'N/A')} samples
- **Features:** Monthly temperature data

### Test Data
- **Period:** 2010-2015
- **Size:** {metrics.get('test_size', 'N/A')} samples

## Limitations and Considerations
1. **Temporal Dependency:** Modèle basé sur des données historiques
2. **Climate Change:** Données ne reflètent que les tendances passées
3. **Geographic Scope:** Données globales moyennes
4. **Uncertainty:** Peut ne pas capturer les événements extrêmes

## Recommendations
- Utiliser pour des prédictions à court terme (< 5 ans)
- Re-entraîner tous les 6 mois avec nouvelles données
- Combiner avec d'autres modèles pour la robustesse
- Monitorer la dérive des données en production

## Contact
- **Maintainer:** ML Team
- **Updated:** {datetime.now().strftime('%Y-%m-%d')}
"""
        
        with open(output_path, 'w') as f:
            f.write(model_card)
        
        logger.info(f"✅ Carte du modèle sauvegardée: {output_path}")
    
    def generate_registry_report(self, output_path="reports/model_registry_report.html"):
        """Génération d'un rapport du registre de modèles"""
        logger.info("📊 Génération du rapport du registre")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.list_registered_models()
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLflow Model Registry Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                    h1 {{ margin: 0; }}
                    .model-card {{ background: white; margin: 20px 0; padding: 20px; border-left: 4px solid #3498db; }}
                    .version-badge {{ display: inline-block; padding: 5px 10px; margin: 5px; border-radius: 3px; }}
                    .prod {{ background-color: #27ae60; color: white; }}
                    .staging {{ background-color: #f39c12; color: white; }}
                    .archived {{ background-color: #95a5a6; color: white; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #34495e; color: white; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 MLflow Model Registry Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h2>📋 Registered Models ({len(models)})</h2>
            """
            
            for model in models:
                html_content += f"""
                <div class="model-card">
                    <h3>{model.name}</h3>
                    <p><strong>Total Versions:</strong> {len(model.latest_versions)}</p>
                    <h4>Versions:</h4>
                    <table>
                        <tr>
                            <th>Version</th>
                            <th>Stage</th>
                            <th>Created</th>
                            <th>Updated</th>
                        </tr>
                """
                
                for version in model.latest_versions:
                    stage_badge = f'<span class="version-badge {version.current_stage.lower()}">{version.current_stage}</span>'
                    html_content += f"""
                        <tr>
                            <td>v{version.version}</td>
                            <td>{stage_badge}</td>
                            <td>{version.creation_timestamp}</td>
                            <td>{version.last_updated_timestamp}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"✅ Rapport généré: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération du rapport: {e}")

def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage du gestionnaire de registre")
    
    registry = ModelRegistry()
    
    # Exemple d'utilisation
    # registry.list_registered_models()
    # model, version = registry.get_production_model("RandomForest_Climate")
    # registry.generate_registry_report()
    
    logger.info("✅ Gestionnaire prêt")

if __name__ == "__main__":
    main()
