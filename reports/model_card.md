# Model Card - Climate MLOps

## Informations générales
- **Modèle**: LinearRegression
- **Version**: 2025.12.21
- **Date d'évaluation**: 2025-12-21T11:08:51.792765
- **Statut**: ✅ Prêt pour déploiement

## Performances
- **RMSE Test**: 0.9836°C
- **R² Test**: 0.9581 (A+ (Excellent))
- **MAE Test**: 0.7579°C

## Dataset
- **Source**: Kaggle Weather Dataset
- **Échantillons total**: 2201
- **Variables cibles**: temperature_2m_mean, temperature_2m_min, temperature_2m_max
- **Features**: 22

## Comparaison des modèles
- 🏆 **LinearRegression**: RMSE=0.9836, R²=0.9581 (A+ (Excellent))
-    **GradientBoosting**: RMSE=1.1967, R²=0.9536 (A+ (Excellent))
-    **RandomForest**: RMSE=1.3463, R²=0.948 (A (Très bon))

## Recommandations
- 🏆 Excellentes performances - Modèle prêt pour la production
- 🚀 Déploiement automatique approuvé
- 💡 Relation linéaire détectée - Modèle simple mais efficace

## Métriques techniques
- **URI MLflow**: file:./mlruns
- **Expérience**: training_20251221
- **Méthode de sélection**: advanced_composite_score

*Généré automatiquement par Climate MLOps Pipeline*
