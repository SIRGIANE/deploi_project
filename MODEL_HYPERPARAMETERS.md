# 🔧 HYPERPARAMÈTRES DÉTAILLÉS - CLIMATE MLOPS

## 📋 **RÉSUMÉ EXÉCUTIF**

**Projet**: Climate MLOps - Prédiction Météo Marrakech  
**Dataset**: Données historiques 2018-2023 (2193 jours)  
**Modèle sélectionné**: **LinearRegression** (RMSE: 0.966°C, R²: 95.9%)  
**Date d'entraînement**: 2025-12-13 18:56:27  

---

## 🏆 **1. LINEAR REGRESSION (MODÈLE SÉLECTIONNÉ)**

### Hyperparamètres
```python
{
    "fit_intercept": True,      # Calcul de l'ordonnée à l'origine
    "copy_X": True,             # Copie des données d'entrée
    "n_jobs": None,             # Pas de parallélisation
    "positive": False           # Coefficients peuvent être négatifs
}
```

### Détail par cible
- `temperature_2m_mean`: RMSE=0.989°C, R²=97.9%
- `temperature_2m_min`: RMSE=1.908°C, R²=89.8% 
- `temperature_2m_max`: RMSE=0.000°C, R²=100% (relation parfaite)

---

## 🌲 **2. RANDOM FOREST (OPTIMISÉ OPTUNA)**

### Hyperparamètres optimisés
```python
{
    "n_estimators": 378,        # Nombre d'arbres dans la forêt
    "max_depth": 25,            # Profondeur maximale des arbres
    "min_samples_split": 7,     # Échantillons minimum pour diviser un nœud
    "min_samples_leaf": 1,      # Échantillons minimum dans une feuille
    "random_state": 42,         # Graine pour reproductibilité
    "max_features": "sqrt",     # √(n_features) par arbre (défaut)
    "bootstrap": True,          # Échantillonnage avec remise
    "oob_score": False,         # Score out-of-bag désactivé
    "n_jobs": -1,              # Utilisation de tous les CPU
    "criterion": "squared_error" # Critère de division
}
```

### Espace de recherche Optuna
```python
OPTUNA_SEARCH_SPACE = {
    "n_estimators": (50, 500),     # 50 à 500 arbres
    "max_depth": (5, 30),          # Profondeur 5 à 30
    "min_samples_split": (2, 20),  # 2 à 20 échantillons
    "min_samples_leaf": (1, 10)    # 1 à 10 échantillons par feuille
}
```

### Configuration Optuna
- **Méthode**: TPE (Tree-structured Parzen Estimator)
- **Nombre d'essais**: 20 trials
- **Métrique d'optimisation**: RMSE (minimisation)
- **Timeout**: Aucun



---

## 📈 **3. GRADIENT BOOSTING**

### Hyperparamètres
```python
{
    "n_estimators": 150,            # Nombre d'estimateurs de boosting
    "learning_rate": 0.1,           # Taux d'apprentissage (shrinkage)
    "max_depth": 6,                 # Profondeur maximale des arbres
    "min_samples_split": 2,         # Échantillons minimum pour diviser
    "min_samples_leaf": 1,          # Échantillons minimum par feuille
    "subsample": 1.0,               # Fraction d'échantillons utilisés
    "max_features": None,           # Toutes les features utilisées
    "random_state": 42,             # Graine aléatoire
    "loss": "squared_error",        # Fonction de perte
    "criterion": "friedman_mse",    # Critère de qualité de division
    "init": None,                   # Estimateur initial par défaut
    "alpha": 0.9,                   # Quantile pour perte Huber/quantile
    "verbose": 0,                   # Pas d'affichage du progrès
    "warm_start": False,            # Pas de réutilisation de solution
    "validation_fraction": 0.1,     # Fraction pour validation early stopping
    "n_iter_no_change": None,       # Pas d'early stopping
    "tol": 1e-4                     # Tolérance pour early stopping
}
```

### Multi-Output Configuration
```python
# Encapsulé dans MultiOutputRegressor pour 3 cibles
MultiOutputRegressor(
    estimator=GradientBoostingRegressor(**params),
    n_jobs=None
)
```


---

## 📊 **CONFIGURATION DES DONNÉES**

### Dataset
```python
{
    "source": "Marrakech Weather Dataset 2018-2023",
    "total_samples": 2193,
    "train_samples": 1754,         # 80% des données
    "test_samples": 439,           # 20% des données
    "train_test_split": 0.8,
    "split_method": "temporal"      # Division chronologique
}
```

### Variables cibles
```python
TARGET_VARIABLES = [
    "temperature_2m_mean",   # Température moyenne (°C)
    "temperature_2m_min",    # Température minimale (°C) 
    "temperature_2m_max"     # Température maximale (°C)
]
```

### Features (22 sélectionnées)
```python
SELECTED_FEATURES = [
    # Features temporelles
    "Year", "Month", "Quarter", "DayOfYear", "WeekOfYear",
    
    # Features cycliques
    "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos",
    
    # Features de lag (décalage temporel)
    "Temp_lag_1", "Temp_lag_3", "Temp_lag_7", "Temp_lag_14", "Temp_lag_30",
    
    # Moyennes mobiles
    "Temp_ma_3", "Temp_ma_7", "Temp_ma_14", "Temp_ma_30",
    
    # Features de tendance et volatilité
    "Temp_trend_30d", "Temp_volatility_7d",
    
    # Différences temporelles
    "Temp_diff_1d", "Temp_diff_7d"
]
```

### Configuration Feature Engineering
```python
FEATURE_CONFIG = {
    "LAG_PERIODS": [1, 3, 7, 14, 30],        # Décalages en jours
    "MOVING_AVERAGE_WINDOWS": [3, 7, 14, 30], # Fenêtres moyennes mobiles
    "TREND_WINDOW": 30,                       # Fenêtre calcul tendance
    "VOLATILITY_WINDOW": 7,                   # Fenêtre calcul volatilité
    "SCALING_METHOD": "StandardScaler"         # Normalisation Z-score
}
```

---

## ⚙️ **SÉLECTION DE MODÈLE AVANCÉE**

### Critères de sélection
```python
MODEL_SELECTION_WEIGHTS = {
    "rmse": 0.4,    # 40% - Erreur quadratique (métrique principale)
    "r2": 0.3,      # 30% - Qualité d'ajustement
    "mae": 0.2,     # 20% - Erreur absolue moyenne
    "time": 0.1     # 10% - Vitesse d'entraînement
}
```

### Scores composites obtenus
```python
COMPOSITE_SCORES = {
    "LinearRegression": 100.0,    # 🏆 GAGNANT
    "GradientBoosting": 54.5,
    "RandomForest": 0.0           # Performance la plus faible
}
```

### Critères de déploiement
```python
DEPLOYMENT_CRITERIA = {
    "min_r2_threshold": 0.7,           # R² minimum: 70%
    "min_rmse_improvement": 0.05,      # Amélioration RMSE: 5%
    "min_r2_improvement": 0.02,        # Amélioration R²: 2%
    "min_data_points": 100,            # Points test minimum
    "max_training_time": 3600,         # Temps max: 1h
    "require_all_tests": True          # Tous critères requis
}
```

### Résultat déploiement
```python
DEPLOYMENT_DECISION = {
    "should_deploy": True,              # ✅ DÉPLOYER
    "model_name": "LinearRegression",
    "reasons": [
        "✅ R² satisfaisant: 95.9% >= 70%",
        "✅ Premier modèle - pas de précédent",
        "✅ Données suffisantes: 439 >= 100"
    ]
}
```

---

## 🚀 **CONFIGURATION MLFLOW**

### Tracking
```python
MLFLOW_CONFIG = {
    "tracking_uri": "file:./mlruns",
    "experiment_name": "training_20251213",
    "backend_store_uri": "./mlruns/mlflow.db",
    "default_artifact_root": "./mlruns"
}
```

### Métriques trackées
```python
TRACKED_METRICS = [
    "train_rmse", "test_rmse",         # Erreur quadratique
    "train_mae", "test_mae",           # Erreur absolue
    "train_r2", "test_r2",             # Coefficient détermination
    "training_time",                    # Temps d'entraînement
    "composite_score"                   # Score de sélection
]
```

---

## 📈 **PERFORMANCES COMPARATIVES**

| Modèle | RMSE Test | R² Test | MAE Test | Temps (s) | Score |
|--------|-----------|---------|----------|-----------|-------|
| **LinearRegression** | **0.966** | **95.9%** | **0.743** | **8.3** | **100** |
| GradientBoosting | 1.174 | 95.5% | 0.918 | 12.3 | 54.5 |
| RandomForest | 1.339 | 94.8% | 1.051 | 104.0 | 0.0 |

---

## 🔧 **ENVIRONNEMENT TECHNIQUE**

### Versions des librairies
```python
DEPENDENCIES = {
    "scikit-learn": ">=1.3.0",
    "optuna": ">=3.0.0",
    "mlflow": ">=2.0.0",
    "pandas": ">=2.0.0",
    "numpy": ">=1.24.0"
}
```

### Configuration système
```python
SYSTEM_CONFIG = {
    "python_version": "3.9+",
    "cpu_cores_used": "all (-1)",
    "memory_usage": "optimized",
    "random_seed": 42
}
```

---

## 💡 **CONCLUSIONS**

1. **LinearRegression** domine grâce à sa **simplicité** et **performance exceptionnelle**
2. La relation température à Marrakech est **largement linéaire**
3. Les modèles complexes (RF, GB) souffrent de **surapprentissage**
4. **Temps d'entraînement** 12x plus rapide pour LinearRegression
5. **Recommandation**: Déployer LinearRegression en production

---

*Généré automatiquement le 2025-12-13 par Climate MLOps Pipeline*