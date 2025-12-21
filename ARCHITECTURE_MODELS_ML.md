# 🏗️ Architecture des Modèles Machine Learning - Climate MLOps

## 📋 Vue d'ensemble du système

Ce document décrit l'architecture des modèles de machine learning utilisés dans le projet Climate MLOps pour la prédiction météorologique de Marrakech.

### 🎯 Objectif
Prédire les variables météorologiques (température max, min, moyenne) basées sur les données historiques de 2018-2023.

---

## 🧠 Modèles Implémentés

### 1. 📊 **Linear Regression (Baseline)**
**Type :** Modèle de référence (baseline)  
**Algorithme :** Régression linéaire multiple  
**Classe utilisée :** `sklearn.linear_model.LinearRegression`

#### Caractéristiques :
- **Complexité :** Faible
- **Interprétabilité :** Très élevée
- **Temps d'entraînement :** Très rapide
- **Capacité de généralisation :** Limitée pour les relations non-linéaires

#### Paramètres :
```python
# Aucun hyperparamètre à ajuster
model = LinearRegression()
```

#### Usage :
- Modèle de référence pour comparaison
- Établissement de la performance minimale acceptable
- Validation que les features ont un signal prédictif

---

### 2. 🌲 **Random Forest Regressor (Modèle Principal)**
**Type :** Ensemble de modèles - Bagging  
**Algorithme :** Forêt aléatoire de régresseurs  
**Classe utilisée :** `sklearn.ensemble.RandomForestRegressor`

#### Caractéristiques :
- **Complexité :** Moyenne à élevée
- **Interprétabilité :** Moyenne (feature importance disponible)
- **Temps d'entraînement :** Modéré
- **Résistance au surapprentissage :** Élevée
- **Gestion des valeurs manquantes :** Naturelle
- **Parallélisation :** Oui (n_jobs=-1)

#### Paramètres par défaut :
```python
DEFAULT_RF_PARAMS = {
    'n_estimators': 200,        # Nombre d'arbres dans la forêt
    'max_depth': 15,            # Profondeur maximale des arbres
    'min_samples_split': 5,     # Échantillons min pour diviser un nœud
    'min_samples_leaf': 2,      # Échantillons min dans une feuille
    'random_state': 42          # Reproductibilité
}
```

#### Optimisation Optuna :
```python
OPTUNA_RF_SEARCH_SPACE = {
    'n_estimators': (50, 500),      # Plage d'optimisation
    'max_depth': (5, 30),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}
```

#### Avantages :
- ✅ Excellent équilibre performance/robustesse
- ✅ Gère naturellement les interactions entre features
- ✅ Fournit l'importance des variables
- ✅ Résistant aux outliers
- ✅ Peu sensible aux hyperparamètres

#### Cas d'usage :
- Modèle principal de production
- Prédictions météorologiques quotidiennes
- Analyse d'importance des features climatiques

---

### 3. 📈 **Gradient Boosting Regressor**
**Type :** Ensemble de modèles - Boosting  
**Algorithme :** Gradient Boosting séquentiel  
**Classe utilisée :** `sklearn.ensemble.GradientBoostingRegressor` avec `MultiOutputRegressor`

#### Caractéristiques :
- **Complexité :** Élevée
- **Interprétabilité :** Faible à moyenne
- **Temps d'entraînement :** Plus lent (séquentiel)
- **Performance :** Potentiellement supérieure avec bon tuning
- **Sensibilité au surapprentissage :** Modérée

#### Paramètres par défaut :
```python
GB_PARAMS = {
    'n_estimators': 150,        # Nombre d'arbres boost
    'learning_rate': 0.1,       # Taux d'apprentissage
    'max_depth': 6,             # Profondeur des arbres faibles
    'random_state': 42
}
```

#### Avantages :
- ✅ Apprentissage séquentiel des erreurs
- ✅ Souvent performance supérieure
- ✅ Contrôle fin via learning_rate
- ✅ Gestion des patterns complexes

#### Inconvénients :
- ❌ Plus sensible au surapprentissage
- ❌ Temps d'entraînement plus long
- ❌ Plus de hyperparamètres à ajuster

---

## 🏗️ Architecture du Pipeline ML

### Pipeline de Données
```
Raw Data → Feature Engineering → Scaling → Train/Test Split → Models
```

#### Features Engineering :
```python
FEATURE_COLUMNS = [
    # Temporelles
    'Year', 'Month', 'Quarter', 'DayOfYear', 'WeekOfYear',
    
    # Cycliques (trigonométriques)
    'Month_sin', 'Month_cos', 
    'DayOfYear_sin', 'DayOfYear_cos',
    
    # Lag features (valeurs passées)
    'Temp_lag_1', 'Temp_lag_3', 'Temp_lag_7', 
    'Temp_lag_14', 'Temp_lag_30',
    
    # Moving averages
    'Temp_ma_3', 'Temp_ma_7', 'Temp_ma_14', 'Temp_ma_30',
    
    # Tendances et volatilité
    'Temp_trend_30d', 'Temp_volatility_7d',
    
    # Différences
    'Temp_diff_1d', 'Temp_diff_7d'
]
```

#### Variables Cibles :
```python
TARGET_VARIABLES = [
    'temperature_2m_max (°C)',
    'temperature_2m_min (°C)', 
    'temperature_2m_mean (°C)'
]
```

---

## 📊 Métriques d'Évaluation

### Métriques Calculées :
```python
EVALUATION_METRICS = [
    'rmse',           # Root Mean Square Error
    'mae',            # Mean Absolute Error  
    'r2_score',       # Coefficient de détermination
    'mape'            # Mean Absolute Percentage Error
]
```

### Calcul Multi-target :
- Métriques individuelles par variable cible
- Moyennes globales pour comparaison des modèles
- Métriques train/test pour détecter le surapprentissage

---

## 🔧 Configuration et Optimisation

### Optimisation Hyperparamètres (Optuna) :
```python
HYPERPARAMETER_OPTIMIZATION = {
    'enabled': True,
    'method': 'optuna',
    'trials': 100,
    'timeout_seconds': 7200,  # 2 heures max
    'objective': 'minimize_rmse'
}
```

### Validation Croisée :
```python
CROSS_VALIDATION = {
    'folds': 5,
    'strategy': 'time_series_split',  # Respecte l'ordre temporel
    'test_size': 0.2,
    'validation_size': 0.1
}
```

---

## 🚀 Stratégie de Déploiement

### Sélection du Modèle :
1. **Entraînement** des 3 modèles en parallèle
2. **Comparaison** basée sur RMSE moyen
3. **Sélection automatique** du meilleur modèle
4. **Sauvegarde** avec versioning MLflow

### Critères de Promotion :
```python
MODEL_PROMOTION_CRITERIA = {
    'min_rmse_improvement': 0.05,       # 5% d'amélioration min
    'min_r2_improvement': 0.02,         # 2% d'amélioration R²
    'min_data_points': 100,             # Données d'évaluation min
    'max_training_time_seconds': 3600,  # Temps max acceptable
    'require_positive_tests': True      # Tous tests passent
}
```

---

## 📈 Monitoring et MLOps

### Tracking MLflow :
- **Paramètres** : Tous hyperparamètres
- **Métriques** : RMSE, MAE, R² par cible
- **Artefacts** : Modèles sérialisés, scalers, pipelines
- **Tags** : Version, environnement, dataset

### Continuous Training :
```python
RETRAINING_CONFIG = {
    'enabled': True,
    'interval_days': 7,           # Hebdomadaire
    'new_data_buffer_size': 7,    # Attendre 7 jours de nouvelles données
    'performance_threshold': 0.05  # Seuil de dégradation
}
```

### Data Drift Detection :
```python
DRIFT_DETECTION = {
    'enabled': True,
    'method': 'statistical',      # KS test, Chi²
    'threshold': 0.3,
    'window_size': 30,           # 30 jours
    'min_samples': 100
}
```

---

## 💾 Persistence et Versioning

### Sauvegarde Modèles :
```
models/
├── rf_model.pkl              # Random Forest sérialisé
├── scaler.pkl               # StandardScaler
├── data_pipeline.joblib     # Pipeline complet
└── registry/
    ├── staging/             # Modèles en validation
    ├── production/          # Modèle actuel en prod
    └── archive/            # Versions archivées
```

### Métadonnées :
```json
{
    "model_type": "RandomForest",
    "version": "v2.1.0",
    "training_date": "2024-12-13T10:30:00",
    "performance": {
        "avg_test_rmse": 2.45,
        "avg_test_r2": 0.87,
        "avg_test_mae": 1.92
    },
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 15
    },
    "features": ["Year", "Month", "Temp_lag_1", ...],
    "targets": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]
}
```

---

## 🎯 Recommandations d'Usage

### Pour la Production :
1. **Utiliser Random Forest** comme modèle principal
2. **Gradient Boosting** pour cas complexes/saisonniers
3. **Linear Regression** comme fallback rapide

### Pour l'Amélioration :
1. Ajouter des features météorologiques externes
2. Implémenter des modèles deep learning (LSTM)
3. Enrichir avec données satellites
4. A/B testing entre modèles

### Pour le Monitoring :
1. Surveiller la dérive des données d'entrée
2. Tracker les performances en temps réel  
3. Alertes automatiques si dégradation
4. Retraining déclenché par seuils

---

*📅 Document mis à jour le : 13 décembre 2024*  
*🔄 Version : 1.0*  
*👨‍💻 Généré automatiquement par Climate MLOps*