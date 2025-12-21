# 📊 Diagramme d'Architecture - Modèles ML Climate MLOps

```mermaid
graph TD
    %% Data Sources
    A[📊 Données Marrakech<br/>2018-2023] --> B[🔧 Feature Engineering]
    
    %% Feature Engineering
    B --> C{📈 Features Types}
    C --> C1[⏰ Temporelles<br/>Year, Month, Quarter]
    C --> C2[🔄 Cycliques<br/>sin/cos encoding]
    C --> C3[📉 Lag Features<br/>1,3,7,14,30 jours]
    C --> C4[📊 Moving Averages<br/>3,7,14,30 jours]
    C --> C5[📈 Tendances<br/>Volatilité 7d]
    
    %% Data Processing
    C1 --> D[⚖️ StandardScaler]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    
    D --> E[✂️ Train/Test Split<br/>80% / 20%]
    
    %% Models
    E --> F{🧠 Modèles ML}
    
    %% Linear Regression
    F --> G[📊 Linear Regression<br/>Baseline]
    G --> G1[⚡ Très rapide<br/>🎯 Interprétable<br/>📉 Performance limitée]
    
    %% Random Forest
    F --> H[🌲 Random Forest<br/>Principal]
    H --> H1[🎯 n_estimators: 200<br/>📏 max_depth: 15<br/>🌿 min_samples_split: 5<br/>🍃 min_samples_leaf: 2]
    H --> H2[🔍 Optuna Optimization<br/>50-500 estimators<br/>5-30 depth]
    
    %% Gradient Boosting
    F --> I[📈 Gradient Boosting<br/>Avancé]
    I --> I1[🎯 n_estimators: 150<br/>📚 learning_rate: 0.1<br/>📏 max_depth: 6<br/>🔄 Sequential training]
    
    %% Evaluation
    G1 --> J[📊 Métriques d'Évaluation]
    H2 --> J
    I1 --> J
    
    J --> J1[📏 RMSE<br/>📐 MAE<br/>📊 R²<br/>📈 MAPE]
    
    %% Model Selection
    J1 --> K{🏆 Sélection Automatique<br/>Meilleur RMSE}
    
    %% MLflow Tracking
    K --> L[📈 MLflow Tracking]
    L --> L1[📊 Paramètres<br/>📈 Métriques<br/>🗂️ Artefacts<br/>🏷️ Tags]
    
    %% Model Registry
    L --> M[🗄️ Model Registry]
    M --> M1[🧪 Staging]
    M --> M2[🚀 Production]
    M --> M3[📦 Archive]
    
    %% Prediction Targets
    K --> N[🎯 Variables Cibles]
    N --> N1[🌡️ Temp Max<br/>🌡️ Temp Min<br/>🌡️ Temp Moyenne]
    
    %% Monitoring
    M2 --> O[📊 Monitoring]
    O --> O1[🔍 Data Drift<br/>📉 Performance<br/>🚨 Alertes]
    
    %% Continuous Training
    O1 --> P[🔄 Retraining<br/>Hebdomadaire]
    P --> B
    
    %% Styling
    classDef dataClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef modelClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef mlopsClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef metricClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class A,B,C,C1,C2,C3,C4,C5,D,E dataClass
    class G,H,I,G1,H1,H2,I1 modelClass
    class L,L1,M,M1,M2,M3,O,O1,P mlopsClass
    class J,J1,K,N,N1 metricClass
```

## 🔄 Pipeline de Données Détaillé

```mermaid
flowchart LR
    %% Raw Data
    A1[📊 CSV Marrakech<br/>2018-2023] --> A2[🧹 Nettoyage<br/>Valeurs manquantes<br/>Outliers]
    
    %% Feature Engineering Steps
    A2 --> B1[⏰ Features Temporelles]
    B1 --> B2[Year: 2018-2023<br/>Month: 1-12<br/>Quarter: Q1-Q4<br/>DayOfYear: 1-365<br/>WeekOfYear: 1-53]
    
    A2 --> C1[🔄 Encoding Cyclique]
    C1 --> C2[Month_sin = sin(2π*month/12)<br/>Month_cos = cos(2π*month/12)<br/>DayOfYear_sin<br/>DayOfYear_cos]
    
    A2 --> D1[📉 Lag Features]
    D1 --> D2[Temp_lag_1: J-1<br/>Temp_lag_3: J-3<br/>Temp_lag_7: J-7<br/>Temp_lag_14: J-14<br/>Temp_lag_30: J-30]
    
    A2 --> E1[📊 Moving Averages]
    E1 --> E2[MA_3: moyenne 3 jours<br/>MA_7: moyenne 7 jours<br/>MA_14: moyenne 14 jours<br/>MA_30: moyenne 30 jours]
    
    A2 --> F1[📈 Features Avancées]
    F1 --> F2[Trend_30d: pente 30j<br/>Volatility_7d: std 7j<br/>Diff_1d: différence J vs J-1<br/>Diff_7d: différence hebdo]
    
    %% Concatenation
    B2 --> G[🔗 Concaténation Features]
    C2 --> G
    D2 --> G
    E2 --> G
    F2 --> G
    
    %% Scaling
    G --> H[⚖️ StandardScaler<br/>μ=0, σ=1]
    
    %% Split
    H --> I[✂️ Train/Test Split<br/>Temporal: avant/après 2022]
    
    %% Final datasets
    I --> J1[🏋️ Train Set<br/>2018-2021<br/>~1460 samples]
    I --> J2[🧪 Test Set<br/>2022-2023<br/>~730 samples]
    
    classDef dataClass fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef featureClass fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef processClass fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class A1,A2 dataClass
    class B1,B2,C1,C2,D1,D2,E1,E2,F1,F2 featureClass
    class G,H,I,J1,J2 processClass
```

## 📊 Comparaison des Modèles

| Aspect | Linear Regression | Random Forest | Gradient Boosting |
|--------|------------------|---------------|-------------------|
| **Complexité** | ⭐ Faible | ⭐⭐⭐ Moyenne | ⭐⭐⭐⭐ Élevée |
| **Temps d'entraînement** | ⚡ <1 min | 🕐 2-5 min | 🕘 5-15 min |
| **Interprétabilité** | ⭐⭐⭐⭐⭐ Très haute | ⭐⭐⭐ Moyenne | ⭐⭐ Faible |
| **Performance typique** | ⭐⭐ Baseline | ⭐⭐⭐⭐ Excellente | ⭐⭐⭐⭐⭐ Optimale |
| **Résistance overfitting** | ⭐⭐⭐⭐⭐ Très haute | ⭐⭐⭐⭐ Haute | ⭐⭐⭐ Moyenne |
| **Hyperparamètres** | Aucun | 4 principaux | 6+ critiques |
| **Parallélisation** | ❌ Non | ✅ Oui | ❌ Non |
| **Usage recommandé** | Baseline/Debug | Production | Compétition |

## 🎯 Flux de Sélection de Modèle

```mermaid
flowchart TD
    Start[🚀 Début Entraînement] --> Train[🏋️ Entraîner 3 Modèles]
    
    Train --> LR[📊 Linear Regression<br/>Baseline rapide]
    Train --> RF[🌲 Random Forest<br/>+ Optuna optimization]
    Train --> GB[📈 Gradient Boosting<br/>Configuration standard]
    
    LR --> Eval[📊 Évaluation RMSE]
    RF --> Eval
    GB --> Eval
    
    Eval --> Compare{🏆 Comparaison<br/>RMSE moyen}
    
    Compare --> |RMSE_LR < autres| SelectLR[✅ Sélection LR<br/>Cas rare - données simples]
    Compare --> |RMSE_RF < autres| SelectRF[✅ Sélection RF<br/>Cas typique - équilibré]
    Compare --> |RMSE_GB < autres| SelectGB[✅ Sélection GB<br/>Cas complexe - performance max]
    
    SelectLR --> MLflow[📈 Log MLflow]
    SelectRF --> MLflow
    SelectGB --> MLflow
    
    MLflow --> Registry[🗄️ Model Registry]
    Registry --> Deploy[🚀 Déploiement]
    
    Deploy --> Monitor[📊 Monitoring Production]
    Monitor --> |Dégradation détectée| Retrain[🔄 Retraining Automatique]
    Retrain --> Train
    
    classDef startClass fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef modelClass fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    classDef evalClass fill:#ffecb3,stroke:#f57f17,stroke-width:2px
    classDef deployClass fill:#b3e5fc,stroke:#0288d1,stroke-width:2px
    
    class Start startClass
    class LR,RF,GB,SelectLR,SelectRF,SelectGB modelClass
    class Eval,Compare,MLflow evalClass
    class Registry,Deploy,Monitor,Retrain deployClass
```