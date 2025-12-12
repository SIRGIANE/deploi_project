"""
Script de test rapide pour entraîner un modèle avec le dataset Marrakech
"""
import sys
from pathlib import Path

# Ajout du dossier src au path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline import WeatherDataPipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib

def train_quick_model():
    """Entraîne rapidement un modèle avec les données Marrakech"""
    
    print("=" * 70)
    print("🚀 ENTRAÎNEMENT RAPIDE DU MODÈLE MÉTÉO MARRAKECH")
    print("=" * 70)
    
    # 1. Chargement du pipeline
    pipeline = WeatherDataPipeline()
    
    # 2. Vérification si les données sont déjà traitées
    features_file = Path("data/features/weather_data_features.csv")
    if features_file.exists():
        print("✅ Données déjà traitées, chargement...")
        ml_data = pipeline.prepare_ml_data()
    else:
        print("⚠️ Données non traitées, exécution du pipeline complet...")
        results = pipeline.run_full_pipeline()
        ml_data = results['ml_data']
    
    # 3. Extraction des données
    X_train = ml_data['X_train']
    X_test = ml_data['X_test']
    y_train = ml_data['y_train']
    y_test = ml_data['y_test']
    target_names = ml_data['target_names']
    
    print(f"\n📊 Forme des données:")
    print(f"   🚂 X_train: {X_train.shape}")
    print(f"   🧪 X_test: {X_test.shape}")
    print(f"   🎯 Nombre de cibles: {len(target_names)}")
    
    # 4. Entraînement d'un modèle pour chaque cible
    models = {}
    results = {}
    
    for idx, target in enumerate(target_names):
        print(f"\n{'=' * 70}")
        print(f"🎯 Entraînement pour: {target}")
        print(f"{'=' * 70}")
        
        # Sélection de la cible
        y_train_target = y_train[:, idx] if y_train.ndim > 1 else y_train
        y_test_target = y_test[:, idx] if y_test.ndim > 1 else y_test
        
        # Entraînement
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        print("🔄 Entraînement en cours...")
        model.fit(X_train, y_train_target)
        
        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métriques
        train_rmse = np.sqrt(mean_squared_error(y_train_target, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test_target, y_pred_test))
        train_r2 = r2_score(y_train_target, y_pred_train)
        test_r2 = r2_score(y_test_target, y_pred_test)
        test_mae = mean_absolute_error(y_test_target, y_pred_test)
        
        print(f"\n📈 Résultats:")
        print(f"   🚂 Train RMSE: {train_rmse:.4f}")
        print(f"   🧪 Test RMSE:  {test_rmse:.4f}")
        print(f"   🚂 Train R²:   {train_r2:.4f}")
        print(f"   🧪 Test R²:    {test_r2:.4f}")
        print(f"   📊 Test MAE:   {test_mae:.4f}")
        
        # Sauvegarde
        models[target] = model
        results[target] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae
        }
    
    # 5. Sauvegarde du meilleur modèle (première cible principale)
    main_target = target_names[0]
    main_model = models[main_target]
    
    model_path = Path("models/rf_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(main_model, model_path)
    print(f"\n💾 Modèle principal sauvegardé: {model_path} (cible: {main_target})")
    
    # 6. Résumé global
    print(f"\n{'=' * 70}")
    print("📊 RÉSUMÉ GLOBAL")
    print(f"{'=' * 70}")
    for target, metrics in results.items():
        print(f"\n🎯 {target}:")
        print(f"   Test R²: {metrics['test_r2']:.4f} | Test RMSE: {metrics['test_rmse']:.4f}")
    
    print(f"\n✅ Entraînement terminé avec succès!")
    return models, results

if __name__ == "__main__":
    train_quick_model()
