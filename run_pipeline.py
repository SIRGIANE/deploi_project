#!/usr/bin/env python3
"""
Script principal pour exécuter le pipeline de données en 3 étapes
1. Téléchargement depuis Kaggle → data/raw/
2. Preprocessing → data/processed/
3. Feature engineering → data/features/
"""

import sys
import argparse
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_pipeline import WeatherDataPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline de données météorologiques en 3 étapes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python run_pipeline.py --all              # Exécuter tout le pipeline
  python run_pipeline.py --step 1           # Télécharger les données brutes
  python run_pipeline.py --step 2           # Preprocessing uniquement
  python run_pipeline.py --step 3           # Créer les features uniquement
  python run_pipeline.py --prepare-ml       # Préparer les données pour ML
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Exécuter tout le pipeline (3 étapes + préparation ML)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3],
        help='Exécuter une étape spécifique (1: download, 2: preprocess, 3: features)'
    )
    
    parser.add_argument(
        '--prepare-ml',
        action='store_true',
        help='Préparer les données pour le machine learning'
    )
    
    args = parser.parse_args()
    
    # Initialisation du pipeline
    pipeline = WeatherDataPipeline()
    
    try:
        if args.all:
            print("\n🚀 Exécution du pipeline complet...")
            results = pipeline.run_full_pipeline()
            
            print("\n" + "=" * 70)
            print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
            print("=" * 70)
            print(f"📂 Fichiers générés:")
            print(f"   1. {results['stats']['raw_file']}")
            print(f"   2. {results['stats']['processed_file']}")
            print(f"   3. {results['stats']['features_file']}")
            print(f"   4. data/features/X_train.npy")
            print(f"   5. data/features/X_test.npy")
            print(f"   6. data/features/y_train.npy")
            print(f"   7. data/features/y_test.npy")
            print(f"\n🎯 Variables cibles: {results['stats']['target']}")
            print(f"📊 Features: {results['stats']['feature_count']}")
            print("=" * 70)
            
        elif args.step == 1:
            print("\n📥 ÉTAPE 1: Téléchargement des données depuis Kaggle...")
            raw_data = pipeline.step1_download_raw_data()
            print(f"✅ Données sauvegardées dans: data/raw/weather_data_raw.csv")
            print(f"📊 Shape: {raw_data.shape}")
            
        elif args.step == 2:
            print("\n🔧 ÉTAPE 2: Preprocessing des données...")
            processed_data = pipeline.step2_preprocess_data()
            print(f"✅ Données sauvegardées dans: data/processed/weather_data_processed.csv")
            print(f"📊 Shape: {processed_data.shape}")
            
        elif args.step == 3:
            print("\n🎨 ÉTAPE 3: Création des features...")
            features_data = pipeline.step3_create_features()
            print(f"✅ Données sauvegardées dans: data/features/weather_data_features.csv")
            print(f"📊 Shape: {features_data.shape}")
            
        elif args.prepare_ml:
            print("\n🤖 Préparation des données pour ML...")
            ml_data = pipeline.prepare_ml_data()
            print(f"✅ Matrices ML sauvegardées dans: data/features/")
            print(f"🚂 Train: {ml_data['X_train'].shape}")
            print(f"🧪 Test: {ml_data['X_test'].shape}")
            print(f"🎯 Cibles: {ml_data['target_names']}")
            
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
