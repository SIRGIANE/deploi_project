#!/usr/bin/env python3
"""
🔍 Script de diagnostic pour la collecte des données météo
Teste la connectivité API, PostgreSQL et vérifie les données d'aujourd'hui
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_connectivity():
    """Teste la connectivité à l'API Open-Meteo"""
    print("\n🌐 TEST CONNECTIVITÉ API OPEN-METEO")
    print("=" * 50)
    
    try:
        import requests
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        
        loader = MarrakechWeatherDataLoader()
        
        # Test simple de l'API
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        params = {
            "latitude": loader.marrakech_lat,
            "longitude": loader.marrakech_lon,
            "start_date": str(yesterday),
            "end_date": str(today),
            "daily": ["temperature_2m_mean", "precipitation_sum"],
            "timezone": "Africa/Casablanca"
        }
        
        print(f"📍 Coordonnées Marrakech: {loader.marrakech_lat}, {loader.marrakech_lon}")
        print(f"📅 Test période: {yesterday} → {today}")
        
        response = requests.get(loader.historical_api_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'daily' in data and data['daily']:
                print("✅ API Open-Meteo: ACCESSIBLE")
                print(f"📊 Données reçues: {len(data['daily']['time'])} jours")
                return True
            else:
                print("❌ API répond mais pas de données quotidiennes")
                return False
        else:
            print(f"❌ Erreur HTTP: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur API: {e}")
        return False

def test_postgresql_connection():
    """Teste la connexion PostgreSQL"""
    print("\n🐘 TEST CONNEXION POSTGRESQL")
    print("=" * 50)
    
    try:
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        
        loader = MarrakechWeatherDataLoader()
        print(f"🔗 String de connexion: {loader._db_info}")
        
        engine = loader._get_db_engine()
        if engine is None:
            print("❌ Impossible de créer le moteur de base de données")
            return False
        
        # Test simple de connexion
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1"))
            test_val = result.scalar()
            
            if test_val == 1:
                print("✅ PostgreSQL: CONNECTÉ")
                
                # Vérifier si la table existe
                table_check = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'weather_data'
                    );
                """))
                table_exists = table_check.scalar()
                
                if table_exists:
                    # Compter les lignes
                    count_result = conn.execute(text("SELECT COUNT(*) FROM weather_data"))
                    row_count = count_result.scalar()
                    print(f"📊 Table 'weather_data': {row_count} lignes")
                    
                    # Date la plus récente
                    latest_result = conn.execute(text("SELECT MAX(time) FROM weather_data"))
                    latest_date = latest_result.scalar()
                    print(f"📅 Dernière date: {latest_date}")
                    
                else:
                    print("⚠️ Table 'weather_data' n'existe pas encore")
                
                return True
            else:
                print("❌ Test de connexion échoué")
                return False
                
    except Exception as e:
        print(f"❌ Erreur PostgreSQL: {e}")
        print("💡 Suggestion: Vérifiez que Docker est lancé (docker-compose up -d)")
        return False

def test_csv_data():
    """Teste les données CSV existantes"""
    print("\n📂 TEST DONNÉES CSV")
    print("=" * 50)
    
    try:
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        
        loader = MarrakechWeatherDataLoader()
        
        # Fichier cumulatif
        cumulative_file = Path(loader.cumulative_data_file)
        if cumulative_file.exists():
            import pandas as pd
            df = pd.read_csv(cumulative_file)
            print(f"✅ Fichier cumulatif: {len(df)} lignes")
            print(f"📅 Période: {df['datetime'].min()} → {df['datetime'].max()}")
        else:
            print("⚠️ Fichier cumulatif n'existe pas")
        
        # Fichier historique
        historical_file = Path(loader.historical_data_file)
        if historical_file.exists():
            import pandas as pd
            df_hist = pd.read_csv(historical_file)
            print(f"✅ Fichier historique: {len(df_hist)} lignes")
            if 'time' in df_hist.columns:
                print(f"📅 Période historique: {df_hist['time'].min()} → {df_hist['time'].max()}")
        else:
            print("❌ Fichier historique manquant")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lecture CSV: {e}")
        return False

def test_today_data_collection():
    """Teste la collecte des données d'aujourd'hui"""
    print("\n📥 TEST COLLECTE DONNÉES AUJOURD'HUI")
    print("=" * 50)
    
    try:
        from src.marrakech_data_loader import MarrakechWeatherDataLoader
        
        loader = MarrakechWeatherDataLoader()
        
        print("🚀 Lancement de la collecte...")
        result = loader.collect_and_store_today_data()
        
        if result['success']:
            print("✅ COLLECTE RÉUSSIE!")
            print(f"📊 Nouvelles données: {result['new_records']} lignes")
            print(f"📦 Total cumulé: {result['total_records']} lignes")
            print(f"🌡️ Température aujourd'hui: {result['today_weather']['temperature_mean']:.1f}°C")
            print(f"🌧️ Précipitations: {result['today_weather']['precipitation']:.1f} mm")
            print(f"💾 Stockage: {result['storage']}")
            return True
        else:
            print("❌ COLLECTE ÉCHOUÉE")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la collecte: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_airflow_environment():
    """Vérifie l'environnement Airflow"""
    print("\n🚁 VÉRIFICATION ENVIRONNEMENT AIRFLOW")
    print("=" * 50)
    
    # Variables d'environnement importantes
    env_vars = [
        'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_HOST', 
        'POSTGRES_PORT', 'POSTGRES_DB', 'AIRFLOW_HOME'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'NON DÉFINIE')
        print(f"🔧 {var}: {value}")
    
    # Vérifier si on est dans Docker
    is_docker = os.path.exists('/.dockerenv')
    print(f"🐳 Dans Docker: {'OUI' if is_docker else 'NON'}")
    
    # Chemin Python
    print(f"🐍 Python: {sys.executable}")
    print(f"📁 Répertoire de travail: {os.getcwd()}")

def main():
    """Fonction principale de diagnostic"""
    print("🔍" * 30)
    print("    DIAGNOSTIC COLLECTE MÉTÉO - MARRAKECH")
    print("🔍" * 30)
    print(f"📅 Date du diagnostic: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Tests séquentiels
    tests = [
        ("Environnement Airflow", check_airflow_environment),
        ("Connectivité API", test_api_connectivity),
        ("Connexion PostgreSQL", test_postgresql_connection),
        ("Données CSV", test_csv_data),
        ("Collecte aujourd'hui", test_today_data_collection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ ERREUR CRITIQUE dans {test_name}: {e}")
            results[test_name] = False
    
    # Résumé final
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ DU DIAGNOSTIC")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "🔍" * 30)
    if all_passed:
        print("🎉 DIAGNOSTIC COMPLET: TOUS LES TESTS PASSENT!")
        print("💡 Votre pipeline de collecte devrait fonctionner correctement.")
    else:
        print("⚠️ PROBLÈMES DÉTECTÉS!")
        print("💡 Corrigez les erreurs ci-dessus avant de relancer le DAG.")
    print("🔍" * 30)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())