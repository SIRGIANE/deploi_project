#!/usr/bin/env python3
"""
🌦️ Script de collecte quotidienne des données météo
Collecte les données d'aujourd'hui via l'API Open-Meteo et les stocke dans:
- Le fichier CSV cumulatif (data/cumulative_weather_data.csv)
- La base de données PostgreSQL (weather-db)

Usage:
    python collect_today_data.py
    
Pour Docker:
    docker-compose exec airflow-worker python /workspace/collect_today_data.py
"""

import sys
import os
from datetime import datetime

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.marrakech_data_loader import MarrakechWeatherDataLoader

def main():
    print("=" * 70)
    print("🌦️  COLLECTE QUOTIDIENNE DES DONNÉES MÉTÉO - MARRAKECH")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Initialiser le loader
        loader = MarrakechWeatherDataLoader()
        
        # Collecter et stocker les données
        result = loader.collect_and_store_today_data()
        
        if result['success']:
            print("\n" + "=" * 70)
            print("✅ COLLECTE RÉUSSIE!")
            print("=" * 70)
            print(f"📊 Nouvelles données collectées: {result['new_records']} jours")
            print(f"📦 Total de données cumulées: {result['total_records']} lignes")
            print(f"📅 Période couverte: {result['date_range']['start']} → {result['date_range']['end']}")
            print(f"🌡️  Météo d'aujourd'hui:")
            print(f"   - Date: {result['today_weather']['date']}")
            print(f"   - Température moyenne: {result['today_weather']['temperature_mean']:.1f}°C")
            print(f"   - Température max: {result['today_weather']['temperature_max']:.1f}°C")
            print(f"   - Température min: {result['today_weather']['temperature_min']:.1f}°C")
            print(f"   - Précipitations: {result['today_weather']['precipitation']:.1f} mm")
            print(f"   - Vent max: {result['today_weather']['windspeed']:.1f} km/h")
            print(f"💾 Stockage: CSV ✓ | PostgreSQL ✓")
            print(f"⏱️  Collecte effectuée à: {result['collection_time']}")
            
            # Vérifier si retraining nécessaire
            if loader.should_trigger_retraining(threshold_days=7):
                print("\n⚠️  ATTENTION: Retraining recommandé (>7 jours depuis le dernier)")
            else:
                days = loader.get_days_since_last_training()
                print(f"\nℹ️  Prochain retraining dans {7 - days} jours")
            
            return 0
        else:
            print("\n❌ ÉCHEC DE LA COLLECTE")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
