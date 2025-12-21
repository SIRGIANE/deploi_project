#!/usr/bin/env python
"""
Point d'entrée principal pour lancer l'API Climate MLOps
Résout les problèmes d'imports relatifs
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire src au chemin Python
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    import uvicorn
    from src.api import app
    from src.config import Config
    import logging
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Démarrage de l'API FastAPI Climate Prediction")
    logger.info(f"📍 Adresse: http://{Config.API_HOST}:{Config.API_PORT}")
    logger.info(f"📚 Documentation: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    logger.info(f"🌐 Interface Web: http://{Config.API_HOST}:{Config.API_PORT}/web")
    logger.info(f"📊 Dashboard: http://{Config.API_HOST}:{Config.API_PORT}/dashboard")
    
    uvicorn.run(
        "src.api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level=Config.LOG_LEVEL.lower()
    )
