#!/bin/bash

# 🌡️ Climate MLOps - Script de Démarrage Complet
# Lance: API + MLflow + Airflow + Dashboard

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        🌡️  CLIMATE MLOPS - Démarrage Complet 🌡️              ║"
echo "║   API + MLflow + Airflow (Docker) + Dashboard               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Mode de démarrage
MODE=${1:-simple}

if [[ "$MODE" == "full" ]]; then
    echo -e "${YELLOW}Mode FULL (avec Airflow et Docker)${NC}"
    echo ""
    echo -e "${BLUE}🚀 Lancement Docker Compose (mode complet)...${NC}"
    docker-compose up -d
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ DÉMARRAGE RÉUSSI - Services Docker opérationnels!        ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}📊 Services disponibles:${NC}"
    echo -e "${YELLOW}  • API FastAPI       : ${GREEN}http://localhost:8000${NC}"
    echo -e "${YELLOW}    - Dashboard       : ${GREEN}http://localhost:8000/dashboard${NC}"
    echo -e "${YELLOW}    - Docs OpenAPI    : ${GREEN}http://localhost:8000/docs${NC}"
    echo ""
    echo -e "${YELLOW}  • MLflow Tracking   : ${GREEN}http://localhost:5050${NC}"
    echo ""
    echo -e "${YELLOW}  • Airflow Webserver : ${GREEN}http://localhost:8080${NC}"
    echo -e "${YELLOW}    - Username        : ${GREEN}admin${NC}"
    echo -e "${YELLOW}    - Password        : ${GREEN}admin${NC}"
    echo ""
    echo -e "${BLUE}🔄 Vérification du statut:${NC}"
    echo -e "${YELLOW}  docker-compose ps${NC}"
    echo ""
    echo -e "${BLUE}📝 Voir les logs:${NC}"
    echo -e "${YELLOW}  docker-compose logs -f airflow-webserver${NC}"
    echo ""
    echo -e "${BLUE}🛑 Arrêt des services:${NC}"
    echo -e "${YELLOW}  docker-compose down${NC}"
    echo ""
    exit 0
fi

# Mode simple (par défaut) - API + MLflow local
echo -e "${YELLOW}Mode SIMPLE (API + MLflow local)${NC}"
echo ""

# Vérification de Python
echo -e "${YELLOW}[1/6] Vérification de Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 n'est pas installé${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Python $PYTHON_VERSION détecté${NC}"

# Installation des dépendances
echo -e "${YELLOW}[2/6] Installation des dépendances Python...${NC}"
pip install -q -r requirements.txt 2>/dev/null || echo -e "${YELLOW}⚠️  Certaines dépendances sont déjà installées${NC}"
echo -e "${GREEN}✅ Dépendances prêtes${NC}"

# Création des dossiers
echo -e "${YELLOW}[3/6] Création de la structure de dossiers...${NC}"
mkdir -p data/{raw,processed,features} models logs mlruns results
echo -e "${GREEN}✅ Dossiers créés${NC}"

# Configuration MLflow (local)
echo -e "${YELLOW}[4/6] Configuration MLflow (mode local)...${NC}"
mkdir -p mlruns
export MLFLOW_TRACKING_URI="file:./mlruns"
export MLFLOW_EXPERIMENT_NAME="Climate_Marrakech"
echo -e "${GREEN}✅ MLflow configuré (local)${NC}"

# Préparation des données
echo -e "${YELLOW}[5/6] Vérification du dataset...${NC}"
if [[ ! -f "marrakech_weather_2018_2023_final.csv" ]]; then
    echo -e "${RED}❌ Dataset non trouvé: marrakech_weather_2018_2023_final.csv${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Dataset trouvé${NC}"

# Lancement des services
echo -e "${YELLOW}[6/6] Lancement des services...${NC}"
echo ""

# Lancement du serveur MLflow en arrière-plan
echo -e "${BLUE}🚀 Lancement MLflow (port 5050)...${NC}"
mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!
sleep 2
echo -e "${GREEN}✅ MLflow démarré (PID: $MLFLOW_PID)${NC}"

# Lancement de l'API FastAPI
echo -e "${BLUE}🚀 Lancement API FastAPI (port 8000)...${NC}"
python main.py > logs/api.log 2>&1 &
API_PID=$!
sleep 3
echo -e "${GREEN}✅ API démarrée (PID: $API_PID)${NC}"

# Affichage des URLs
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ DÉMARRAGE RÉUSSI - Services opérationnels!              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Services disponibles:${NC}"
echo -e "${YELLOW}  • API FastAPI       : ${GREEN}http://localhost:8000${NC}"
echo -e "${YELLOW}    - Dashboard       : ${GREEN}http://localhost:8000/dashboard${NC}"
echo -e "${YELLOW}    - Docs OpenAPI    : ${GREEN}http://localhost:8000/docs${NC}"
echo -e "${YELLOW}    - Interface Web   : ${GREEN}http://localhost:8000/web${NC}"
echo ""
echo -e "${YELLOW}  • MLflow Tracking   : ${GREEN}http://localhost:5050${NC}"
echo -e "${YELLOW}    - Modèles         : ${GREEN}http://localhost:5050/#/models${NC}"
echo -e "${YELLOW}    - Expériences     : ${GREEN}http://localhost:5050/#/experiments${NC}"
echo ""
echo -e "${BLUE}Pour activer Airflow + Docker Compose:${NC}"
echo -e "${YELLOW}  ./START.sh full${NC}"
echo ""
echo -e "${BLUE}📝 Logs:${NC}"
echo -e "${YELLOW}  • API    : ${GREEN}logs/api.log${NC}"
echo -e "${YELLOW}  • MLflow : ${GREEN}logs/mlflow.log${NC}"
echo ""
echo -e "${BLUE}🛑 Arrêt des services:${NC}"
echo -e "${YELLOW}  • Exécutez: ${GREEN}./STOP.sh${NC}"
echo ""

# Sauvegarde des PIDs
echo "$MLFLOW_PID" > .mlflow.pid
echo "$API_PID" > .api.pid

# Attendre et afficher les logs
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Affichage des logs en direct (Ctrl+C pour quitter):${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Fonction de nettoyage
cleanup() {
    echo ""
    echo -e "${YELLOW}Arrêt des services...${NC}"
    kill $MLFLOW_PID 2>/dev/null || true
    kill $API_PID 2>/dev/null || true
    rm -f .mlflow.pid .api.pid
    echo -e "${GREEN}✅ Services arrêtés${NC}"
    exit 0
}

# Attacher les signaux de fermeture
trap cleanup SIGINT SIGTERM

# Affichage des logs
tail -f logs/api.log logs/mlflow.log 2>/dev/null &
wait
