#!/bin/bash

# 🛑 Climate MLOps - Script d'Arrêt

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Arrêt des services Climate MLOps...${NC}"
echo ""

# Arrêt via les fichiers PID
if [[ -f ".mlflow.pid" ]]; then
    MLFLOW_PID=$(cat .mlflow.pid)
    if kill -0 $MLFLOW_PID 2>/dev/null; then
        kill $MLFLOW_PID
        echo -e "${GREEN}✅ MLflow arrêté (PID: $MLFLOW_PID)${NC}"
    fi
    rm -f .mlflow.pid
fi

if [[ -f ".api.pid" ]]; then
    API_PID=$(cat .api.pid)
    if kill -0 $API_PID 2>/dev/null; then
        kill $API_PID
        echo -e "${GREEN}✅ API arrêtée (PID: $API_PID)${NC}"
    fi
    rm -f .api.pid
fi

# Arrêt des processus restants
echo -e "${YELLOW}Vérification des processus restants...${NC}"
pkill -f "mlflow server" || true
pkill -f "uvicorn" || true
pkill -f "python main.py" || true

sleep 1

echo ""
echo -e "${GREEN}✅ Tous les services ont été arrêtés${NC}"
echo -e "${YELLOW}Pour redémarrer: ${GREEN}./START.sh${NC}"
