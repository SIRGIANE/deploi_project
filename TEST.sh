#!/bin/bash

# 🧪 Climate MLOps - Script de Test Complet

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     🧪 CLIMATE MLOPS - Tests Complets des Services 🧪         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

FAILED=0
PASSED=0

# Fonction pour tester un endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local method=$3
    
    echo -ne "${YELLOW}[TEST] $name...${NC} "
    
    if [[ "$method" == "POST" ]]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$url" 2>&1)
    else
        response=$(curl -s -w "\n%{http_code}" "$url" 2>&1)
    fi
    
    http_code=$(echo "$response" | tail -n 1)
    
    if [[ "$http_code" == "200" ]]; then
        echo -e "${GREEN}✅ OK (HTTP $http_code)${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAILED (HTTP $http_code)${NC}"
        ((FAILED++))
        return 1
    fi
}

# Fonction pour tester un port
test_port() {
    local name=$1
    local port=$2
    
    echo -ne "${YELLOW}[TEST] Port $port ($name)...${NC} "
    
    if timeout 2 bash -c "< /dev/null > /dev/tcp/localhost/$port" 2>/dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAILED (port fermé)${NC}"
        ((FAILED++))
        return 1
    fi
}

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}1️⃣  TEST DES PORTS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

test_port "API FastAPI" 8000
test_port "MLflow" 5050

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}2️⃣  TEST DE L'API FASTAPI${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

test_endpoint "GET /health" "http://localhost:8000/health"
test_endpoint "GET /models" "http://localhost:8000/models"
test_endpoint "GET /dashboard" "http://localhost:8000/dashboard"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}3️⃣  TEST DES FICHIERS CRITIQUES${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Vérifier les fichiers essentiels
check_file() {
    local file=$1
    local name=$2
    
    echo -ne "${YELLOW}[TEST] Fichier: $name...${NC} "
    
    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✅ OK${NC}"
        ((PASSED++))
    else
        echo -e "${RED}❌ MANQUANT${NC}"
        ((FAILED++))
    fi
}

check_file "marrakech_weather_2018_2023_final.csv" "Dataset"
check_file "models/rf_model.pkl" "Modèle RandomForest"
check_file "models/scaler.pkl" "Scaler"
check_file "models/data_pipeline.joblib" "Pipeline"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}4️⃣  TEST DES DOSSIERS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

check_dir() {
    local dir=$1
    local name=$2
    
    echo -ne "${YELLOW}[TEST] Dossier: $name...${NC} "
    
    if [[ -d "$dir" ]]; then
        echo -e "${GREEN}✅ OK${NC}"
        ((PASSED++))
    else
        echo -e "${RED}❌ MANQUANT${NC}"
        ((FAILED++))
    fi
}

check_dir "data/raw" "Data Raw"
check_dir "data/processed" "Data Processed"
check_dir "data/features" "Data Features"
check_dir "models" "Models"
check_dir "logs" "Logs"
check_dir "mlruns" "MLruns"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}5️⃣  TEST DU DATASET${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [[ -f "marrakech_weather_2018_2023_final.csv" ]]; then
    rows=$(wc -l < marrakech_weather_2018_2023_final.csv)
    echo -e "${YELLOW}Dataset Info:${NC}"
    echo -e "  • ${GREEN}Rows: $(($rows - 1))${NC}"
    echo -e "  • ${GREEN}Size: $(ls -lh marrakech_weather_2018_2023_final.csv | awk '{print $5}')${NC}"
    echo -e "${GREEN}✅ Dataset OK${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ Dataset manquant${NC}"
    ((FAILED++))
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}RÉSUMÉ DES TESTS${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

TOTAL=$((PASSED + FAILED))
SUCCESS_RATE=$((PASSED * 100 / TOTAL))

echo -e "${GREEN}✅ Tests réussis: $PASSED${NC}"
echo -e "${RED}❌ Tests échoués: $FAILED${NC}"
echo -e "${BLUE}📊 Total: $TOTAL tests${NC}"
echo -e "${YELLOW}📈 Taux de réussite: $SUCCESS_RATE%${NC}"

echo ""

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  🎉 TOUS LES TESTS SONT PASSÉS! 🎉                         ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}🚀 Prêt pour le démarrage!${NC}"
    echo -e "${YELLOW}  • ./START.sh          (Mode simple: API + MLflow)${NC}"
    echo -e "${YELLOW}  • ./START.sh full     (Mode complet: Airflow + Docker)${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ⚠️  CERTAINS TESTS ONT ÉCHOUÉ                              ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Actions recommandées:${NC}"
    echo -e "  • Vérifiez que l'API et MLflow sont démarrés: ./START.sh"
    echo -e "  • Vérifiez que le dataset existe dans le répertoire courant"
    echo -e "  • Vérifiez les permissions des dossiers"
    echo ""
    exit 1
fi
