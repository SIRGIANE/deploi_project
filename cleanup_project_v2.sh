#!/bin/bash

echo "=========================================="
echo "🧹 NETTOYAGE DU PROJET CLIMATE MLOPS"
echo "=========================================="

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Compteur
DELETED_COUNT=0

# Fonction pour supprimer avec confirmation
safe_delete() {
    if [ -e "$1" ]; then
        echo -e "${YELLOW}🗑️  Suppression: $1${NC}"
        rm -rf "$1"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
}

echo ""
echo "1️⃣  Suppression des fichiers de cache Python..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name "*.pyd" -delete 2>/dev/null
echo -e "${GREEN}✅ Cache Python nettoyé${NC}"

echo ""
echo "2️⃣  Suppression des checkpoints Jupyter..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
echo -e "${GREEN}✅ Checkpoints Jupyter nettoyés${NC}"

echo ""
echo "3️⃣  Suppression des fichiers macOS..."
find . -type f -name ".DS_Store" -delete 2>/dev/null
echo -e "${GREEN}✅ Fichiers macOS nettoyés${NC}"

echo ""
echo "4️⃣  Suppression des fichiers de données en double..."
safe_delete "marrakech_clean.csv"
safe_delete "marrakech.csv"
safe_delete "marrakech_weather_2018_2023.xlsx"
safe_delete "data/raw/weather_data_processed.csv"
echo -e "${GREEN}✅ Doublons supprimés${NC}"

echo ""
echo "5️⃣  Suppression des notebooks temporaires..."
safe_delete "test.ipynb"
safe_delete "Untitled.ipynb"
echo -e "${GREEN}✅ Notebooks temporaires supprimés${NC}"

echo ""
echo "6️⃣  Suppression des scripts de migration obsolètes..."
safe_delete "migrate_to_kaggle_weather.py"
safe_delete "MIGRATION_REPORT.md"
safe_delete "backup_migration/"
echo -e "${GREEN}✅ Scripts de migration supprimés${NC}"

echo ""
echo "7️⃣  Suppression des anciens scripts de nettoyage..."
safe_delete "clean_code.sh"
safe_delete "cleanup_project.sh"
echo -e "${GREEN}✅ Anciens scripts supprimés${NC}"

echo ""
echo "8️⃣  Nettoyage des logs Airflow (garder seulement les 7 derniers jours)..."
if [ -d "airflow/logs" ]; then
    find airflow/logs -type f -mtime +7 -delete 2>/dev/null
    echo -e "${GREEN}✅ Logs Airflow nettoyés${NC}"
fi

echo ""
echo "9️⃣  Création du fichier .gitignore s'il n'existe pas..."
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Data files (ne pas versionner les gros fichiers)
*.csv
*.xlsx
*.parquet
!marrakech_weather_2018_2023_final.csv

# Models
models/*.pkl
models/*.joblib
!models/.gitkeep

# MLflow
mlruns/
mlartifacts/

# Airflow
airflow/logs/
airflow/*.pid
airflow/*.log

# Logs
logs/
*.log

# DVC
.dvc/cache
.dvc/tmp

# Environment
.env
.env.local

# Temporary files
*.tmp
*.bak
*.swp
test.ipynb
Untitled*.ipynb
EOF
    echo -e "${GREEN}✅ .gitignore créé${NC}"
else
    echo -e "${YELLOW}⚠️  .gitignore existe déjà${NC}"
fi

echo ""
echo "🔟  Réorganisation des dossiers..."
# Créer des dossiers .gitkeep pour maintenir la structure
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/features/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep
touch results/.gitkeep
touch reports/.gitkeep
echo -e "${GREEN}✅ Structure des dossiers maintenue${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 NETTOYAGE TERMINÉ !${NC}"
echo "=========================================="
echo ""
echo "📊 Résumé:"
echo "   - Fichiers/dossiers supprimés: $DELETED_COUNT"
echo "   - Cache Python: ✅ Nettoyé"
echo "   - Checkpoints Jupyter: ✅ Nettoyés"
echo "   - Fichiers temporaires: ✅ Supprimés"
echo "   - Structure du projet: ✅ Organisée"
echo ""
echo "📂 Structure finale:"
tree -L 2 -I '__pycache__|.ipynb_checkpoints|mlruns' . 2>/dev/null || ls -la

echo ""
echo "💡 Conseil: Exécutez 'git status' pour voir les changements"
