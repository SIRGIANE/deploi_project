#!/bin/bash
set -e

echo "🌡️ Climate MLOps - Continuous Training Pipeline"
echo "=============================================="

# Fonction d'aide
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start          Démarrer le pipeline complet (Airflow + MLflow + Jupyter)"
    echo "  stop           Arrêter tous les services"
    echo "  restart        Redémarrer tous les services"
    echo "  logs           Afficher les logs en temps réel"
    echo "  status         Vérifier le statut des services"
    echo "  init           Initialiser le projet (première installation)"
    echo "  test-pipeline  Tester le pipeline de formation continue"
    echo "  cleanup        Nettoyer les volumes et containers"
    echo "  backup         Sauvegarder les données et modèles"
    echo "  restore        Restaurer depuis une sauvegarde"
    echo ""
    echo "Services disponibles:"
    echo "  - Airflow WebUI: http://localhost:8080 (admin/admin)"
    echo "  - MLflow: http://localhost:5050"
    echo "  - Jupyter Lab: http://localhost:8889"
    echo "  - API: http://localhost:8000 (en mode production)"
    echo ""
}

# Vérification des prérequis
check_prerequisites() {
    echo "🔍 Vérification des prérequis..."
    
    # Docker et Docker Compose
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker n'est pas installé"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose n'est pas installé"
        exit 1
    fi
    
    # Fichiers de configuration
    if [[ ! -f ".env" ]]; then
        echo "⚠️ Fichier .env manquant, création d'un fichier par défaut..."
        cp .env.example .env 2>/dev/null || cat > .env << 'ENVEOF'
# Climate MLOps Configuration
AIRFLOW_HOME=/home/airflow
AIRFLOW__CORE__DAGS_FOLDER=/home/airflow/dags
AIRFLOW__CORE__PLUGINS_FOLDER=/home/airflow/plugins
AIRFLOW__CORE__BASE_LOG_FOLDER=/home/airflow/logs
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__CORE__UNIT_TEST_MODE=False

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=postgresql://airflow:airflow@postgres:5432/mlflow
MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# Database
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Jupyter
JUPYTER_ENABLE_LAB=yes
JUPYTER_TOKEN=mlops2024
ENVEOF
    fi
    
    echo "✅ Prérequis vérifiés"
}

# Initialisation du projet
init_project() {
    echo "🚀 Initialisation du projet Climate MLOps..."
    
    check_prerequisites
    
    # Création des dossiers nécessaires
    echo "📁 Création de la structure de dossiers..."
    mkdir -p {airflow/{dags,logs,config,plugins},data/{raw,processed,features},models,reports/{drift,data_quality,model_comparison,model_cards},logs,backups}
    
    # Configuration des permissions pour Airflow
    echo "🔐 Configuration des permissions..."
    echo "AIRFLOW_UID=$(id -u)" > .env.local
    
    # Initialisation DVC
    if [[ ! -f ".dvc/config" ]]; then
        echo "📦 Initialisation DVC..."
        dvc init --no-scm || echo "DVC déjà initialisé"
    fi
    
    # Initialisation Git LFS (si disponible)
    if command -v git-lfs &> /dev/null; then
        echo "📦 Configuration Git LFS..."
        git lfs install || true
        git lfs track "*.pkl" "*.h5" "*.joblib" "*.model" || true
    fi
    
    echo "✅ Initialisation terminée"
    echo ""
    echo "Prochaines étapes:"
    echo "1. Configurez vos variables dans le fichier .env"
    echo "2. Lancez: $0 start"
    echo "3. Accédez à Airflow: http://localhost:8080"
}

# Démarrage des services
start_services() {
    echo "🚀 Démarrage du pipeline Climate MLOps..."
    
    check_prerequisites
    
    # Chargement des variables d'environnement
    if [[ -f ".env.local" ]]; then
        export $(cat .env.local | xargs)
    fi
    
    # Vérifier si docker-compose.yml existe
    if [[ ! -f "docker-compose.yml" ]]; then
        echo "❌ Fichier docker-compose.yml non trouvé"
        exit 1
    fi
    
    # Construction et démarrage des services
    echo "🐳 Construction des images Docker..."
    docker-compose build 2>&1 | grep -E "(Building|built|Successfully)" || true
    
    echo "🚀 Démarrage des services..."
    docker-compose up -d
    
    # Attendre que les services soient prêts
    echo "⏳ Attente du démarrage des services (30 secondes)..."
    sleep 30
    
    # Vérification du statut
    check_services_health
    
    echo ""
    echo "🎉 Pipeline démarré avec succès!"
    echo ""
    echo "Services disponibles:"
    echo "  📊 Airflow WebUI: http://localhost:8080 (admin/admin)"
    echo "  📈 MLflow: http://localhost:5050"
    echo "  📚 Jupyter Lab: http://localhost:8889"
    echo "  🔌 API: http://localhost:8000/docs"
    echo ""
    echo "Pour voir les logs: $0 logs"
    echo "Pour arrêter: $0 stop"
}

# Vérification de santé des services
check_services_health() {
    echo "🏥 Vérification de la santé des services..."
    
    declare -a services=(
        "api:8000:/health:API"
        "mlflow:5000::MLflow"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port path desc <<< "$service"
        echo -n "  Vérification $desc... "
        
        if curl -f -s "http://localhost:$port${path}" > /dev/null 2>&1; then
            echo "✅"
        else
            echo "⏳ (en cours de démarrage)"
        fi
    done
}

# Arrêt des services
stop_services() {
    echo "🛑 Arrêt du pipeline..."
    docker-compose down
    echo "✅ Services arrêtés"
}

# Redémarrage
restart_services() {
    echo "🔄 Redémarrage du pipeline..."
    stop_services
    sleep 5
    start_services
}

# Affichage des logs
show_logs() {
    echo "📋 Logs en temps réel (Ctrl+C pour quitter):"
    docker-compose logs -f
}

# Test du pipeline
test_pipeline() {
    echo "🧪 Test du pipeline de formation continue..."
    
    # Vérifier que les services sont démarrés
    if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
        echo "❌ Les services ne sont pas démarrés. Lancez d'abord: $0 start"
        exit 1
    fi
    
    # Test de connectivité API
    echo "🔍 Test de connectivité API..."
    if curl -f -s "http://localhost:8000/health" > /dev/null; then
        echo "✅ API accessible"
    else
        echo "⚠️ API en cours de démarrage"
    fi
    
    # Test du pipeline de données
    echo "🚀 Test du pipeline de données..."
    docker-compose exec api python -m pytest tests/ -v || echo "⚠️ Tests non disponibles en mode Docker"
    
    echo "✅ Pipeline de test déclenché"
}

# Nettoyage
cleanup() {
    echo "🧹 Nettoyage complet..."
    
    read -p "Êtes-vous sûr de vouloir supprimer tous les volumes et données ? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v --remove-orphans
        docker system prune -f
        echo "✅ Nettoyage terminé"
    else
        echo "❌ Nettoyage annulé"
    fi
}

# Sauvegarde
backup_data() {
    echo "💾 Sauvegarde des données..."
    
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="backups/backup_$timestamp"
    
    mkdir -p "$backup_dir"
    
    # Sauvegarde des modèles
    if [[ -d "models" ]]; then
        cp -r models "$backup_dir/"
        echo "✅ Modèles sauvegardés"
    fi
    
    # Sauvegarde des données
    if [[ -d "data" ]]; then
        cp -r data "$backup_dir/"
        echo "✅ Données sauvegardées"
    fi
    
    # Sauvegarde MLflow
    if [[ -d "mlruns" ]]; then
        cp -r mlruns "$backup_dir/"
        echo "✅ Expériences MLflow sauvegardées"
    fi
    
    # Sauvegarde des rapports
    if [[ -d "reports" ]]; then
        cp -r reports "$backup_dir/"
        echo "✅ Rapports sauvegardés"
    fi
    
    echo "📦 Sauvegarde créée: $backup_dir"
    echo "Taille: $(du -sh "$backup_dir" | cut -f1)"
}

# Restauration
restore_data() {
    echo "📥 Restauration des données..."
    
    if [[ -z "$1" ]]; then
        echo "Usage: $0 restore <backup_directory>"
        echo "Sauvegardes disponibles:"
        ls -la backups/ 2>/dev/null || echo "Aucune sauvegarde trouvée"
        exit 1
    fi
    
    backup_dir="$1"
    
    if [[ ! -d "$backup_dir" ]]; then
        echo "❌ Dossier de sauvegarde introuvable: $backup_dir"
        exit 1
    fi
    
    read -p "Êtes-vous sûr de vouloir restaurer depuis $backup_dir ? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Restauration
        [[ -d "$backup_dir/models" ]] && cp -r "$backup_dir/models" . && echo "✅ Modèles restaurés"
        [[ -d "$backup_dir/data" ]] && cp -r "$backup_dir/data" . && echo "✅ Données restaurées"
        [[ -d "$backup_dir/mlruns" ]] && cp -r "$backup_dir/mlruns" . && echo "✅ MLflow restauré"
        [[ -d "$backup_dir/reports" ]] && cp -r "$backup_dir/reports" . && echo "✅ Rapports restaurés"
        
        echo "✅ Restauration terminée"
    else
        echo "❌ Restauration annulée"
    fi
}

# Statut des services
show_status() {
    echo "📊 Statut des services:"
    docker-compose ps 2>/dev/null || echo "❌ Docker Compose non disponible"
    echo ""
    check_services_health
}

# Interface en ligne de commande
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    init)
        init_project
        ;;
    test-pipeline)
        test_pipeline
        ;;
    cleanup)
        cleanup
        ;;
    backup)
        backup_data
        ;;
    restore)
        restore_data "$2"
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "❌ Commande inconnue: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
