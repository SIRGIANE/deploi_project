# 🚀 Guide de Déploiement Azure - Climate MLOps

Ce guide vous accompagne pas à pas pour déployer votre application Climate MLOps sur **Azure Container Apps**, la solution idéale pour des applications multi-conteneurs comme la vôtre.

---

## 📋 Prérequis

### 1. Installation d'Azure CLI

```bash
# macOS
brew update && brew install azure-cli

# Vérifier l'installation
az --version
```

### 2. Connexion à Azure

```bash
# Se connecter à votre compte Azure
az login

# Vérifier votre abonnement actif
az account show

# (Optionnel) Changer d'abonnement si nécessaire
az account list --output table
az account set --subscription "VOTRE_SUBSCRIPTION_ID"
```

### 3. Installer l'extension Container Apps

```bash
az extension add --name containerapp --upgrade
```

---

## 🏗️ Architecture sur Azure

Votre application sera déployée avec:
- **Azure Container Apps** : Pour héberger vos conteneurs (API, Airflow, MLflow)
- **Azure Database for PostgreSQL** : Pour les bases de données (Airflow, Weather Data)
- **Azure Cache for Redis** : Pour Celery/Airflow
- **Azure Container Registry** : Pour stocker vos images Docker
- **Azure File Share** : Pour les volumes persistants (mlruns, models, data)

---

## 🔧 Étape 1 : Configuration Initiale

### 1.1 Définir les variables d'environnement

```bash
# Configuration de base
export RESOURCE_GROUP="rg-projet"
export SUBSCRIPTION_ID="1815cb03-0ab6-4382-9f78-d03c507c84e4"
export LOCATION="francecentral"
export ENVIRONMENT="climate-mlops-env"
export ACR_NAME="climatemlopsreg$(date +%s)"

# Définir la souscription
az account set --subscription $SUBSCRIPTION_ID
export API_APP="weather-api"
export AIRFLOW_WEB="airflow-webserver"
export AIRFLOW_SCHEDULER="airflow-scheduler"
export MLFLOW_APP="mlflow-server"
```

### 1.2 Créer le Resource Group

```bash
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

---

## 📦 Étape 2 : Azure Container Registry (ACR)

### 2.1 Créer le Registry

```bash
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

# Récupérer les credentials
export ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
export ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
export ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

echo "ACR Login Server: $ACR_LOGIN_SERVER"
```

### 2.2 Build et Push l'image Docker

```bash
# Se connecter au registry
az acr login --name $ACR_NAME

# Build et push l'image unifiée
docker build -t $ACR_LOGIN_SERVER/climate-mlops:latest -f Dockerfile .
docker push $ACR_LOGIN_SERVER/climate-mlops:latest

# Vérifier que l'image est bien poussée
az acr repository list --name $ACR_NAME --output table
```

---

## 🗄️ Étape 3 : Bases de Données et Services Managés

### 3.1 Créer Azure Database for PostgreSQL

```bash
# PostgreSQL pour Airflow
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name airflow-postgres-$(date +%s) \
  --location $LOCATION \
  --admin-user airflowadmin \
  --admin-password "VotreMotDePasseSecurise123!" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --storage-size 32 \
  --version 13

# Créer la base de données Airflow
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name <SERVER_NAME_FROM_ABOVE> \
  --database-name airflow

# PostgreSQL pour Weather Data
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name weather-postgres-$(date +%s) \
  --location $LOCATION \
  --admin-user weatheradmin \
  --admin-password "VotreMotDePasseSecurise123!" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --storage-size 32 \
  --version 13

# Créer la base de données Weather
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name <SERVER_NAME_FROM_ABOVE> \
  --database-name weather_data
```

### 3.2 Créer Azure Cache for Redis

```bash
az redis create \
  --resource-group $RESOURCE_GROUP \
  --name climate-mlops-redis \
  --location $LOCATION \
  --sku Basic \
  --vm-size c0 \
  --enable-non-ssl-port

# Récupérer la clé d'accès
export REDIS_KEY=$(az redis list-keys --resource-group $RESOURCE_GROUP --name climate-mlops-redis --query primaryKey --output tsv)
```

---

## ☁️ Étape 4 : Azure Container Apps Environment

### 4.1 Créer l'environnement Container Apps

```bash
az containerapp env create \
  --name $ENVIRONMENT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

### 4.2 Créer un Azure File Share pour les volumes persistants

```bash
# Créer un Storage Account
export STORAGE_ACCOUNT="climatemlops$(date +%s)"

az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

# Récupérer la clé
export STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT --query [0].value --output tsv)

# Créer les File Shares
az storage share create --name mlruns --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
az storage share create --name models --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
az storage share create --name data --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
az storage share create --name airflow-logs --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY

# Ajouter le storage à l'environnement Container Apps
az containerapp env storage set \
  --name $ENVIRONMENT \
  --resource-group $RESOURCE_GROUP \
  --storage-name mlruns \
  --azure-file-account-name $STORAGE_ACCOUNT \
  --azure-file-account-key $STORAGE_KEY \
  --azure-file-share-name mlruns \
  --access-mode ReadWrite

# Répéter pour les autres shares (models, data, airflow-logs)
```

---

## 🚀 Étape 5 : Déployer les Container Apps

### 5.1 Déployer l'API Weather

```bash
az containerapp create \
  --name $API_APP \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image $ACR_LOGIN_SERVER/climate-mlops:latest \
  --target-port 8000 \
  --ingress external \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 1 \
  --max-replicas 3 \
  --command "python" "main.py" \
  --env-vars \
    POSTGRES_HOST="<WEATHER_POSTGRES_SERVER>.postgres.database.azure.com" \
    POSTGRES_PORT="5432" \
    POSTGRES_USER="weatheradmin" \
    POSTGRES_PASSWORD="VotreMotDePasseSecurise123!" \
    POSTGRES_DB="weather_data" \
    API_HOST="0.0.0.0" \
    API_PORT="8000"
```

### 5.2 Déployer MLflow

```bash
az containerapp create \
  --name $MLFLOW_APP \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image $ACR_LOGIN_SERVER/climate-mlops:latest \
  --target-port 5000 \
  --ingress external \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --cpu 0.5 \
  --memory 1.0Gi \
  --command "mlflow" "server" "--host" "0.0.0.0" "--port" "5000" "--backend-store-uri" "sqlite:////mlruns/mlflow.db" "--default-artifact-root" "/mlruns/artifacts"
```

### 5.3 Déployer Airflow Webserver

```bash
az containerapp create \
  --name $AIRFLOW_WEB \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image $ACR_LOGIN_SERVER/climate-mlops:latest \
  --target-port 8080 \
  --ingress external \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --cpu 1.0 \
  --memory 2.0Gi \
  --command "airflow" "webserver" \
  --env-vars \
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://airflowadmin:VotreMotDePasseSecurise123!@<AIRFLOW_POSTGRES_SERVER>.postgres.database.azure.com:5432/airflow" \
    AIRFLOW__CORE__EXECUTOR="LocalExecutor" \
    AIRFLOW__CORE__LOAD_EXAMPLES="False"
```

### 5.4 Déployer Airflow Scheduler (Job en tâche de fond)

```bash
az containerapp create \
  --name $AIRFLOW_SCHEDULER \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image $ACR_LOGIN_SERVER/climate-mlops:latest \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --cpu 0.5 \
  --memory 1.0Gi \
  --command "airflow" "scheduler" \
  --env-vars \
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://airflowadmin:VotreMotDePasseSecurise123!@<AIRFLOW_POSTGRES_SERVER>.postgres.database.azure.com:5432/airflow" \
    AIRFLOW__CORE__EXECUTOR="LocalExecutor" \
    AIRFLOW__CORE__LOAD_EXAMPLES="False"
```

---

## 🔐 Étape 6 : Configuration Réseau et Sécurité

### 6.1 Permettre l'accès aux bases PostgreSQL

```bash
# Autoriser Azure services
az postgres flexible-server firewall-rule create \
  --resource-group $RESOURCE_GROUP \
  --name <AIRFLOW_POSTGRES_SERVER> \
  --rule-name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

# Répéter pour weather-postgres-server
```

### 6.2 Configurer HTTPS (Optionnel mais recommandé)

Azure Container Apps fournit automatiquement un certificat HTTPS. Pour un domaine personnalisé:

```bash
az containerapp hostname add \
  --name $API_APP \
  --resource-group $RESOURCE_GROUP \
  --hostname votre-domaine.com

az containerapp hostname bind \
  --name $API_APP \
  --resource-group $RESOURCE_GROUP \
  --hostname votre-domaine.com \
  --validation-method HTTP
```

---

## ✅ Étape 7 : Initialiser Airflow

Avant de lancer Airflow, il faut initialiser la base de données:

```bash
# Créer un job temporaire pour initialiser Airflow
az containerapp job create \
  --name airflow-init-job \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image $ACR_LOGIN_SERVER/climate-mlops:latest \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --trigger-type Manual \
  --replica-timeout 300 \
  --command "/bin/bash" "-c" "airflow db upgrade && airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin" \
  --env-vars \
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://airflowadmin:VotreMotDePasseSecurise123!@<AIRFLOW_POSTGRES_SERVER>.postgres.database.azure.com:5432/airflow"

# Exécuter le job
az containerapp job start --name airflow-init-job --resource-group $RESOURCE_GROUP
```

---

## 📊 Étape 8 : Vérification et Tests

### 8.1 Récupérer les URLs

```bash
# API URL
az containerapp show --name $API_APP --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv

# MLflow URL
az containerapp show --name $MLFLOW_APP --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv

# Airflow URL
az containerapp show --name $AIRFLOW_WEB --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv
```

### 8.2 Tester l'API

```bash
export API_URL=$(az containerapp show --name $API_APP --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)

curl https://$API_URL/health
curl https://$API_URL/docs  # Swagger UI
```

---

## 🔄 Étape 9 : Mises à jour et CI/CD (Optionnel)

### 9.1 Mettre à jour une application

```bash
# Rebuild l'image
docker build -t $ACR_LOGIN_SERVER/climate-mlops:latest -f Dockerfile .
docker push $ACR_LOGIN_SERVER/climate-mlops:latest

# Redéployer
az containerapp update \
  --name $API_APP \
  --resource-group $RESOURCE_GROUP \
  --image $ACR_LOGIN_SERVER/climate-mlops:latest
```

### 9.2 Configuration GitHub Actions (Optionnel)

Créez `.github/workflows/azure-deploy.yml` pour le déploiement automatique à chaque push.

---

## 💰 Estimation des Coûts (mensuel approximatif)

| Service | Configuration | Coût estimé |
|---------|--------------|-------------|
| Container Apps (API) | 1 vCPU, 2Gi RAM | ~€30 |
| Container Apps (Airflow Web + Scheduler) | 1.5 vCPU total | ~€25 |
| Container Apps (MLflow) | 0.5 vCPU, 1Gi RAM | ~€15 |
| PostgreSQL Flexible (x2) | Basic tier | ~€20 chacun |
| Redis Cache | Basic C0 | ~€16 |
| Container Registry | Basic | €4 |
| Storage Account | 100GB | ~€2 |
| **TOTAL** | | **~€132/mois** |

> 💡 **Conseil** : Utilisez des budgets Azure (`az consumption budget create`) pour éviter les surprises.

---

## 🛠️ Commandes Utiles

```bash
# Voir les logs d'une app
az containerapp logs show --name $API_APP --resource-group $RESOURCE_GROUP --follow

# Scaler une app
az containerapp update --name $API_APP --resource-group $RESOURCE_GROUP --min-replicas 2 --max-replicas 5

# Arrêter une app (scale to 0)
az containerapp update --name $API_APP --resource-group $RESOURCE_GROUP --min-replicas 0 --max-replicas 0

# Supprimer tout le Resource Group (ATTENTION : irréversible)
az group delete --name $RESOURCE_GROUP --yes --no-wait
```

---

## 🚨 Troubleshooting

### Problème : L'app ne démarre pas

```bash
# Vérifier les logs
az containerapp logs show --name $API_APP --resource-group $RESOURCE_GROUP --follow

# Vérifier les replicas
az containerapp replica list --name $API_APP --resource-group $RESOURCE_GROUP -o table
```

### Problème : Connexion PostgreSQL échoue

1. Vérifiez que la règle firewall est bien configurée
2. Vérifiez la chaîne de connexion (format : `host.postgres.database.azure.com`)
3. Activez SSL si nécessaire (`sslmode=require`)

### Problème : Images ne se pullent pas

```bash
# Vérifier que l'ACR est bien configuré
az acr check-health --name $ACR_NAME --yes

# Vérifier les credentials
az containerapp registry list --name $API_APP --resource-group $RESOURCE_GROUP
```

---

## 📚 Ressources Supplémentaires

- [Documentation Azure Container Apps](https://learn.microsoft.com/azure/container-apps/)
- [Azure CLI Reference](https://learn.microsoft.com/cli/azure/)
- [PostgreSQL Flexible Server](https://learn.microsoft.com/azure/postgresql/flexible-server/)
- [Azure Cache for Redis](https://learn.microsoft.com/azure/azure-cache-for-redis/)

---

**✨ Votre application Climate MLOps est maintenant prête pour Azure !**
