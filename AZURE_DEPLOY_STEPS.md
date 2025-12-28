# 🚀 Guide de Déploiement Azure : Commande par Commande

Ce guide vous permet de redéployer tout le projet sur Azure. Remplacez les variables si nécessaire.

## 1. Initialisation et Connexion
Vérifiez que vous êtes connecté au bon compte.
```bash
# Connexion interactive
az login --use-device-code

# Définir la souscription
export SUBSCRIPTION_ID="1815cb03-0ab6-4382-9f78-d03c507c84e4"
az account set --subscription $SUBSCRIPTION_ID

# Variables de base
export RESOURCE_GROUP="rg-projet"
export LOCATION="switzerlandnorth"
export ACR_NAME="climatemlopsreg$(date +%s)"
export ENVIRONMENT="climate-mlops-env"
export API_APP="weather-api"
```

## 2. Enregistrement des Services Azure (Une seule fois)
Indispensable pour permettre à votre compte de créer ces types de ressources.
```bash
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

## 3. Création du Registre (ACR)
```bash
# Créer le registre
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true --location $LOCATION

# Se connecter au registre localement
az acr login --name $ACR_NAME

# Récupérer le serveur de login (ex: climatemlopsreg123.azurecr.io)
export ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
```

## 4. Build et Push de l'Image Docker
Assurez-vous d'être à la racine du projet `climate-mlops`.
```bash
# Taguer l'image locale (construite avec 'docker-compose build')
docker tag climate-mlops-weather-api:latest $ACR_LOGIN_SERVER/climate-mlops:latest

# Pousser l'image vers Azure
docker push $ACR_LOGIN_SERVER/climate-mlops:latest
```

## 5. Création de l'Environnement Container Apps
```bash
# Créer l'environnement
az containerapp env create --name $ENVIRONMENT --resource-group $RESOURCE_GROUP --location $LOCATION
```

## 6. Déploiement de l'API (Weather API)
```bash
# Récupérer les identifiants ACR pour l'authentification automatique
export ACR_USER=$(az acr credential show --name $ACR_NAME --query "username" --output tsv)
export ACR_PASS=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)

# Créer et lancer l'application
az containerapp create \
  --name $API_APP \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT \
  --image $ACR_LOGIN_SERVER/climate-mlops:latest \
  --target-port 8000 \
  --ingress external \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USER \
  --registry-password $ACR_PASS \
  --cpu 1.0 --memory 2.0Gi \
  --command "python" "main.py" \
  --env-vars PYTHONPATH="/workspace"
```

## 7. Vérification finale
```bash
# Récupérer l'URL publique
export FQDN=$(az containerapp show --name $API_APP --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn --output tsv)
echo "Votre API est en ligne ici : https://$FQDN"

# Tester la santé
curl "https://$FQDN/health"
```
