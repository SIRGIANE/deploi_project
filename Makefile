# Makefile pour Climate MLOps
# Une seule commande pour tout gérer

# Variables
DC = docker-compose
PY = python3

.PHONY: help up down restart logs test clean status

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## Démarre tous les services (construit si nécessaire)
	$(DC) up -d --build --remove-orphans
	@echo "⏳ Attente du démarrage des services..."
	@sleep 10
	@$(PY) verify_deployment.py

down: ## Arrête tous les services
	$(DC) down

restart: down up ## Redémarre tout de zéro

logs: ## Affiche les logs en temps réel
	$(DC) logs -f

status: ## Vérifie l'état des services
	@$(PY) verify_deployment.py

clean: down ## Nettoie tout (volumes inclus)
	$(DC) down -v
	docker system prune -f

test: ## Lance les tests unitaires dans le conteneur
	$(DC) exec weather-api pytest tests/ -v
