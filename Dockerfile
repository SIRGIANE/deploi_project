# ============================================================================
# DOCKERFILE - Climate MLOps
# ============================================================================

FROM apache/airflow:2.7.3-python3.10

# Métadonnées
LABEL maintainer="Climate MLOps Team"
LABEL description="Unified image for Airflow, API, and ML services"

# ============================================================================
# 1. Installation système (Root)
# ============================================================================
USER root

# Installation des outils de compilation et librairies système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Créer les répertoires nécessaires et ajuster les permissions
RUN mkdir -p /workspace/data /workspace/models /workspace/logs /workspace/mlruns \
    && chown -R airflow:root /workspace

# ============================================================================
# 2. Installation des dépendances Python (Airflow User)
# ============================================================================
USER airflow

# Copier requirements.txt
COPY requirements.txt /tmp/requirements.txt

# Installer toutes les dépendances (ML libs + API + Airflow providers)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
    apache-airflow-providers-postgres \
    apache-airflow-providers-http \
    apache-airflow-providers-celery \
    apache-airflow-providers-redis && \
    pip install --no-cache-dir --upgrade "email-validator>=2.0.0"

# ============================================================================
# 3. Configuration de l'environnement
# ============================================================================
# Définir le path pour inclure le code source
ENV PYTHONPATH="${PYTHONPATH}:/workspace/src:/workspace"
ENV AIRFLOW_HOME=/opt/airflow

WORKDIR /workspace

# Copier tout le code source dans l'image
COPY --chown=airflow:root . /workspace

# Copier les DAGs vers le dossier Airflow
RUN cp -r /workspace/airflow/dags/* /opt/airflow/dags/ 2>/dev/null || true

# Exposer les ports potentiels (Airflow, API, MLflow, Jupyter)
EXPOSE 8080 8793 8000 8001 5000 8888

# Le point d'entrée par défaut reste celui d'Airflow, mais on peut le surcharger
# CMD ["webserver"] ou ["python", "main.py"] selon le service
