# ──────────────────────────────────────────────────────────────────────────────
# train_moisture.Dockerfile — Imagen efímera para reentrenar el MLP de humedad.
#
# Build (desde prod/):
#   docker build -f services/training/moisture/Dockerfile -t irridea-train-moisture .
#
# Uso (lo invoca DockerOperator desde Airflow):
#   docker run --rm \
#       --network deployment_default \
#       -v <host>/models:/app/models \
#       -v <host>/data:/app/data \
#       -v mlflow-artifacts:/mlflow-artifacts \
#       -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
#       irridea-train-moisture \
#       python -m src.moisture.pipeline --models-dir /app/models
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stack pesado: TF + scikit + ONNX + pyet (para ET₀) + duckdb (loader)
RUN pip install \
        tensorflow-cpu==2.15.0 \
        tf2onnx==1.16.1 \
        onnx==1.16.0 \
        onnxruntime==1.18.1 \
        scikit-learn==1.5.1 \
        joblib==1.4.2 \
        mlflow==2.15.1 \
        pandas==2.2.2 \
        numpy==1.26.4 \
        pyet==1.3.1 \
        duckdb==1.0.0

# Código de modelado (la "librería" pura)
COPY src/__init__.py              src/__init__.py
COPY src/common/__init__.py       src/common/__init__.py
COPY src/common/data_loader.py    src/common/data_loader.py
COPY src/moisture/__init__.py     src/moisture/__init__.py
COPY src/moisture/train.py        src/moisture/train.py
COPY src/moisture/export.py       src/moisture/export.py
COPY src/moisture/infer.py        src/moisture/infer.py
COPY src/moisture/pipeline.py     src/moisture/pipeline.py

# Sin CMD: el comando lo decide DockerOperator
