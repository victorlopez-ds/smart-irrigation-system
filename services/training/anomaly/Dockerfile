# ──────────────────────────────────────────────────────────────────────────────
# train_anomaly.Dockerfile — Imagen efímera para reentrenar el detector CNN-LSTM.
#
# Build (desde prod/):
#   docker build -f services/training/anomaly/Dockerfile -t irridea-train-anomaly .
#
# Uso (lo invoca DockerOperator desde Airflow):
#   docker run --rm \
#       --network deployment_default \
#       -v <host>/models:/app/models \
#       -v <host>/data:/app/data:ro \
#       -v mlflow-artifacts:/mlflow-artifacts \
#       -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
#       irridea-train-anomaly \
#       python -m src.anomaly.pipeline --raw-watermeter /app/data/raw/...
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias pesadas (TF + ecosistema). Se instala una sola vez al construir
# la imagen, no en cada ejecución del DAG.
RUN pip install \
        tensorflow-cpu==2.15.0 \
        tf2onnx==1.16.1 \
        onnx==1.16.0 \
        onnxruntime==1.18.1 \
        scikit-learn==1.5.1 \
        joblib==1.4.2 \
        mlflow==2.15.1 \
        pandas==2.2.2 \
        numpy==1.26.4

# Código de modelado (la "librería" pura)
COPY src/__init__.py             src/__init__.py
COPY src/anomaly/__init__.py     src/anomaly/__init__.py
COPY src/anomaly/train.py        src/anomaly/train.py
COPY src/anomaly/export.py       src/anomaly/export.py
COPY src/anomaly/infer.py        src/anomaly/infer.py
COPY src/anomaly/pipeline.py     src/anomaly/pipeline.py

# Sin CMD: el comando lo decide DockerOperator
