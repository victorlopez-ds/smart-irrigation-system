# ──────────────────────────────────────────────────────────────────────────────
# rl_finetune.Dockerfile — Imagen para fine-tune online del agente TQC.
#
# Extiende el stack de rl_export con:
#   - mlflow      (logging del run)
#   - pyarrow     (lectura del parquet de transiciones reales)
#   - requests    (descarga del parquet desde ingest-svc /rl/export)
#
# Build (desde prod/):
#   docker build -f services/training/rl_finetune/Dockerfile -t irridea-rl-finetune .
#
# El DAG finetune_rl invoca:
#   python -m src.rl.finetune \
#       --base-model     /app/models/best_model_rl.zip \
#       --real-parquet   /app/data/tmp/real_transitions.parquet \
#       ...
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install \
        torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install \
        sb3-contrib==2.3.0 \
        stable-baselines3==2.3.0 \
        gymnasium==0.29.1 \
        tf-keras==2.15.0 \
        tensorflow-cpu==2.15.0 \
        onnx==1.16.0 \
        onnxruntime==1.18.1 \
        scikit-learn==1.5.1 \
        joblib==1.4.2 \
        pandas==2.2.2 \
        numpy==1.26.4 \
        pyet==1.3.1 \
        pyarrow==15.0.2 \
        mlflow==2.15.1 \
        requests==2.32.3

# Código RL
COPY src/__init__.py              src/__init__.py
COPY src/rl/__init__.py           src/rl/__init__.py
COPY src/rl/env.py                src/rl/env.py
COPY src/rl/reward.py             src/rl/reward.py
COPY src/rl/export.py             src/rl/export.py
COPY src/rl/finetune.py           src/rl/finetune.py
