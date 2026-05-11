# ──────────────────────────────────────────────────────────────────────────────
# rl_export.Dockerfile — Imagen one-shot para convertir el actor TQC a ONNX.
#
# No entrena nada: solo carga best_model_rl.zip, hace un warmup de
# VecNormalize en RiegoEnv, congela las stats y exporta a ONNX.
#
# Build (desde prod/):
#   docker build -f services/training/rl_export/Dockerfile -t irridea-rl-export .
#
# Uso:
#   docker run --rm \
#       -v <host>/models:/app/models \
#       -v <host>/data:/app/data:ro \
#       irridea-rl-export \
#       python -m src.rl.export \
#           --model-zip      /app/models/best_model_rl.zip \
#           --moisture-model /app/models/moisture_mlp.keras \
#           --moisture-sx    /app/models/moisture_scaler_x.pkl \
#           --moisture-sy    /app/models/moisture_scaler_y.pkl \
#           --dataset        /app/data/processed/dataset_diario.csv \
#           --output-onnx    /app/models/rl_actor.onnx
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stack: torch (CPU) + sb3-contrib + tf-keras (para cargar el moisture .keras
# que usa RiegoEnv) + ONNX
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
        pyet==1.3.1

# Código
COPY src/__init__.py              src/__init__.py
COPY src/rl/__init__.py           src/rl/__init__.py
COPY src/rl/env.py                src/rl/env.py
COPY src/rl/export.py             src/rl/export.py
