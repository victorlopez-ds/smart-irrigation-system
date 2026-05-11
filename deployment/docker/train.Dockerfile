# ──────────────────────────────────────────────────────────────────────────────
# train.Dockerfile — Imagen de entrenamiento con soporte GPU (CUDA 12.1)
#
# Uso (en el servidor con GPU):
#   docker build -f deployment/docker/train.Dockerfile -t irridea-train .
#   docker run --gpus all -v $(pwd):/workspace irridea-train \
#       python -m src.rl.train --lambda_agua 1.5 --N 2
# ──────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip build-essential git \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch con CUDA 12.1 (instalado primero para aprovechar caché de capas)
RUN pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Dependencias del proyecto
COPY requirements.txt .
RUN pip install -r requirements.txt

# Dependencias de entrenamiento adicionales
RUN pip install \
    mlflow==2.15.1 \
    apache-airflow==2.9.3 \
    tf2onnx==1.16.1 \
    onnx==1.16.2 \
    onnxruntime==1.18.1 \
    onnxmltools==1.12.0 \
    duckdb==1.0.0 \
    paho-mqtt==2.1.0 \
    optuna==3.6.1 \
    pyet==1.2.2

COPY . .

# Instalar el paquete src en modo editable
RUN pip install -e . --no-deps 2>/dev/null || true

CMD ["python", "-m", "src.rl.train"]
