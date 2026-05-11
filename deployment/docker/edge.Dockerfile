# ──────────────────────────────────────────────────────────────────────────────
# edge.Dockerfile — Imagen ligera para inferencia en Raspberry Pi (ARM64)
#
# Construcción multiplataforma desde el servidor x86:
#   docker buildx build --platform linux/arm64 \
#       -f deployment/docker/edge.Dockerfile \
#       -t irridea-edge:latest --load .
#
# En la Raspberry Pi:
#   docker run --env-file .env irridea-edge
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Solo las dependencias necesarias en el edge
COPY requirements-edge.txt .
RUN pip install -r requirements-edge.txt

# Código del edge
COPY src/rl/infer.py       src/rl/infer.py
COPY src/anomaly/infer.py  src/anomaly/infer.py
COPY deployment/edge/edge_inference.py .
COPY deployment/edge/sync_models.sh .

RUN chmod +x sync_models.sh

# Los modelos se montan como volumen o se descargan con sync_models.sh
VOLUME ["/app/models"]

CMD ["python", "edge_inference.py"]
