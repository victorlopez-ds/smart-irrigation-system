# ──────────────────────────────────────────────────────────────────────────────
# ingest.Dockerfile — Worker MQTT que persiste datos en DuckDB/Parquet
#
# docker build -f deployment/docker/ingest.Dockerfile -t irridea-ingest .
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install \
    paho-mqtt==2.1.0 \
    duckdb==1.0.0 \
    pandas==2.2.2 \
    python-dotenv==1.0.1 \
    onnxruntime==1.18.1 \
    joblib==1.4.2 \
    numpy==1.26.4

COPY src/__init__.py src/__init__.py
COPY src/common/__init__.py src/common/__init__.py
COPY src/common/mqtt_subscriber.py src/common/mqtt_subscriber.py
COPY src/anomaly/__init__.py src/anomaly/__init__.py
COPY src/anomaly/infer.py src/anomaly/infer.py
COPY src/anomaly/detect_window.py src/anomaly/detect_window.py

VOLUME ["/app/data"]

CMD ["python", "-m", "src.common.mqtt_subscriber"]
