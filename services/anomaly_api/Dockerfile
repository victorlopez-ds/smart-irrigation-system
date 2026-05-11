# ──────────────────────────────────────────────────────────────────────────────
# anomaly_api — Inferencia del detector de anomalías hidráulicas (ONNX).
#
# Build (desde prod/):
#   docker build -f services/anomaly_api/Dockerfile -t irridea-anomaly-api .
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/anomaly_api/requirements.txt .
RUN pip install -r requirements.txt

# La librería pura (src/) la importa la API
COPY src/__init__.py             src/__init__.py
COPY src/anomaly/__init__.py     src/anomaly/__init__.py
COPY src/anomaly/infer.py        src/anomaly/infer.py

COPY services/anomaly_api/app.py            ./app.py
COPY services/anomaly_api/gunicorn.conf.py  ./gunicorn.conf.py

VOLUME ["/app/models"]
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health').read()" || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
