# ──────────────────────────────────────────────────────────────────────────────
# moisture_api — Inferencia del predictor MLP de humedad (ONNX).
#
# Build (desde prod/):
#   docker build -f services/moisture_api/Dockerfile -t irridea-moisture-api .
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/moisture_api/requirements.txt .
RUN pip install -r requirements.txt

COPY services/moisture_api/app.py            ./app.py
COPY services/moisture_api/gunicorn.conf.py  ./gunicorn.conf.py

VOLUME ["/app/models"]
EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health').read()" || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
