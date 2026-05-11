# ──────────────────────────────────────────────────────────────────────────────
# rl_api — Inferencia del agente TQC (ONNX, normalización + actor + unscaling
# ya empaquetados en el grafo).
#
# Build (desde prod/):
#   docker build -f services/rl_api/Dockerfile -t irridea-rl-api .
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/rl_api/requirements.txt .
RUN pip install -r requirements.txt

COPY services/rl_api/app.py            ./app.py
COPY services/rl_api/gunicorn.conf.py  ./gunicorn.conf.py

VOLUME ["/app/models"]
EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health').read()" || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
