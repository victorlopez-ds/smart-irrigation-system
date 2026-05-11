# ──────────────────────────────────────────────────────────────────────────────
# ingest_svc — API Flask + suscriptor MQTT + buffer rotatorio DuckDB
#
# Build (desde prod/):
#   docker build -f services/ingest_svc/Dockerfile -t irridea-ingest-svc .
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/ingest_svc/requirements.txt .
RUN pip install -r requirements.txt

COPY services/ingest_svc/app.py             ./app.py
COPY services/ingest_svc/gunicorn.conf.py   ./gunicorn.conf.py

# Función de recompensa compartida (single source of truth con RiegoEnv).
# La importa /rl/build para calcular reward de transiciones reales.
COPY src/rl/reward.py                        ./src_repo/rl/reward.py
RUN touch ./src_repo/__init__.py ./src_repo/rl/__init__.py
ENV REPO_ROOT=/app/src_repo

VOLUME ["/app/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
