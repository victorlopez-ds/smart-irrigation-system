# ──────────────────────────────────────────────────────────────────────────────
# model_distributor — Sirve los modelos del filesystem por HTTP al edge.
#
# Build (desde prod/):
#   docker build -f services/model_distributor/Dockerfile -t irridea-model-distributor .
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/model_distributor/requirements.txt .
RUN pip install -r requirements.txt

COPY services/model_distributor/app.py            ./app.py
COPY services/model_distributor/gunicorn.conf.py  ./gunicorn.conf.py

VOLUME ["/app/models"]
EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8004/health').read()" || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
