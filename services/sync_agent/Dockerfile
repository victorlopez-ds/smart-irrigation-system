# ──────────────────────────────────────────────────────────────────────────────
# sync_agent — Cliente que en el edge sincroniza modelos desde central y
# llama /reload de las APIs locales (anomaly-api, rl-api).
#
# Build (desde prod/):
#   docker build -f services/sync_agent/Dockerfile -t irridea-sync-agent .
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# tini: PID-1 limpio + cron en foreground
RUN apt-get update && apt-get install -y --no-install-recommends \
        cron \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install requests==2.32.3

COPY services/sync_agent/sync.py        ./sync.py
COPY services/sync_agent/crontab        /etc/cron.d/sync-agent
COPY services/sync_agent/entrypoint.sh  /usr/local/bin/entrypoint.sh
RUN chmod 0644 /etc/cron.d/sync-agent \
    && chmod 0755 /usr/local/bin/entrypoint.sh

VOLUME ["/app/models"]

# tini → entrypoint, que arranca cron en foreground.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
