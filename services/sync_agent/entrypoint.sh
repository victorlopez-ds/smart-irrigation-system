#!/bin/sh
set -eu

# Volcar el entorno actual (las env vars de docker compose) a un fichero que
# cron pueda "source"-ar. cron arranca cada job sin las env vars del PID 1.
{
    for var in CENTRAL_URL MODELS_DIR HTTP_TIMEOUT GROUPS_JSON SYNC_TOKEN; do
        eval "val=\${$var:-}"
        if [ -n "$val" ]; then
            # escapar comillas simples
            esc=$(printf "%s" "$val" | sed "s/'/'\\\\''/g")
            echo "export $var='$esc'"
        fi
    done
} > /etc/sync_env
chmod 0644 /etc/sync_env

# Sincronizar una vez al arrancar (no esperar al primer cron tick).
echo "[entrypoint] sync inicial…"
. /etc/sync_env
python /app/sync.py || echo "[entrypoint] sync inicial terminó con errores"

echo "[entrypoint] arrancando cron en foreground"
exec cron -f
