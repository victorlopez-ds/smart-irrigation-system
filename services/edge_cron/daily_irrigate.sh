#!/bin/sh
# daily_irrigate.sh — Orquesta el ciclo diario de riego a las 06:00.
#
# 1. Pide la observación RL a ingest-svc
# 2. Envía la observación a rl-api para obtener tau_minutes
# 3. Publica el comando de válvula vía edge-mqtt-client

set -e

INGEST_URL="${INGEST_URL:-http://ingest-svc:8000}"
RL_API_URL="${RL_API_URL:-http://rl-api:8003}"
MQTT_CLIENT_URL="${MQTT_CLIENT_URL:-http://edge-mqtt-client:8010}"

echo "[daily_irrigate] Fetching RL observation..."
OBS=$(curl -sf "${INGEST_URL}/rl/obs")
echo "[daily_irrigate] obs=${OBS}"

echo "[daily_irrigate] Requesting RL action..."
ACTION=$(curl -sf -X POST -H "Content-Type: application/json" -d "${OBS}" "${RL_API_URL}/act")
echo "[daily_irrigate] action=${ACTION}"

TAU=$(echo "${ACTION}" | sed -n 's/.*"tau_minutes" *: *\([0-9.]*\).*/\1/p')
if [ -z "${TAU}" ]; then
    echo "[daily_irrigate] ERROR: could not parse tau_minutes from response"
    exit 1
fi

echo "[daily_irrigate] Publishing valve command: tau=${TAU} min"
curl -sf -X POST -H "Content-Type: application/json" \
    -d "{\"tau_minutes\": ${TAU}}" \
    "${MQTT_CLIENT_URL}/valve/command"

echo ""
echo "[daily_irrigate] Done."
