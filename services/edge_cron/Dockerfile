# ──────────────────────────────────────────────────────────────────────────────
# edge_cron — Sidecar minimal con crond para disparar los jobs operacionales
# del edge: detect_anomaly cada 15 min, daily_irrigate a las 06:00, etc.
#
# En producción real esto sería el `cron` del sistema operativo de la Raspberry.
# Aquí lo encapsulamos como contenedor para reproducir el comportamiento sin
# tocar el host.
# ──────────────────────────────────────────────────────────────────────────────

FROM alpine:3.20

RUN apk add --no-cache curl tzdata

COPY services/edge_cron/crontab /etc/crontabs/root
COPY services/edge_cron/daily_irrigate.sh /app/daily_irrigate.sh
RUN chmod +x /app/daily_irrigate.sh

# crond -f -L /dev/stdout → logs por stdout para `docker logs`
CMD ["crond", "-f", "-L", "/dev/stdout"]
