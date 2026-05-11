"""
services/edge_mqtt_client/app.py — Cliente MQTT del edge.

Suscribe a los topics de telemetría del broker mosquitto, acumula las
lecturas en buffers en RAM y dispara la detección de anomalías cuando
la ventana de caudal está completa.

También expone un endpoint HTTP para que edge-cron publique comandos
de riego al broker (irridea/valve).

Endpoints:
  GET  /health
  POST /valve/command   — body: {"tau_minutes": float}
                          Publica en irridea/valve

Variables de entorno:
  MQTT_BROKER_HOST       host del broker (default: mosquitto)
  MQTT_BROKER_PORT       puerto (default: 1883)
  MQTT_USER, MQTT_PASS   credenciales
  MQTT_TOPIC_FLOW        topic de caudal   (default: irridea/flow)
  MQTT_TOPIC_SOIL        topic de humedad  (default: irridea/soil)
  MQTT_TOPIC_VALVE       topic de válvula  (default: irridea/valve)
  ANOMALY_API_URL        URL de anomaly-api (default: http://anomaly-api:8001)
  ANOMALY_WINDOW         tamaño de ventana (default: 30)
  INGEST_URL             URL de ingest-svc para obtener vol_diff actual
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque

import paho.mqtt.client as mqtt
import requests
from flask import Flask, jsonify, request as flask_request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("edge-mqtt-client")

# ── Config ────────────────────────────────────────────────────────────────────

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mosquitto")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USER   = os.getenv("MQTT_USER")
MQTT_PASS   = os.getenv("MQTT_PASS")

TOPIC_FLOW  = os.getenv("MQTT_TOPIC_FLOW",  "irridea/flow")
TOPIC_SOIL  = os.getenv("MQTT_TOPIC_SOIL",  "irridea/soil")
TOPIC_VALVE = os.getenv("MQTT_TOPIC_VALVE", "irridea/valve")

ANOMALY_API = os.getenv("ANOMALY_API_URL", "http://anomaly-api:8001")
INGEST_URL  = os.getenv("INGEST_URL", "http://ingest-svc:8000")
WINDOW_SIZE = int(os.getenv("ANOMALY_WINDOW", "30"))

# ── Buffers en RAM ────────────────────────────────────────────────────────────

_flow_buffer: deque[dict] = deque(maxlen=WINDOW_SIZE)
_buffer_lock = threading.Lock()


def _try_detect() -> None:
    """Si el buffer de caudal está lleno, llama a anomaly-api /detect_latest."""
    with _buffer_lock:
        if len(_flow_buffer) < WINDOW_SIZE:
            return

    try:
        r = requests.post(
            f"{ANOMALY_API}/detect_latest",
            timeout=10,
        )
        if r.status_code == 200:
            result = r.json()
            if result.get("status") == "ANOMALY":
                log.warning("Anomaly detected: %s", result)
            else:
                log.info("Detection OK: %s", result.get("status"))
        else:
            log.warning("anomaly-api responded %s", r.status_code)
    except Exception:
        log.exception("Failed calling anomaly-api /detect_latest")


# ── MQTT ──────────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("Connected to MQTT %s:%s", BROKER_HOST, BROKER_PORT)
        client.subscribe(TOPIC_FLOW, qos=1)
        client.subscribe(TOPIC_SOIL, qos=1)
        log.info("Subscribed to %s, %s", TOPIC_FLOW, TOPIC_SOIL)
    else:
        log.error("MQTT connection failed (rc=%s)", rc)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        log.exception("Failed parsing MQTT message on %s", msg.topic)
        return

    if msg.topic == TOPIC_FLOW:
        with _buffer_lock:
            _flow_buffer.append(payload)
        log.debug("Flow buffer: %d/%d", len(_flow_buffer), WINDOW_SIZE)
        _try_detect()

    elif msg.topic == TOPIC_SOIL:
        log.debug("Soil reading received: %s", payload)


_mqtt_client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=f"edge-mqtt-client-{os.getpid()}",
    clean_session=False,
)
if MQTT_USER:
    _mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
_mqtt_client.on_connect = on_connect
_mqtt_client.on_message = on_message
_mqtt_client.connect_async(BROKER_HOST, BROKER_PORT, keepalive=60)
_mqtt_client.loop_start()

# ── Flask ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/health")
def health():
    with _buffer_lock:
        buf_len = len(_flow_buffer)
    return jsonify(
        status="ok",
        mqtt_connected=_mqtt_client.is_connected(),
        flow_buffer_size=buf_len,
        window_size=WINDOW_SIZE,
    )


@app.post("/valve/command")
def valve_command():
    """
    Recibe una orden de riego de edge-cron y la publica en MQTT.

    Body: {"tau_minutes": float}
    """
    body = flask_request.get_json(force=True)
    tau = float(body.get("tau_minutes", 0))

    payload = json.dumps({
        "command": "irrigate",
        "tau_minutes": tau,
    })
    info = _mqtt_client.publish(TOPIC_VALVE, payload, qos=1)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        log.error("Failed publishing valve command (rc=%s)", info.rc)
        return jsonify(status="error", rc=info.rc), 500

    log.info("Valve command published: tau=%.1f min", tau)
    return jsonify(status="published", tau_minutes=tau, topic=TOPIC_VALVE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010)
