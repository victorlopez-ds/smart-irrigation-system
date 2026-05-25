"""
services/edge_mqtt_client/app.py — Cliente MQTT del edge.

Suscribe al broker wiclouds/ODINS (topic /{APIKEY}/{SERIAL}/attrs),
acumula las lecturas en buffers en RAM y dispara la detección de
anomalías cuando la ventana de caudal está completa.

Protocolo wiclouds ODINS:
  - Topic de entrada/salida: /{APIKEY}/{SERIAL}/attrs
  - Entrada: {"cnt24": 1234.56, "ev40": 1, ...}
    • cnt* → flow (volumen acumulado)
    • ev*  → valve (0/1)
  - Salida (alertas):  {"anomaly_alert": 1, "hist": [...], "_source": "irridea"}
  - Salida (riego):    {"tau_minutes": 60, "hist": [...], "_source": "irridea"}

Flujo de detección (según diagrama de arquitectura):
  1. Recibe lecturas MQTT cnt*/ev* → las acumula en deques locales
  2. Cuando ambos buffers tienen >= WINDOW lecturas:
     - Construye status_window (últimos WINDOW estados de válvula)
     - Calcula vol_diff (diferencia entre las dos últimas lecturas de flow)
     - Llama a POST /predict de anomaly-api con la ventana
  3. Si anomaly-api devuelve anomalía → la persiste en ingest-svc y publica
     alerta MQTT en formato wiclouds

También expone un endpoint HTTP para que edge-cron publique comandos
de riego al broker en formato wiclouds.

Endpoints:
  GET  /health
  POST /valve/command   — body: {"tau_minutes": float}

Variables de entorno:
  MQTT_BROKER_HOST       host del broker (default: mosquitto)
  MQTT_BROKER_PORT       puerto (default: 1883)
  MQTT_USER, MQTT_PASS   credenciales
  WICLOUDS_APIKEY        API key ODINS (default: odins)
  WICLOUDS_SERIAL        serial del dispositivo (required for publishing)
  ANOMALY_API_URL        URL de anomaly-api (default: http://anomaly-api:8001)
  ANOMALY_WINDOW         tamaño de ventana (default: 30)
  INGEST_URL             URL de ingest-svc (para persistir alertas)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
import time
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

# ── Protocolo wiclouds ODINS ─────────────────────────────────────────────────
WICLOUDS_APIKEY = os.getenv("WICLOUDS_APIKEY", "odins")
WICLOUDS_SERIAL = os.getenv("WICLOUDS_SERIAL", "")  # serial del dispositivo
WICLOUDS_TOPIC  = f"/{WICLOUDS_APIKEY}/+/attrs"      # suscripción con wildcard

RE_CNT = re.compile(r"^cnt", re.IGNORECASE)
RE_EV  = re.compile(r"^ev",  re.IGNORECASE)

ANOMALY_API = os.getenv("ANOMALY_API_URL", "http://anomaly-api:8001")
INGEST_URL  = os.getenv("INGEST_URL", "http://ingest-svc:8000")
WINDOW_SIZE = int(os.getenv("ANOMALY_WINDOW", "30"))

# ── Buffers en RAM ────────────────────────────────────────────────────────────
# El diagrama especifica: deque(maxlen=32) en RAM
BUFFER_LEN = max(WINDOW_SIZE + 2, 32)  # margen para calcular vol_diff

_flow_buffer:  deque[dict] = deque(maxlen=BUFFER_LEN)
_valve_buffer: deque[dict] = deque(maxlen=BUFFER_LEN)
_buffer_lock = threading.Lock()

# Serial del último mensaje recibido (para saber a qué dispositivo publicar)
_last_serial: str = WICLOUDS_SERIAL


def _serial_from_topic(topic: str) -> str | None:
    """Extrae el SERIAL de un topic /{APIKEY}/{SERIAL}/attrs."""
    parts = topic.strip("/").split("/")
    return parts[1] if len(parts) >= 3 else None


def _wiclouds_publish_topic(serial: str | None = None) -> str:
    """Construye el topic de publicación para un dispositivo."""
    s = serial or _last_serial or WICLOUDS_SERIAL
    return f"/{WICLOUDS_APIKEY}/{s}/attrs"


def _try_detect() -> None:
    """
    Si los buffers de flow y valve tienen suficientes lecturas,
    construye la ventana localmente y llama a POST /predict de anomaly-api.
    """
    with _buffer_lock:
        if len(_valve_buffer) < WINDOW_SIZE:
            log.debug("Valve buffer insuficiente: %d/%d",
                      len(_valve_buffer), WINDOW_SIZE)
            return
        if len(_flow_buffer) < 2:
            log.debug("Flow buffer insuficiente: %d/2", len(_flow_buffer))
            return

        # Construir ventana de status de válvula (últimos WINDOW valores)
        valve_list = list(_valve_buffer)
        status_window = [v.get("status", 0) for v in valve_list[-WINDOW_SIZE:]]

        # vol_diff = diferencia entre las dos últimas lecturas de flow
        flow_list = list(_flow_buffer)
        vol_prev = float(flow_list[-2].get("vol", 0))
        vol_curr = float(flow_list[-1].get("vol", 0))
        actual_vol = vol_curr - vol_prev

    # Llamar a anomaly-api /predict con la ventana local (fuera del lock)
    try:
        r = requests.post(
            f"{ANOMALY_API}/predict",
            json={
                "status_window": status_window,
                "actual_vol": actual_vol,
            },
            timeout=10,
        )
        if r.status_code == 200:
            result = r.json()
            if result.get("is_anomaly"):
                log.warning("Anomalía detectada: error=%.4f actual=%.2f",
                            result.get("error", 0), actual_vol)
                _persist_alert(result, actual_vol)
            else:
                log.info("Detección OK: error=%.4f (umbral=%.2f)",
                         result.get("error", 0), result.get("threshold", 0))
        elif r.status_code == 503:
            log.info("anomaly-api no lista (modelo no cargado)")
        else:
            log.warning("anomaly-api respondió %s: %s", r.status_code, r.text)
    except Exception:
        log.exception("Fallo llamando a anomaly-api /predict")


def _persist_alert(result: dict, actual_vol: float) -> None:
    """Persiste la alerta en ingest-svc y publica en MQTT (formato wiclouds)."""
    alert = {
        "ts":          result.get("ts") or dt.datetime.now(dt.timezone.utc).isoformat(),
        "error":       result.get("error"),
        "threshold":   result.get("threshold"),
        "predicted_l": result.get("predicted_volume"),
        "actual_l":    actual_vol,
    }
    # Persistir en DuckDB vía ingest-svc
    try:
        requests.post(
            f"{INGEST_URL}/alert", json=alert, timeout=5,
        ).raise_for_status()
    except Exception:
        log.exception("Fallo persistiendo alerta en ingest-svc")

    # Publicar en MQTT en formato wiclouds
    now_ts = int(time.time())
    wiclouds_payload = {
        "anomaly_alert": 1,
        "anomaly_error": round(float(result.get("error", 0)), 4),
        "hist": [
            {"n": "anomaly_alert", "t": now_ts, "v": 1},
            {"n": "anomaly_error", "t": now_ts, "v": round(float(result.get("error", 0)), 4)},
        ],
        "_source": "irridea",
    }
    try:
        topic = _wiclouds_publish_topic()
        _mqtt_client.publish(topic, json.dumps(wiclouds_payload), qos=1)
        log.info("Alerta publicada en %s", topic)
    except Exception:
        log.exception("Fallo publicando alerta MQTT")


# ── MQTT ──────────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("Connected to MQTT %s:%s", BROKER_HOST, BROKER_PORT)
        client.subscribe(WICLOUDS_TOPIC, qos=1)
        log.info("Subscribed to %s", WICLOUDS_TOPIC)
    else:
        log.error("MQTT connection failed (rc=%s)", rc)


def on_message(client, userdata, msg):
    """
    Procesa mensajes wiclouds: /{APIKEY}/{SERIAL}/attrs
    cnt* → flow buffer, ev* → valve buffer.
    Ignora mensajes con _source=irridea (publicados por nosotros).
    """
    global _last_serial
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        log.exception("Failed parsing MQTT message on %s", msg.topic)
        return

    # Ignorar mensajes publicados por nosotros
    if payload.get("_source") == "irridea":
        return

    serial = _serial_from_topic(msg.topic)
    if not serial:
        return
    _last_serial = serial

    got_flow = False
    for key, value in payload.items():
        if key in ("hist", "_source"):
            continue

        if RE_CNT.match(key):
            with _buffer_lock:
                _flow_buffer.append({"vol": float(value)})
            log.debug("[%s] %s=%.2f → flow buffer (%d/%d)",
                      serial, key, float(value), len(_flow_buffer), BUFFER_LEN)
            got_flow = True

        elif RE_EV.match(key):
            with _buffer_lock:
                _valve_buffer.append({"status": int(value)})
            log.debug("[%s] %s=%d → valve buffer (%d/%d)",
                      serial, key, int(value), len(_valve_buffer), BUFFER_LEN)

    if got_flow:
        _try_detect()


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
        flow_len = len(_flow_buffer)
        valve_len = len(_valve_buffer)
    return jsonify(
        status="ok",
        mqtt_connected=_mqtt_client.is_connected(),
        flow_buffer_size=flow_len,
        valve_buffer_size=valve_len,
        window_size=WINDOW_SIZE,
    )


@app.post("/valve/command")
def valve_command():
    """
    Recibe una orden de riego de edge-cron y la publica en MQTT
    en formato wiclouds.

    Body: {"tau_minutes": float}
    """
    body = flask_request.get_json(force=True)
    tau = float(body.get("tau_minutes", 0))

    now_ts = int(time.time())
    wiclouds_payload = json.dumps({
        "tau_minutes": tau,
        "hist": [
            {"n": "tau_minutes", "t": now_ts, "v": tau},
        ],
        "_source": "irridea",
    })

    topic = _wiclouds_publish_topic()
    info = _mqtt_client.publish(topic, wiclouds_payload, qos=1)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        log.error("Failed publishing valve command (rc=%s)", info.rc)
        return jsonify(status="error", rc=info.rc), 500

    log.info("Valve command published: tau=%.1f min → %s", tau, topic)
    return jsonify(status="published", tau_minutes=tau, topic=topic)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010)
