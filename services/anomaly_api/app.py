"""
services/anomaly_api/app.py — API de inferencia del detector de anomalías.

Endpoints:
  GET  /health                — liveness + estado del modelo (loaded / not_ready)
  POST /predict               — inferencia sin estado (recibe la ventana ya armada)
                                body: {"status_window":[...30 ints], "actual_vol":float}
  POST /detect_latest         — inferencia con estado: tira de ingest-svc/window,
                                arma la ventana, infiere; si anomalía, llama a
                                ingest-svc/alert + publica en MQTT irridea/alerts
  POST /reload                — recarga el modelo ONNX y el scaler desde disco
                                (lo invoca sync-agent tras descargar nueva versión)

Variables de entorno:
  INGEST_URL              http del servicio de ingesta (default http://ingest-svc:8000)
  MODEL_PATH              ruta al .onnx (default /app/models/anomaly_cnn_lstm.onnx)
  SCALER_PATH             ruta al .pkl (default /app/models/anomaly_scaler.pkl)
  ANOMALY_WINDOW          tamaño de ventana (default 30)
  ANOMALY_THRESH          umbral (default 0.25)
  MQTT_BROKER_HOST/PORT   broker para publicar alertas
  MQTT_USER/PASS          credenciales (opcionales)
  MQTT_TOPIC_ALERTS       tópico (default irridea/alerts)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
from flask import Flask, jsonify, request

import sys
sys.path.insert(0, "/app")
from src.anomaly.infer import AnomalyDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("anomaly-api")

INGEST_URL  = os.getenv("INGEST_URL", "http://ingest-svc:8000")
MODEL_PATH  = os.getenv("MODEL_PATH",  "/app/models/anomaly_cnn_lstm.onnx")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/models/anomaly_scaler.pkl")
WINDOW      = int(os.getenv("ANOMALY_WINDOW", "30"))
THRESHOLD   = float(os.getenv("ANOMALY_THRESH", "0.25"))

MQTT_HOST   = os.getenv("MQTT_BROKER_HOST", "mosquitto")
MQTT_PORT   = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USER   = os.getenv("MQTT_USER")
MQTT_PASS   = os.getenv("MQTT_PASS")
TOPIC_ALERT = os.getenv("MQTT_TOPIC_ALERTS", "irridea/alerts")


# ──────────────────────────────────────────────────────────────────────────────
# Carga / recarga del modelo
# ──────────────────────────────────────────────────────────────────────────────

class ModelHolder:
    """Mantiene la referencia al detector y permite recargarlo bajo lock."""

    def __init__(self):
        self._lock = threading.RLock()
        self._detector: AnomalyDetector | None = None
        self._version: str | None = None
        self.load()

    def load(self) -> bool:
        with self._lock:
            if not Path(MODEL_PATH).exists() or not Path(SCALER_PATH).exists():
                log.warning("Modelo o scaler no encontrados todavía (%s, %s)",
                            MODEL_PATH, SCALER_PATH)
                self._detector = None
                return False
            self._detector = AnomalyDetector(
                model_path=MODEL_PATH,
                scaler_path=SCALER_PATH,
                threshold=THRESHOLD,
                window=WINDOW,
            )
            stat = Path(MODEL_PATH).stat()
            self._version = f"{int(stat.st_mtime)}-{stat.st_size}"
            log.info("Modelo cargado (version=%s)", self._version)
            return True

    @property
    def ready(self) -> bool:
        return self._detector is not None

    @property
    def detector(self) -> AnomalyDetector:
        if self._detector is None:
            raise RuntimeError("model not loaded")
        return self._detector

    @property
    def version(self) -> str | None:
        return self._version


model = ModelHolder()


# ──────────────────────────────────────────────────────────────────────────────
# Cliente MQTT (solo publica)
# ──────────────────────────────────────────────────────────────────────────────

import paho.mqtt.client as mqtt

_mqtt_client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=f"anomaly-api-{os.getpid()}",
)
if MQTT_USER:
    _mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
_mqtt_client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
_mqtt_client.loop_start()


def publish_alert(payload: dict) -> None:
    info = _mqtt_client.publish(TOPIC_ALERT, json.dumps(payload), qos=1)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        log.warning("Fallo publicando alerta MQTT (rc=%s)", info.rc)


# ──────────────────────────────────────────────────────────────────────────────
# Lógica de inferencia
# ──────────────────────────────────────────────────────────────────────────────

def fetch_window_from_ingest() -> dict:
    """
    Pide a ingest-svc:
      • últimas WINDOW filas de raw_valve (status)
      • últimas 2 filas de raw_flow para calcular vol_diff actual
    Devuelve {status_window, last_ts, actual_vol}.
    """
    r = requests.get(
        f"{INGEST_URL}/window",
        params={"topic": "valve", "n": WINDOW},
        timeout=5,
    )
    r.raise_for_status()
    valve_rows = r.json()["rows"]
    if len(valve_rows) < WINDOW:
        raise ValueError(f"insufficient valve history ({len(valve_rows)}/{WINDOW})")

    status_window = np.array([row["status"] for row in valve_rows],
                             dtype=np.float32).reshape(WINDOW, 1)
    last_ts = valve_rows[-1]["ts"]

    # vol_diff actual = diferencia entre las dos últimas lecturas crudas
    r = requests.get(
        f"{INGEST_URL}/window",
        params={"topic": "flow", "n": 2},
        timeout=5,
    )
    r.raise_for_status()
    flow_rows = r.json()["rows"]
    if len(flow_rows) < 2:
        raise ValueError("need at least 2 flow rows to compute vol_diff")

    actual_vol = float(flow_rows[-1]["vol_l"]) - float(flow_rows[-2]["vol_l"])
    return {
        "status_window": status_window,
        "last_ts": last_ts,
        "actual_vol": actual_vol,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(
        status="ok" if model.ready else "not_ready",
        model_version=model.version,
        model_path=MODEL_PATH,
    )


@app.post("/predict")
def predict():
    if not model.ready:
        return jsonify(error="model not loaded"), 503
    body = request.get_json(force=True)
    sw = np.array(body["status_window"], dtype=np.float32).reshape(WINDOW, 1)
    actual = float(body["actual_vol"])
    actual_scaled = float(model.detector.scaler.transform([[actual]])[0, 0])
    result = model.detector.detect(sw, actual_scaled)
    return jsonify(asdict(result))


@app.post("/detect_latest")
def detect_latest():
    if not model.ready:
        return jsonify(status="skipped", reason="model not loaded"), 503

    try:
        ctx = fetch_window_from_ingest()
    except Exception as exc:
        log.warning("No se pudo obtener ventana: %s", exc)
        return jsonify(status="skipped", reason=str(exc)), 200

    actual_scaled = float(
        model.detector.scaler.transform([[ctx["actual_vol"]]])[0, 0]
    )
    result = model.detector.detect(ctx["status_window"], actual_scaled)

    payload = {
        "ts":          ctx["last_ts"],
        "is_anomaly":  result.is_anomaly,
        "error":       result.error,
        "threshold":   result.threshold,
        "predicted_l": result.predicted_volume,
        "actual_l":    result.actual_volume,
    }

    if result.is_anomaly:
        # Persistir en DuckDB vía ingest-svc
        try:
            requests.post(f"{INGEST_URL}/alert", json={
                "ts":          payload["ts"],
                "error":       payload["error"],
                "threshold":   payload["threshold"],
                "predicted_l": payload["predicted_l"],
                "actual_l":    payload["actual_l"],
            }, timeout=5).raise_for_status()
        except Exception:
            log.exception("Fallo registrando alerta en ingest-svc")
        # Publicar en MQTT
        publish_alert(payload)
        log.warning("⚠️ Anomalía: ts=%s error=%.4f", payload["ts"], payload["error"])
        payload["status"] = "ANOMALY"
    else:
        payload["status"] = "ok"

    return jsonify(payload)


@app.post("/reload")
def reload_model():
    ok = model.load()
    return jsonify(
        status="ok" if ok else "not_ready",
        model_version=model.version,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
