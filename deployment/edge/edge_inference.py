"""
deployment/edge/edge_inference.py — Proceso de inferencia diaria en la Raspberry Pi.

Ejecutado por un systemd timer a las 06:00 cada día.

Flujo:
  1. Lee la humedad actual del suelo (último valor en DuckDB o MQTT)
  2. Consulta la API meteorológica para obtener ET y precipitación (N=2 días)
  3. Construye la observación del agente RL
  4. Llama al actor TorchScript para obtener τ (minutos de riego)
  5. Publica la acción a la electroválvula vía MQTT
  6. Ejecuta el detector de anomalías sobre las últimas 24h de caudal
  7. Publica alerta si se detecta anomalía

Variables de entorno:
  MLFLOW_TRACKING_URI  URL del servidor MLflow
  MQTT_BROKER_HOST     Broker Mosquitto
  MQTT_BROKER_PORT     Puerto del broker (por defecto 1883)
  MQTT_TOPIC_VALVE_CMD Tópico de comando de la electroválvula
  MQTT_TOPIC_ALERT     Tópico de alertas de anomalía
  WEATHER_API_KEY      Clave de la API meteorológica (OpenWeatherMap)
  WEATHER_LAT          Latitud de la parcela
  WEATHER_LON          Longitud de la parcela
  MODELS_DIR           Directorio local con los modelos descargados
  DATA_DIR             Directorio con la base de datos DuckDB
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import paho.mqtt.client as mqtt
import requests
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuración ─────────────────────────────────────────────────────────────

MQTT_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
TOPIC_VALVE_CMD = os.getenv("MQTT_TOPIC_VALVE_CMD", "irridea/valve/cmd")
TOPIC_ALERT = os.getenv("MQTT_TOPIC_ALERT", "irridea/alerts")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
LAT = float(os.getenv("WEATHER_LAT", "38.028"))
LON = float(os.getenv("WEATHER_LON", "-1.490"))

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
DB_PATH = DATA_DIR / "irridea.duckdb"

ACTOR_PATH = MODELS_DIR / "tqc_actor.pt"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_cnn_lstm.onnx"
ANOMALY_SCALER_PATH = MODELS_DIR / "anomaly_scaler.pkl"
VECNORM_STATS_PATH = MODELS_DIR / "vecnorm_stats.npz"

N = 2               # horizonte de previsión (debe coincidir con entrenamiento)
TAU_MAX = 180.0     # minutos máximos de riego
W_MAX = 2500.0      # litros máximos
ANOMALY_WINDOW = 30 # ventanas de 15 min = 7.5 horas


# ── Obtención de datos meteorológicos ────────────────────────────────────────

def get_weather_forecast(n: int = 2) -> tuple[list[float], list[float]]:
    """
    Obtiene previsión de ET₀ y precipitación para los próximos n días.
    Fuente: OpenWeatherMap API 3.0 One Call.

    Returns (et_forecast, pt_forecast) — listas de longitud n
    """
    if not WEATHER_API_KEY:
        log.warning("WEATHER_API_KEY no configurada. Usando valores por defecto.")
        return [3.0] * n, [0.0] * n

    url = (
        f"https://api.openweathermap.org/data/3.0/onecall"
        f"?lat={LAT}&lon={LON}&exclude=current,minutely,hourly,alerts"
        f"&appid={WEATHER_API_KEY}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", [])[:n]

        et_list, pt_list = [], []
        for day in daily:
            # ET₀ simplificada (Penman-Monteith requiere más variables;
            # aquí usamos estimación FAO-56 simplificada basada en Tmáx)
            tmax = day.get("temp", {}).get("max", 25.0)
            tmin = day.get("temp", {}).get("min", 15.0)
            et_approx = 0.0023 * (tmax + tmin) / 2 * ((tmax - tmin) ** 0.5) * 16
            et_list.append(max(et_approx, 0.0))
            pt_list.append(float(day.get("rain", 0.0)))

        return et_list, pt_list

    except Exception as exc:
        log.error("Error obteniendo previsión meteorológica: %s. Usando defaults.", exc)
        return [3.0] * n, [0.0] * n


# ── Lectura del estado del suelo ──────────────────────────────────────────────

def get_current_state() -> tuple[float, float]:
    """
    Lee del DuckDB la humedad actual del suelo y el agua de riego de ayer.

    Returns (theta_t, w_prev)
    """
    if not DB_PATH.exists():
        log.warning("Base de datos no encontrada. Usando valores por defecto.")
        return 30.0, 0.0

    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        row = con.execute(
            "SELECT theta, water_used_l FROM features_daily ORDER BY date DESC LIMIT 1"
        ).fetchone()
        con.close()
        if row:
            return float(row[0] or 30.0), float(row[1] or 0.0)
    except Exception as exc:
        log.error("Error leyendo DuckDB: %s", exc)

    return 30.0, 0.0


# ── Inferencia RL ─────────────────────────────────────────────────────────────

def predict_irrigation(
    theta_t: float,
    w_prev: float,
    et_forecast: list[float],
    pt_forecast: list[float],
) -> float:
    """
    Construye la observación y llama al actor TorchScript.

    Returns τ (minutos de riego) ∈ [0, TAU_MAX]
    """
    obs = np.array(
        [theta_t, w_prev] + list(et_forecast) + list(pt_forecast),
        dtype=np.float32,
    )

    # Normalizar con las estadísticas VecNormalize guardadas
    if VECNORM_STATS_PATH.exists():
        stats = np.load(VECNORM_STATS_PATH)
        mean = stats["obs_mean"]
        var = stats["obs_var"]
        obs_norm = (obs - mean) / np.sqrt(var + 1e-8)
        obs_norm = np.clip(obs_norm, -10.0, 10.0)
    else:
        log.warning("Estadísticas VecNormalize no encontradas. Usando obs sin normalizar.")
        obs_norm = obs

    actor = torch.jit.load(str(ACTOR_PATH), map_location="cpu")
    actor.eval()

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs_norm).unsqueeze(0)
        action = actor(obs_tensor)
        tau = float(action[0, 0].item())

    return float(np.clip(tau, 0.0, TAU_MAX))


# ── Detección de anomalías ────────────────────────────────────────────────────

def check_anomaly() -> bool:
    """
    Ejecuta el detector CNN-LSTM sobre las últimas ANOMALY_WINDOW lecturas de caudal.

    Returns True si se detecta anomalía.
    """
    if not ANOMALY_MODEL_PATH.exists():
        log.warning("Modelo de anomalías no encontrado. Saltando detección.")
        return False

    try:
        import onnxruntime as ort
        import joblib
        from sklearn.preprocessing import StandardScaler

        con = duckdb.connect(str(DB_PATH), read_only=True)
        rows = con.execute(
            f"SELECT status, vol_diff FROM raw_valve v "
            f"JOIN raw_flow f ON v.ts = f.ts "
            f"ORDER BY v.ts DESC LIMIT {ANOMALY_WINDOW + 1}"
        ).fetchall()
        con.close()

        if len(rows) < ANOMALY_WINDOW:
            log.info("Datos insuficientes para detección de anomalías.")
            return False

        status_arr = np.array([r[0] for r in rows[::-1]], dtype=np.float32)
        vol_arr = np.array([r[1] for r in rows[::-1]], dtype=np.float32)

        scaler: StandardScaler = joblib.load(ANOMALY_SCALER_PATH)
        vol_scaled = scaler.transform(vol_arr.reshape(-1, 1)).flatten()

        window_status = status_arr[-ANOMALY_WINDOW:].reshape(1, ANOMALY_WINDOW, 1)
        actual_vol = vol_scaled[-1]

        session = ort.InferenceSession(str(ANOMALY_MODEL_PATH))
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: window_status})[0][0, 0]

        error = abs(pred - actual_vol)
        threshold = 0.25
        is_anomaly = bool(error > threshold)

        log.info(
            "Detección anomalías: error=%.4f threshold=%.4f anomalía=%s",
            error, threshold, is_anomaly,
        )
        return is_anomaly

    except Exception as exc:
        log.error("Error en detección de anomalías: %s", exc)
        return False


# ── Publicación MQTT ──────────────────────────────────────────────────────────

def publish_mqtt(topic: str, payload: dict) -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.publish(topic, json.dumps(payload), qos=1)
    client.disconnect()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now = datetime.now(tz=timezone.utc).isoformat()
    log.info("=== Inferencia diaria %s ===", now)

    # 1. Estado actual
    theta_t, w_prev = get_current_state()
    log.info("Estado: θ=%.1f%%  w_prev=%.0f L", theta_t, w_prev)

    # 2. Previsión meteorológica
    et_fc, pt_fc = get_weather_forecast(N)
    log.info("Previsión: ET=%s  PT=%s", et_fc, pt_fc)

    # 3. Decisión de riego
    tau = predict_irrigation(theta_t, w_prev, et_fc, pt_fc)
    agua_litros = tau * 12.57  # q = 12.57 L/min
    log.info("Decisión: τ=%.1f min  →  %.0f litros", tau, agua_litros)

    # 4. Publicar comando a la electroválvula
    publish_mqtt(TOPIC_VALVE_CMD, {
        "TimeInstant": now,
        "tau_minutes": tau,
        "water_litros": agua_litros,
        "theta_actual": theta_t,
    })
    log.info("Comando enviado a %s", TOPIC_VALVE_CMD)

    # 5. Detección de anomalías
    is_anomaly = check_anomaly()
    if is_anomaly:
        log.warning("⚠️  ANOMALÍA HIDRÁULICA DETECTADA")
        publish_mqtt(TOPIC_ALERT, {
            "TimeInstant": now,
            "type": "hydraulic_anomaly",
            "message": "Caudal anómalo detectado por CNN-LSTM",
        })

    log.info("=== Inferencia completada ===")


if __name__ == "__main__":
    main()
