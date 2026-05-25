"""
services/sensor_simulator/app.py — Simulador de sensores IoT.

Emula el comportamiento de los sensores reales de una parcela de riego
publicando datos en formato wiclouds ODINS:
  - Topic: /{APIKEY}/{SERIAL}/attrs
  - Payload: {"cnt24": vol_acum, "ev40": 0|1, "hist": [...]}
  - cnt* = volumen acumulado (caudalímetro)
  - ev*  = estado de electroválvula (0/1)

El modelo físico simplificado:
  - La humedad del suelo sube con el riego y la lluvia, baja con la ET0.
  - El caudal acumula litros cuando la válvula está abierta (Q_LH l/h).
  - La temperatura del suelo sigue un ciclo sinusoidal diario.
  - Periódicamente se inyecta una anomalía de caudal (fuga simulada).

Variables de entorno:
  MQTT_BROKER_HOST       host del broker (default: mosquitto)
  MQTT_BROKER_PORT       puerto (default: 1883)
  MQTT_USER, MQTT_PASS   credenciales
  WICLOUDS_APIKEY        API key ODINS (default: odins)
  WICLOUDS_SERIAL        serial del dispositivo (default: IA0301G00000000004)
  SIM_ATTR_FLOW          nombre del atributo de caudal (default: cnt24)
  SIM_ATTR_VALVE         nombre del atributo de válvula (default: ev40)
  SIM_INTERVAL_S         intervalo en segundos (default: 900 = 15 min)
  SIM_ANOMALY_PROB       probabilidad de anomalía por tick (default: 0.02)
  SIM_INITIAL_MOISTURE   humedad inicial % (default: 25.0)
  SIM_Q_LH              caudal de la válvula l/h (default: 12.57)
  SIM_ET0_MM_DAY         ET0 base mm/día (default: 5.0)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("sensor-simulator")

# ── Config ────────────────────────────────────────────────────────────────────

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mosquitto")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USER   = os.getenv("MQTT_USER")
MQTT_PASS   = os.getenv("MQTT_PASS")

# ── Protocolo wiclouds ODINS ──────────────────────────────────────────────────
WICLOUDS_APIKEY = os.getenv("WICLOUDS_APIKEY", "odins")
WICLOUDS_SERIAL = os.getenv("WICLOUDS_SERIAL", "IA0301G00000000004")
WICLOUDS_TOPIC  = f"/{WICLOUDS_APIKEY}/{WICLOUDS_SERIAL}/attrs"

# Nombres de atributos wiclouds (cnt* → flow, ev* → valve)
ATTR_FLOW  = os.getenv("SIM_ATTR_FLOW",  "cnt24")   # volumen acumulado
ATTR_VALVE = os.getenv("SIM_ATTR_VALVE", "ev40")     # estado válvula 0/1

INTERVAL_S     = int(os.getenv("SIM_INTERVAL_S", "900"))       # 15 min
ANOMALY_PROB   = float(os.getenv("SIM_ANOMALY_PROB", "0.02"))  # 2% por tick
INITIAL_MOIST  = float(os.getenv("SIM_INITIAL_MOISTURE", "25.0"))
Q_LH           = float(os.getenv("SIM_Q_LH", "12.57"))         # l/h
ET0_MM_DAY     = float(os.getenv("SIM_ET0_MM_DAY", "5.0"))

# ── Estado del simulador ─────────────────────────────────────────────────────

vol_accumulated = 0.0        # litros acumulados del caudalímetro
soil_moisture   = INITIAL_MOIST  # % volumétrico
valve_open      = False      # estado de la electroválvula
valve_open_until = 0.0       # timestamp hasta cuando estará abierta


def _hour_of_day() -> float:
    """Hora del día como float (0.0 - 24.0) en UTC."""
    now = datetime.now(timezone.utc)
    return now.hour + now.minute / 60.0


def _soil_temperature() -> float:
    """
    Temperatura del suelo con ciclo diario sinusoidal.
    Mínimo ~18°C a las 06:00, máximo ~30°C a las 15:00.
    """
    h = _hour_of_day()
    base = 24.0
    amplitude = 6.0
    # Pico a las 15:00 (h=15), mínimo a las 03:00 (h=3)
    temp = base + amplitude * math.sin(2 * math.pi * (h - 9) / 24.0)
    return round(temp + random.gauss(0, 0.5), 1)


def _et0_per_tick() -> float:
    """
    ET0 por tick (mm). Distribuida a lo largo del día con pico al mediodía.
    Suma diaria ≈ ET0_MM_DAY.
    """
    h = _hour_of_day()
    # Solo hay evapotranspiración durante el día (6:00 - 20:00)
    if h < 6 or h > 20:
        return 0.0
    # Distribución sinusoidal con pico a las 13:00
    factor = max(0, math.sin(math.pi * (h - 6) / 14.0))
    # Normalizar para que la suma de 96 ticks (15 min cada uno) ≈ ET0_MM_DAY
    # Integral de sin(π·x/14) de 0 a 14 = 28/π ≈ 8.91
    # Hay 56 ticks diurnos (14h × 4/h), factor medio ≈ 0.637
    # → et0_per_tick = ET0_MM_DAY / (56 × 0.637) ≈ ET0_MM_DAY / 35.7
    et0_tick = ET0_MM_DAY * factor / 35.7
    return et0_tick


def simulate_tick() -> dict:
    """
    Avanza un tick del simulador (15 min) y devuelve los payloads a publicar.
    Retorna dict con keys 'flow', 'soil', 'valve'.
    """
    global vol_accumulated, soil_moisture, valve_open, valve_open_until

    now = datetime.now(timezone.utc)
    ts = now.isoformat()

    # ── Válvula: se abre si viene una orden (simulamos riego matutino) ────
    # Simular riego entre 06:00-06:30 (abre ~120 min de media)
    h = _hour_of_day()
    if not valve_open and 6.0 <= h <= 6.5 and random.random() < 0.3:
        # Abrir válvula por 60-180 minutos
        duration_min = random.uniform(60, 180)
        valve_open = True
        valve_open_until = time.time() + duration_min * 60
        log.info("Válvula ABIERTA por %.0f min", duration_min)

    # Cerrar si ha pasado el tiempo
    if valve_open and time.time() > valve_open_until:
        valve_open = False
        log.info("Válvula CERRADA")

    # ── Caudal ────────────────────────────────────────────────────────────
    if valve_open:
        # Litros en este tick (15 min = 0.25h)
        liters = Q_LH * (INTERVAL_S / 3600.0)
        liters += random.gauss(0, 0.1)  # ruido del sensor
    else:
        liters = random.gauss(0.0, 0.02)  # goteo residual / ruido
        liters = max(0, liters)

    # Anomalía: fuga aleatoria (incremento brusco)
    is_anomaly = False
    if random.random() < ANOMALY_PROB:
        leak = random.uniform(30, 80)  # fuga de 30-80 litros
        liters += leak
        is_anomaly = True
        log.warning("ANOMALÍA simulada: fuga de %.1f L", leak)

    vol_accumulated += liters

    # ── Humedad del suelo ─────────────────────────────────────────────────
    # El riego sube la humedad (simplificación: 1L ≈ 0.01% en parcela tipo)
    if valve_open:
        irrigation_effect = liters * 0.01
    else:
        irrigation_effect = 0.0

    # La ET0 baja la humedad
    et0_effect = _et0_per_tick() * 0.8  # factor de conversión mm → %

    # Lluvia aleatoria ocasional (baja probabilidad)
    rain_effect = 0.0
    if random.random() < 0.005:  # ~0.5% por tick → ~7 veces/día
        rain_mm = random.uniform(0.5, 3.0)
        rain_effect = rain_mm * 0.5  # mm → % simplificado
        log.info("Lluvia: %.1f mm", rain_mm)

    soil_moisture += irrigation_effect - et0_effect + rain_effect
    # Añadir ruido del sensor
    soil_moisture += random.gauss(0, 0.1)
    # Clamp
    soil_moisture = max(5.0, min(95.0, soil_moisture))

    temp = _soil_temperature()

    # ── Payloads ──────────────────────────────────────────────────────────
    flow_payload = {
        "vol": round(vol_accumulated, 2),
        "TimeInstant": ts,
    }
    soil_payload = {
        "soilMoisture": round(soil_moisture, 1),
        "temperature": temp,
        "TimeInstant": ts,
    }
    valve_payload = {
        "status": 1 if valve_open else 0,
        "TimeInstant": ts,
    }

    return {
        "flow": flow_payload,
        "soil": soil_payload,
        "valve": valve_payload,
        "is_anomaly": is_anomaly,
    }


# ── MQTT ──────────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("Conectado a MQTT %s:%s", BROKER_HOST, BROKER_PORT)
    else:
        log.error("Conexión MQTT fallida (rc=%s)", rc)


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=f"sensor-simulator-{os.getpid()}",
)
if MQTT_USER:
    client.username_pw_set(MQTT_USER, MQTT_PASS)
client.on_connect = on_connect


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log.info("Sensor simulator starting")
    log.info("  Broker:   %s:%s", BROKER_HOST, BROKER_PORT)
    log.info("  Interval: %ds (%d min)", INTERVAL_S, INTERVAL_S // 60)
    log.info("  Topic:    %s", WICLOUDS_TOPIC)
    log.info("  Attrs:    %s (flow), %s (valve)", ATTR_FLOW, ATTR_VALVE)
    log.info("  Anomaly prob: %.1f%% per tick", ANOMALY_PROB * 100)

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    # Esperar conexión
    time.sleep(2)

    tick = 0
    try:
        while True:
            tick += 1
            data = simulate_tick()

            # Publicar en formato wiclouds: un único mensaje con todos los
            # atributos al topic /{APIKEY}/{SERIAL}/attrs
            now_ts = int(time.time())
            wiclouds_payload = {
                ATTR_FLOW:  round(data["flow"]["vol"], 2),
                ATTR_VALVE: data["valve"]["status"],
                "hist": [
                    {"n": ATTR_FLOW,  "t": now_ts, "v": round(data["flow"]["vol"], 2)},
                    {"n": ATTR_VALVE, "t": now_ts, "v": data["valve"]["status"]},
                ],
            }
            client.publish(WICLOUDS_TOPIC, json.dumps(wiclouds_payload), qos=1)

            status = "ANOMALY" if data["is_anomaly"] else "ok"
            log.info(
                "Tick %d | %s=%.1f L | θ=%.1f%% | T=%.1f°C | %s=%s | %s",
                tick,
                ATTR_FLOW, data["flow"]["vol"],
                data["soil"]["soilMoisture"],
                data["soil"]["temperature"],
                ATTR_VALVE, "OPEN" if data["valve"]["status"] else "CLOSED",
                status,
            )

            time.sleep(INTERVAL_S)

    except KeyboardInterrupt:
        log.info("Simulator stopped")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
