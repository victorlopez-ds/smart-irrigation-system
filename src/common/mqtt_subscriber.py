"""
src/common/mqtt_subscriber.py — Worker MQTT que consume Mosquitto y persiste en Parquet.

Suscribe a los tópicos de caudal, humedad y electroválvula publicados cada
15 minutos por el sistema de sensores. Cada mensaje se parsea y se añade a
la tabla Parquet correspondiente, particionada por fecha.

Uso (contenedor ingest.Dockerfile):
    python -m src.common.mqtt_subscriber

Variables de entorno requeridas:
    MQTT_BROKER_HOST   (por defecto: localhost)
    MQTT_BROKER_PORT   (por defecto: 1883)
    MQTT_TOPIC_FLOW    tópico del contador de agua
    MQTT_TOPIC_VALVE   tópico de la electroválvula
    MQTT_TOPIC_SOIL    tópico del sensor de humedad
    DATA_DIR           directorio raíz donde se escriben los Parquet (por defecto: data/raw_stream)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuración desde entorno
# ──────────────────────────────────────────────────────────────────────────────

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
TOPIC_FLOW = os.getenv("MQTT_TOPIC_FLOW", "irridea/flow")
TOPIC_VALVE = os.getenv("MQTT_TOPIC_VALVE", "irridea/valve")
TOPIC_SOIL = os.getenv("MQTT_TOPIC_SOIL", "irridea/soil")
DATA_DIR = Path(os.getenv("DATA_DIR", "data/raw_stream"))
DB_PATH = DATA_DIR / "irridea.duckdb"


# ──────────────────────────────────────────────────────────────────────────────
# Inicialización de la base de datos DuckDB
# ──────────────────────────────────────────────────────────────────────────────

def init_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_flow (
            ts        TIMESTAMPTZ NOT NULL,
            vol_l     DOUBLE,
            vol_diff  DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_valve (
            ts     TIMESTAMPTZ NOT NULL,
            status INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_soil (
            ts            TIMESTAMPTZ NOT NULL,
            soil_moisture DOUBLE,
            temperature   DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS features_daily (
            date         DATE        NOT NULL,
            theta        DOUBLE,
            water_used_l DOUBLE,
            precip_mm    DOUBLE,
            et0_mm       DOUBLE,
            PRIMARY KEY (date)
        )
    """)
    return con


# ──────────────────────────────────────────────────────────────────────────────
# Parsers por tópico
# ──────────────────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def handle_flow(payload: dict, con: duckdb.DuckDBPyConnection) -> None:
    ts = pd.to_datetime(payload.get("TimeInstant", _now_utc()))
    vol = float(payload.get("vol", 0))
    # La diferencia se calculará en la agregación diaria
    con.execute(
        "INSERT INTO raw_flow VALUES (?, ?, NULL)",
        [ts, vol],
    )


def handle_valve(payload: dict, con: duckdb.DuckDBPyConnection) -> None:
    ts = pd.to_datetime(payload.get("TimeInstant", _now_utc()))
    status = int(payload.get("status", 0))
    con.execute("INSERT INTO raw_valve VALUES (?, ?)", [ts, status])


def handle_soil(payload: dict, con: duckdb.DuckDBPyConnection) -> None:
    ts = pd.to_datetime(payload.get("TimeInstant", _now_utc()))
    moisture = float(payload.get("soilMoisture", 0))
    temp = float(payload.get("temperature", 0))
    con.execute("INSERT INTO raw_soil VALUES (?, ?, ?)", [ts, moisture, temp])


HANDLERS = {
    TOPIC_FLOW: handle_flow,
    TOPIC_VALVE: handle_valve,
    TOPIC_SOIL: handle_soil,
}


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks MQTT
# ──────────────────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("Conectado al broker MQTT %s:%s", BROKER_HOST, BROKER_PORT)
        for topic in HANDLERS:
            client.subscribe(topic)
            log.info("Suscrito a: %s", topic)
    else:
        log.error("Error de conexión MQTT. Código: %s", rc)


def on_message(client, userdata, msg):
    con: duckdb.DuckDBPyConnection = userdata["con"]
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        handler = HANDLERS.get(msg.topic)
        if handler:
            handler(payload, con)
            log.info("Mensaje %s guardado corectamente", msg.topic)
    except Exception as exc:
        log.warning("Error procesando mensaje [%s]: %s", msg.topic, exc)


# ──────────────────────────────────────────────────────────────────────────────
# Agregación diaria (llamada por el DAG de Airflow al final del día)
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_daily(target_date: str | None = None, db_path: Path = DB_PATH) -> None:
    """
    Agrega las lecturas 15-min del día `target_date` (YYYY-MM-DD) en
    features_daily. Si target_date es None, usa ayer.
    """
    import datetime as dt

    if target_date is None:
        target_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    con = duckdb.connect(str(db_path))

    # Humedad media del día (lectura de las 00:00)
    theta = con.execute(f"""
        SELECT AVG(soil_moisture)
        FROM raw_soil
        WHERE CAST(ts AS DATE) = '{target_date}'
    """).fetchone()[0]

    # Agua total de riego del día
    water_used = con.execute(f"""
        SELECT SUM(vol_diff)
        FROM raw_flow
        WHERE CAST(ts AS DATE) = '{target_date}' AND vol_diff > 0
    """).fetchone()[0] or 0.0

    log.info("Agregación diaria %s: θ=%.1f%%, agua=%.0f L", target_date, theta or 0, water_used)

    con.execute("""
        INSERT OR REPLACE INTO features_daily (date, theta, water_used_l)
        VALUES (?, ?, ?)
    """, [target_date, theta, water_used])
    con.close()


# ──────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    con = init_db(DB_PATH)
    log.info("Base de datos inicializada en %s", DB_PATH)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.user_data_set({"con": con})
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    def _shutdown(sig, frame):
        log.info("Señal %s recibida. Cerrando worker MQTT...", sig)
        client.disconnect()
        con.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("Worker MQTT en marcha. Esperando mensajes...")
    client.loop_forever()


if __name__ == "__main__":
    main()
