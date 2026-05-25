"""
services/ingest_svc/app.py — Servicio de ingesta del edge.

Funciones:
  • Suscribe al broker MQTT (en un hilo daemon) y persiste cada mensaje en DuckDB.
  • Mantiene un buffer rotatorio: solo se conservan los últimos N días en DuckDB
    (el histórico completo vive en el servidor central).
  • Expone una API REST mínima para que las otras APIs / cron jobs interactúen
    con la BD sin necesidad de tocar el fichero (resuelve el lock de DuckDB).
  • Ejecuta la agregación diaria a features_daily.

Endpoints:
  GET  /health
  GET  /window?topic=valve&n=30        — últimas N filas de la tabla cruda
  POST /alert                          — inserta en anomaly_alerts
  POST /aggregate?date=YYYY-MM-DD      — agrega features del día indicado
                                         (por defecto, ayer)

Variables de entorno:
  MQTT_BROKER_HOST       host del broker (central, autenticado)
  MQTT_BROKER_PORT       puerto (default 1883)
  MQTT_USER, MQTT_PASS   credenciales (opcionales)
  WICLOUDS_APIKEY        API key ODINS (default: odins)
  WICLOUDS_TOPIC         topic wildcard (default: /{APIKEY}/+/attrs)
  DATA_DIR               directorio de la BD (default /app/data)
  BUFFER_DAYS_RAW        días a retener en raw_*  (default 30)
  BUFFER_DAYS_DAILY      días a retener en features_daily (default 90)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import duckdb
import paho.mqtt.client as mqtt
import pandas as pd
import requests as _requests
from flask import Flask, jsonify, request

# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("ingest-svc")

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USER   = os.getenv("MQTT_USER")
MQTT_PASS   = os.getenv("MQTT_PASS")

# ── Protocolo MQTT (formato wiclouds ODINS) ──────────────────────────────────
# Topic: /{APIKEY}/{SERIAL}/attrs
# Entrada: {"cnt24": 1234.56, "ev40": 1, ...}  (cnt* → flow, ev* → valve)
# Salida:  {"anomaly_alert": 1, "tau_minutes": 120, "hist": [...]}
WICLOUDS_APIKEY = os.getenv("WICLOUDS_APIKEY", "odins")
WICLOUDS_TOPIC  = os.getenv("WICLOUDS_TOPIC", f"/{WICLOUDS_APIKEY}/+/attrs")

# Mapeo de atributos por regex (configurable por dispositivo)
# cnt* → flow (volumen acumulado), ev* → valve (0/1)
import re
RE_CNT = re.compile(r"^cnt", re.IGNORECASE)
RE_EV  = re.compile(r"^ev",  re.IGNORECASE)

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
DB_PATH  = DATA_DIR / "irridea.duckdb"

BUFFER_DAYS_RAW   = int(os.getenv("BUFFER_DAYS_RAW",   "30"))
BUFFER_DAYS_DAILY = int(os.getenv("BUFFER_DAYS_DAILY", "90"))

# Tabla → columna timestamp (para /window y para la rotación)
RAW_TABLES = {
    "flow":   ("raw_flow",  "ts"),
    "valve":  ("raw_valve", "ts"),
    "soil":   ("raw_soil",  "ts"),
    "alerts": ("anomaly_alerts", "ts"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Capa de BD (un único conexión protegida por lock — DuckDB no es thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

class Database:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con  = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS raw_flow (
                    ts TIMESTAMPTZ NOT NULL, vol_l DOUBLE, vol_diff DOUBLE
                )""")
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS raw_valve (
                    ts TIMESTAMPTZ NOT NULL, status INTEGER
                )""")
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS raw_soil (
                    ts TIMESTAMPTZ NOT NULL, soil_moisture DOUBLE, temperature DOUBLE
                )""")
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS features_daily (
                    date DATE NOT NULL,
                    theta DOUBLE, water_used_l DOUBLE,
                    precip_mm DOUBLE, et0_mm DOUBLE,
                    PRIMARY KEY (date)
                )""")
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_alerts (
                    ts TIMESTAMPTZ, error DOUBLE, threshold DOUBLE,
                    predicted_l DOUBLE, actual_l DOUBLE
                )""")
            # ── Weather forecast (AEMET) ───────────────────────────────────
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS weather_forecast (
                    date       DATE NOT NULL,
                    et0_mm     DOUBLE,
                    precip_mm  DOUBLE,
                    fetched_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (date)
                )""")
            # ── RL ──────────────────────────────────────────────────────────
            # Transiciones crudas tal cual las loggea rl-api en cada /act:
            # observación + acción + timestamp. La recompensa y next_obs se
            # calculan a posteriori en build_rl_transitions DAG (reward
            # depende de θ_{t+1}, conocido solo al día siguiente).
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS rl_transitions_raw (
                    ts        TIMESTAMP    NOT NULL,
                    obs       DOUBLE[],
                    action    DOUBLE,
                    device_id VARCHAR DEFAULT 'edge-default'
                )""")
            # Transiciones completas, listas para alimentar el replay buffer.
            self._con.execute("""
                CREATE TABLE IF NOT EXISTS rl_transitions (
                    ts        TIMESTAMP    NOT NULL,
                    obs       DOUBLE[],
                    action    DOUBLE,
                    reward    DOUBLE,
                    next_obs  DOUBLE[],
                    done      BOOLEAN,
                    device_id VARCHAR DEFAULT 'edge-default',
                    PRIMARY KEY (ts, device_id)
                )""")

    def execute(self, sql: str, params: list | None = None):
        with self._lock:
            return self._con.execute(sql, params or [])

    def close(self) -> None:
        with self._lock:
            self._con.close()


db = Database(DB_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Buffer rotatorio
# ──────────────────────────────────────────────────────────────────────────────

def rotate_buffer() -> None:
    """Borra filas más antiguas que la ventana configurada."""
    cutoff_raw   = f"now() - INTERVAL '{BUFFER_DAYS_RAW} days'"
    cutoff_daily = f"current_date - {BUFFER_DAYS_DAILY}"
    db.execute(f"DELETE FROM raw_flow       WHERE ts < {cutoff_raw}")
    db.execute(f"DELETE FROM raw_valve      WHERE ts < {cutoff_raw}")
    db.execute(f"DELETE FROM raw_soil       WHERE ts < {cutoff_raw}")
    db.execute(f"DELETE FROM anomaly_alerts WHERE ts < {cutoff_raw}")
    db.execute(f"DELETE FROM features_daily WHERE date < {cutoff_daily}")


# Hilo dedicado a rotar (cada hora) — no se hace inline en cada inserción para
# no ralentizar la ingesta.
def _rotation_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(timeout=3600):
        try:
            rotate_buffer()
            log.info("Buffer rotado")
        except Exception:
            log.exception("Fallo rotando buffer")


# ──────────────────────────────────────────────────────────────────────────────
# Handlers MQTT
# ──────────────────────────────────────────────────────────────────────────────

def _ts(payload: dict) -> pd.Timestamp:
    return pd.to_datetime(payload.get("TimeInstant", dt.datetime.now(dt.timezone.utc)))


def handle_flow(payload: dict) -> None:
    db.execute("INSERT INTO raw_flow VALUES (?, ?, NULL)",
               [_ts(payload), float(payload.get("vol", 0))])


def handle_valve(payload: dict) -> None:
    db.execute("INSERT INTO raw_valve VALUES (?, ?)",
               [_ts(payload), int(payload.get("status", 0))])


def handle_soil(payload: dict) -> None:
    db.execute("INSERT INTO raw_soil VALUES (?, ?, ?)",
               [_ts(payload), float(payload.get("soilMoisture", 0)),
                float(payload.get("temperature", 0))])


# Referencia al cliente MQTT (se asigna en start_mqtt)
_mqtt_client: mqtt.Client | None = None


def _serial_from_topic(topic: str) -> str | None:
    """Extrae el SERIAL de un topic /{APIKEY}/{SERIAL}/attrs."""
    parts = topic.strip("/").split("/")
    return parts[1] if len(parts) >= 3 else None


def _wiclouds_topic(serial: str) -> str:
    """Construye el topic de publicación para un dispositivo."""
    return f"/{WICLOUDS_APIKEY}/{serial}/attrs"


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("Conectado a MQTT %s:%s", BROKER_HOST, BROKER_PORT)
        client.subscribe(WICLOUDS_TOPIC, qos=1)
        log.info("Suscrito a %s", WICLOUDS_TOPIC)
    else:
        log.error("Conexión MQTT fallida (rc=%s)", rc)


def on_message(client, userdata, msg):
    """
    Procesa mensajes wiclouds: /{APIKEY}/{SERIAL}/attrs
    Payload: {"cnt24": 1234.56, "ev40": 1, ...}
    Ignora atributos publicados por nosotros (_source=irridea).
    """
    try:
        payload = json.loads(msg.payload.decode("utf-8"))

        # Ignorar mensajes que nosotros mismos publicamos
        if payload.get("_source") == "irridea":
            return

        serial = _serial_from_topic(msg.topic)
        if not serial:
            return

        now = dt.datetime.now(dt.timezone.utc)

        for key, value in payload.items():
            if key in ("hist", "_source"):
                continue

            if RE_CNT.match(key):
                vol = float(value)
                handle_flow({"vol": vol, "TimeInstant": now.isoformat()})
                log.info("[%s] %s=%.2f → raw_flow", serial, key, vol)

            elif RE_EV.match(key):
                status = int(value)
                handle_valve({"status": status, "TimeInstant": now.isoformat()})
                log.info("[%s] %s=%d → raw_valve", serial, key, status)

    except Exception:
        log.exception("Error procesando mensaje [%s]", msg.topic)


def start_mqtt() -> mqtt.Client:
    global _mqtt_client
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"ingest-svc-{os.getpid()}",
        clean_session=False,  # mantiene cola al reconectar
    )
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()  # hilo daemon
    _mqtt_client = client
    return client


# ──────────────────────────────────────────────────────────────────────────────
# Agregación diaria
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_daily(target_date: Optional[str] = None) -> dict:
    if target_date is None:
        target_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    # Calcular vol_diff de las filas que falten. DuckDB no permite window
    # functions en UPDATE, así que se hace en dos pasos: CTE con LAG → JOIN.
    db.execute("""
        WITH diffs AS (
            SELECT ts, vol_l - LAG(vol_l) OVER (ORDER BY ts) AS d
            FROM raw_flow
        )
        UPDATE raw_flow AS r
        SET vol_diff = diffs.d
        FROM diffs
        WHERE diffs.ts = r.ts AND r.vol_diff IS NULL
    """)

    theta = db.execute(
        "SELECT AVG(soil_moisture) FROM raw_soil WHERE CAST(ts AS DATE) = ?",
        [target_date],
    ).fetchone()[0]

    water = db.execute(
        "SELECT COALESCE(SUM(vol_diff), 0) FROM raw_flow "
        "WHERE CAST(ts AS DATE) = ? AND vol_diff > 0",
        [target_date],
    ).fetchone()[0] or 0.0

    db.execute(
        "INSERT OR REPLACE INTO features_daily (date, theta, water_used_l) "
        "VALUES (?, ?, ?)",
        [target_date, theta, water],
    )

    log.info("Agregación %s: θ=%s, agua=%.0fL", target_date, theta, water)
    return {"date": target_date, "theta": theta, "water_used_l": water}


# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", db=str(DB_PATH))


@app.get("/window")
def window():
    topic = request.args.get("topic", "valve")
    n     = int(request.args.get("n", "30"))

    if topic not in RAW_TABLES:
        return jsonify(error=f"topic must be one of {list(RAW_TABLES)}"), 400

    table, ts_col = RAW_TABLES[topic]
    rows = db.execute(
        f"SELECT * FROM {table} ORDER BY {ts_col} DESC LIMIT ?", [n]
    ).fetchall()
    cols = [d[0] for d in db.execute(f"DESCRIBE {table}").fetchall()]
    rows = [dict(zip(cols, r)) for r in reversed(rows)]
    # Serializar timestamps
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, (dt.datetime, dt.date)):
                r[k] = v.isoformat()
    return jsonify(rows=rows, count=len(rows))


@app.post("/alert")
def alert():
    body = request.get_json(force=True)
    db.execute(
        "INSERT INTO anomaly_alerts VALUES (?, ?, ?, ?, ?)",
        [
            body["ts"], body["error"], body["threshold"],
            body["predicted_l"], body["actual_l"],
        ],
    )
    return jsonify(status="inserted"), 201


@app.post("/rl/transition")
def rl_transition():
    """
    Persiste una transición cruda emitida por rl-api en cada /act.

    Body:
      { "ts": "ISO-8601",
        "obs": [θ, w_prev, E_t, ..., E_{t+N-1}, P_t, ..., P_{t+N-1}],
        "action": <τ_minutes>,
        "device_id": "edge-XXX" (opcional) }

    Solo guardamos lo observable en el momento de la decisión; el reward y
    next_obs se completan a posteriori en el DAG build_rl_transitions.
    """
    body = request.get_json(force=True)
    try:
        ts        = body["ts"]
        obs       = list(map(float, body["obs"]))
        action    = float(body["action"])
        device_id = body.get("device_id", "edge-default")
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify(error=f"bad input: {exc}"), 400

    db.execute(
        "INSERT INTO rl_transitions_raw VALUES (?, ?, ?, ?)",
        [ts, obs, action, device_id],
    )
    return jsonify(status="inserted"), 201


def build_rl_transitions(target_date: Optional[str] = None) -> dict:
    """
    Empareja raw t con raw t+1 (mismo device_id) y calcula reward.

    Política de pareo:
      - Para cada transición raw del día `target_date`, busca la primera
        transición raw del MISMO device_id con ts > la suya. Esa será
        s_{t+1}; θ_{t+1} es obs_next[0].
      - reward = compute_reward(next_theta=θ_{t+1}, tau_minutes=action).
      - done = False (el problema de riego es continuo, no hay terminal real).
      - Si no hay transición posterior, no se construye fila (queda
        pendiente para el día siguiente).
      - PRIMARY KEY (ts, device_id) impide duplicar al re-ejecutar.
    """
    # Importa la función pura embebida en el repo (volumen compartido por compose).
    # En el container ingest-svc esto NO está disponible; el endpoint solo se
    # llama desde Airflow (que sí monta /opt/airflow/project/src) o por test.
    import sys
    repo_root = os.getenv("REPO_ROOT", "/app/src_repo")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from rl.reward import compute_reward  # type: ignore

    if target_date is None:
        target_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    rows = db.execute("""
        WITH src AS (
            SELECT ts, obs, action, device_id
            FROM rl_transitions_raw
            WHERE CAST(ts AS DATE) = ?
        ),
        nxt AS (
            SELECT
                s.ts        AS ts,
                s.obs       AS obs,
                s.action    AS action,
                s.device_id AS device_id,
                (SELECT r2.obs FROM rl_transitions_raw r2
                  WHERE r2.device_id = s.device_id
                    AND r2.ts > s.ts
                  ORDER BY r2.ts ASC LIMIT 1) AS next_obs
            FROM src s
        )
        SELECT ts, obs, action, device_id, next_obs FROM nxt
        WHERE next_obs IS NOT NULL
    """, [target_date]).fetchall()

    inserted = 0
    skipped  = 0
    for ts, obs, action, device_id, next_obs in rows:
        next_theta = float(next_obs[0])
        reward     = compute_reward(next_theta=next_theta, tau_minutes=float(action))
        try:
            db.execute(
                "INSERT INTO rl_transitions VALUES (?, ?, ?, ?, ?, ?, ?)",
                [ts, list(obs), float(action), reward, list(next_obs), False, device_id],
            )
            inserted += 1
        except duckdb.ConstraintException:
            skipped += 1

    log.info("build_rl_transitions(%s): %d insertadas, %d ya existían",
             target_date, inserted, skipped)
    return {"date": target_date, "inserted": inserted, "skipped": skipped}


@app.post("/rl/build")
def rl_build_endpoint():
    target_date = request.args.get("date")
    try:
        result = build_rl_transitions(target_date)
        return jsonify(status="ok", **result)
    except Exception as exc:
        log.exception("Fallo en /rl/build")
        return jsonify(status="error", error=str(exc)), 500


@app.get("/rl/export")
def rl_export_endpoint():
    """
    Devuelve todas las transiciones completas en JSON para que el DAG de
    fine-tune las vuelque a parquet en su entorno.
    """
    rows = db.execute(
        "SELECT ts, obs, action, reward, next_obs, done, device_id "
        "FROM rl_transitions ORDER BY ts ASC"
    ).fetchall()
    out = [
        {
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "obs": list(obs),
            "action": float(action),
            "reward": float(reward),
            "next_obs": list(next_obs),
            "done": bool(done),
            "device_id": device_id,
        }
        for ts, obs, action, reward, next_obs, done, device_id in rows
    ]
    return jsonify(rows=out, count=len(out))


@app.post("/aggregate")
def aggregate_endpoint():
    target_date = request.args.get("date")
    try:
        result = aggregate_daily(target_date)
        return jsonify(status="ok", **result)
    except Exception as exc:
        log.exception("Fallo en /aggregate")
        return jsonify(status="error", error=str(exc)), 500


# ──────────────────────────────────────────────────────────────────────────────
# Observación RL (para el ciclo diario de riego)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/rl/obs")
def rl_obs():
    """
    Construye el vector de observación para el agente RL.

    Devuelve: {obs: [theta, w_prev, E_t, E_t+1, P_t, P_t+1]}

    - theta: humedad media del suelo del día más reciente en features_daily
    - w_prev: agua usada (litros) del día más reciente → convertida a minutos
    - E_t, E_t+1: ET0 de hoy y mañana (de weather_forecast)
    - P_t, P_t+1: precipitación de hoy y mañana (de weather_forecast)
    """
    today = dt.date.today().isoformat()
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()

    row = db.execute(
        "SELECT theta, water_used_l FROM features_daily "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()

    if row is None:
        theta = 0.0
        w_prev = 0.0
    else:
        theta = float(row[0]) if row[0] is not None else 0.0
        q_lh = float(os.getenv("Q_LH", "12.57"))
        w_prev = float(row[1]) / q_lh if row[1] and q_lh > 0 else 0.0

    forecast_today = db.execute(
        "SELECT et0_mm, precip_mm FROM weather_forecast WHERE date = ?",
        [today],
    ).fetchone()

    forecast_tomorrow = db.execute(
        "SELECT et0_mm, precip_mm FROM weather_forecast WHERE date = ?",
        [tomorrow],
    ).fetchone()

    et_t  = float(forecast_today[0])    if forecast_today    and forecast_today[0]    else 0.0
    pt_t  = float(forecast_today[1])    if forecast_today    and forecast_today[1]    else 0.0
    et_t1 = float(forecast_tomorrow[0]) if forecast_tomorrow and forecast_tomorrow[0] else 0.0
    pt_t1 = float(forecast_tomorrow[1]) if forecast_tomorrow and forecast_tomorrow[1] else 0.0

    obs = [theta, w_prev, et_t, et_t1, pt_t, pt_t1]

    log.info("/rl/obs: theta=%.2f w_prev=%.2f E=[%.1f,%.1f] P=[%.1f,%.1f]",
             theta, w_prev, et_t, et_t1, pt_t, pt_t1)
    return jsonify(obs=obs, date=today)


# ──────────────────────────────────────────────────────────────────────────────
# Weather forecast (AEMET)
# ──────────────────────────────────────────────────────────────────────────────

AEMET_API_KEY = os.getenv("AEMET_API_KEY", "")
AEMET_MUNICIPIO = os.getenv("AEMET_MUNICIPIO", "30030")  # Murcia por defecto
LATITUDE_DEG = float(os.getenv("WEATHER_LAT", "38.028"))  # Murcia


def _hargreaves_et0(tmax: float, tmin: float, date: dt.date) -> float:
    """
    Calcula ET₀ diaria (mm/día) con la ecuación de Hargreaves-Samani (1985).

    ET₀ = 0.0023 · (T_mean + 17.8) · (T_max − T_min)^0.5 · Ra
    donde Ra = radiación extraterrestre (mm/día equivalentes).

    Ref: Allen et al. (1998), FAO Irrigation & Drainage Paper 56, ec. 52.
    """
    import math

    lat_rad = math.radians(LATITUDE_DEG)
    doy = date.timetuple().tm_yday

    # Distancia inversa relativa Tierra-Sol (FAO-56, ec. 23)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    # Declinación solar (FAO-56, ec. 24)
    delta = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    # Ángulo horario de puesta de sol (FAO-56, ec. 25)
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta))

    # Radiación extraterrestre Ra en MJ/m²/día (FAO-56, ec. 21)
    Gsc = 0.0820  # constante solar, MJ/m²/min
    ra_mj = (24.0 * 60.0 / math.pi) * Gsc * dr * (
        ws * math.sin(lat_rad) * math.sin(delta)
        + math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )

    # Convertir Ra de MJ/m²/día a mm/día equivalentes (÷ 2.45 MJ/kg)
    ra_mm = ra_mj / 2.45

    tmean = (tmax + tmin) / 2.0
    tdiff = max(tmax - tmin, 0.0)

    et0 = 0.0023 * (tmean + 17.8) * math.sqrt(tdiff) * ra_mm
    return round(max(et0, 0.0), 2)


@app.post("/weather/fetch")
def weather_fetch():
    """
    Obtiene el pronóstico de AEMET para los próximos 2 días (hoy y mañana)
    y lo guarda en la tabla weather_forecast.

    Si AEMET_API_KEY no está configurada, usa valores por defecto
    (útil para desarrollo sin acceso a la API real).
    """
    if not AEMET_API_KEY:
        log.warning("/weather/fetch: AEMET_API_KEY not set, using defaults")
        today = dt.date.today()
        defaults = []
        for i in range(2):
            d = today + dt.timedelta(days=i)
            db.execute(
                "INSERT INTO weather_forecast (date, et0_mm, precip_mm) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT (date) DO UPDATE SET et0_mm = EXCLUDED.et0_mm, "
                "precip_mm = EXCLUDED.precip_mm, fetched_at = now()",
                [d.isoformat(), 6.0, 0.0],
            )
            defaults.append({"date": d.isoformat(), "et0_mm": 6.0, "precip_mm": 0.0})
        return jsonify(status="defaults", days=defaults)

    try:
        url = (
            f"https://opendata.aemet.es/opendata/api/prediccion/"
            f"especifica/municipio/diaria/{AEMET_MUNICIPIO}"
        )
        r = _requests.get(
            url,
            headers={"api_key": AEMET_API_KEY},
            timeout=15,
        )
        r.raise_for_status()
        data_url = r.json().get("datos")
        if not data_url:
            return jsonify(status="error", error="no datos URL in AEMET response"), 502

        r2 = _requests.get(data_url, timeout=15)
        r2.raise_for_status()
        forecast_data = r2.json()

        saved = []
        for day_data in forecast_data[0].get("prediccion", {}).get("dia", [])[:2]:
            fecha = day_data["fecha"][:10]

            # ── Precipitación (mm) ──
            precip = 0.0
            if "probPrecipitacion" in day_data:
                for p in day_data["probPrecipitacion"]:
                    if p.get("valor"):
                        precip = max(precip, float(p["valor"]))

            # ── ET₀ vía Hargreaves con Tmax/Tmin de AEMET ──
            temp = day_data.get("temperatura", {})
            tmax = float(temp.get("maxima", 30))
            tmin = float(temp.get("minima", 15))
            forecast_date = dt.date.fromisoformat(fecha)
            et0 = _hargreaves_et0(tmax, tmin, forecast_date)

            db.execute(
                "INSERT INTO weather_forecast (date, et0_mm, precip_mm) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT (date) DO UPDATE SET et0_mm = EXCLUDED.et0_mm, "
                "precip_mm = EXCLUDED.precip_mm, fetched_at = now()",
                [fecha, et0, precip],
            )
            saved.append({"date": fecha, "et0_mm": et0, "precip_mm": precip})

        log.info("/weather/fetch: saved %d days from AEMET", len(saved))
        return jsonify(status="ok", days=saved)

    except Exception as exc:
        log.exception("AEMET fetch failed")
        return jsonify(status="error", error=str(exc)), 502


# ──────────────────────────────────────────────────────────────────────────────
# Grafana time-series endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/grafana/flow")
def grafana_flow():
    """Últimas N lecturas de caudal con vol_diff calculado al vuelo."""
    n = int(request.args.get("n", 100))
    rows = db.execute(
        "SELECT ts, vol_l, "
        "  vol_l - LAG(vol_l) OVER (ORDER BY ts) AS vol_diff "
        "FROM (SELECT ts, vol_l FROM raw_flow ORDER BY ts DESC LIMIT ?) "
        "ORDER BY ts",
        [n + 1],
    ).fetchall()
    # Skip first row (vol_diff is NULL for it)
    return jsonify([
        {"ts": r[0].isoformat() if r[0] else None, "vol_l": r[1], "vol_diff": r[2]}
        for r in rows if r[2] is not None
    ])


@app.get("/grafana/soil")
def grafana_soil():
    """Últimas N lecturas de humedad del suelo."""
    n = int(request.args.get("n", 100))
    rows = db.execute(
        "SELECT ts, soil_moisture, temperature FROM raw_soil ORDER BY ts DESC LIMIT ?", [n]
    ).fetchall()
    return jsonify([
        {"ts": r[0].isoformat() if r[0] else None, "soil_moisture": r[1], "temperature": r[2]}
        for r in reversed(rows)
    ])


@app.get("/grafana/alerts")
def grafana_alerts():
    """Últimas N alertas de anomalía."""
    n = int(request.args.get("n", 50))
    rows = db.execute(
        "SELECT ts, error, threshold, actual_l, predicted_l "
        "FROM anomaly_alerts ORDER BY ts DESC LIMIT ?", [n]
    ).fetchall()
    return jsonify([
        {"ts": r[0].isoformat() if r[0] else None, "error_score": r[1],
         "threshold": r[2], "actual_l": r[3], "predicted_l": r[4]}
        for r in reversed(rows)
    ])


@app.get("/grafana/daily")
def grafana_daily():
    """Histórico de features_daily (θ, agua, riego)."""
    n = int(request.args.get("n", 90))
    rows = db.execute(
        "SELECT date, theta, water_used_l FROM features_daily "
        "ORDER BY date DESC LIMIT ?", [n]
    ).fetchall()
    return jsonify([
        {"date": r[0].isoformat() if r[0] else None, "theta": r[1], "water_used_l": r[2]}
        for r in reversed(rows)
    ])


@app.get("/grafana/weather")
def grafana_weather():
    """Forecast meteorológico almacenado."""
    rows = db.execute(
        "SELECT date, et0_mm, precip_mm FROM weather_forecast ORDER BY date"
    ).fetchall()
    return jsonify([
        {"date": r[0].isoformat() if r[0] else None, "et0_mm": r[1], "precip_mm": r[2]}
        for r in rows
    ])


@app.get("/grafana/rl")
def grafana_rl():
    """Últimas acciones del agente RL."""
    n = int(request.args.get("n", 90))
    rows = db.execute(
        "SELECT ts, action, device_id FROM rl_transitions_raw "
        "ORDER BY ts DESC LIMIT ?", [n]
    ).fetchall()
    return jsonify([
        {"ts": r[0].isoformat() if r[0] else None,
         "date": str(r[0])[:10] if r[0] else None,
         "tau_minutes": r[1], "device_id": r[2]}
        for r in reversed(rows)
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap (al importar bajo gunicorn)
# ──────────────────────────────────────────────────────────────────────────────

_stop_event = threading.Event()
_mqtt_client = start_mqtt()
_rotation_thread = threading.Thread(
    target=_rotation_loop, args=(_stop_event,), daemon=True, name="buffer-rotator"
)
_rotation_thread.start()


# Para ejecución en local sin gunicorn (`python app.py`)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
