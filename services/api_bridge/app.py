"""
services/api_bridge/app.py — Bridge API REST wiclouds → MQTT.

Sustituye al sensor-simulator en producción. En vez de generar datos
sintéticos, consulta periódicamente la API REST de wiclouds
(riego.mimurcia.murcia.es) y publica los datos nuevos al broker MQTT
en formato ODINS.

Datos que consulta:
  1. Caudalímetro  → /controller/{ctrl}/device/{dev}/historical
     Respuesta: {"value": [{"d": "ISO", "v": float}]}
     → Publica como cnt24 (volumen acumulado)
  2. Humedad suelo → /controller/{ctrl}/device/{dev}/historical
     Respuesta: {"value": [{"d": "ISO", "v": float}]}
     → Publica como soil_moisture
  3. Riegos        → /controller/{ctrl}/irrideahistorical
     Respuesta: {"historicals": [{inicioProgramaRiego, finProgramaRiego, ...}]}
     → Deriva ev40 (0/1) según si hay riego activo

Protocolo de salida (wiclouds ODINS):
  Topic:   /{APIKEY}/{SERIAL}/attrs
  Payload: {"cnt24": v, "hist": [{"n":"cnt24","t":unix_ts,"v":v}]}

Variables de entorno:
  MQTT_BROKER_HOST        (default: mosquitto)
  MQTT_BROKER_PORT        (default: 1883)
  MQTT_USER / MQTT_PASS
  WICLOUDS_APIKEY         (default: odins)
  WICLOUDS_SERIAL         (default: IA0301G00000000004)
  WICLOUDS_API_BASE       (default: https://riego.mimurcia.murcia.es/backend)
  WICLOUDS_API_TOKEN      JWT para x-access-token
  WICLOUDS_CTRL_FLOW      Controller ID del caudalímetro
  WICLOUDS_DEV_FLOW       Device ID del caudalímetro
  WICLOUDS_CTRL_SOIL      Controller ID del sensor de humedad
  WICLOUDS_DEV_SOIL       Device ID del sensor de humedad
  POLL_INTERVAL_S         (default: 900 = 15 min)
  ATTR_FLOW               (default: cnt24)
  ATTR_VALVE              (default: ev40)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import paho.mqtt.client as mqtt
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("api-bridge")

# ── Config ────────────────────────────────────────────────────────────────────

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mosquitto")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USER   = os.getenv("MQTT_USER")
MQTT_PASS   = os.getenv("MQTT_PASS")

WICLOUDS_APIKEY = os.getenv("WICLOUDS_APIKEY", "odins")
WICLOUDS_SERIAL = os.getenv("WICLOUDS_SERIAL", "IA0301G00000000004")
WICLOUDS_TOPIC  = f"/{WICLOUDS_APIKEY}/{WICLOUDS_SERIAL}/attrs"

API_BASE  = os.getenv("WICLOUDS_API_BASE", "https://riego.mimurcia.murcia.es/backend")
API_TOKEN = os.getenv("WICLOUDS_API_TOKEN", "")

CTRL_FLOW = os.getenv("WICLOUDS_CTRL_FLOW", "62b5ae654b36644a08d464ac")
DEV_FLOW  = os.getenv("WICLOUDS_DEV_FLOW",  "62b5ae654b366462ced45f0c")
CTRL_SOIL = os.getenv("WICLOUDS_CTRL_SOIL", "64ccb0f4bb7b241ff2f8a41f")
DEV_SOIL  = os.getenv("WICLOUDS_DEV_SOIL",  "64ccb0f4bb7b241561f8a415")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_S", "900"))
ATTR_FLOW     = os.getenv("ATTR_FLOW",  "cnt24")
ATTR_VALVE    = os.getenv("ATTR_VALVE", "ev40")

# Headers comunes para la API wiclouds
API_HEADERS = {
    "Accept": "*/*",
    "Application": "WiCloudS",
    "x-access-token": API_TOKEN,
}

# ── Estado ────────────────────────────────────────────────────────────────────

# Timestamps ISO del último dato procesado por tipo.
# Al arrancar, se inicializan a now - POLL_INTERVAL para evitar publicar
# histórico antiguo.
_last_seen: dict[str, str] = {}


def _init_last_seen() -> None:
    """Inicializa los timestamps de seguimiento."""
    start = (datetime.now(timezone.utc) - timedelta(seconds=POLL_INTERVAL)).isoformat()
    _last_seen["flow"] = start
    _last_seen["soil"] = start
    _last_seen["irrigation"] = start


# ── Helpers de la API ─────────────────────────────────────────────────────────

def _iso_quote(ts: str) -> str:
    """Envuelve un timestamp ISO entre comillas dobles para el query param."""
    return f'"{ts}"'


def _fetch_historical(ctrl: str, dev: str, from_ts: str, to_ts: str) -> list[dict]:
    """
    GET /controller/{ctrl}/device/{dev}/historical?from=...&to=...
    Devuelve lista de {"d": ISO, "v": float}.
    """
    url = f"{API_BASE}/controller/{ctrl}/device/{dev}/historical"
    params = {
        "from": _iso_quote(from_ts),
        "to": _iso_quote(to_ts),
        "type": _iso_quote("undefined"),
    }
    try:
        r = requests.get(url, params=params, headers=API_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("value", [])
    except Exception:
        log.exception("Error consultando %s", url)
        return []


def _fetch_irrigation(ctrl: str, from_ts: str, to_ts: str) -> list[dict]:
    """
    GET /controller/{ctrl}/irrideahistorical?from=...&to=...
    Devuelve lista de programas de riego.
    """
    url = f"{API_BASE}/controller/{ctrl}/irrideahistorical"
    params = {
        "from": _iso_quote(from_ts),
        "to": _iso_quote(to_ts),
    }
    try:
        r = requests.get(url, params=params, headers=API_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("historicals", [])
    except Exception:
        log.exception("Error consultando %s", url)
        return []


def _iso_to_unix(iso: str) -> int:
    """Convierte ISO 8601 a timestamp Unix."""
    # Manejar formatos con y sin Z
    iso = iso.replace("Z", "+00:00")
    dt_obj = datetime.fromisoformat(iso)
    return int(dt_obj.timestamp())


# ── Lógica de polling ─────────────────────────────────────────────────────────

def _publish(client: mqtt.Client, payload: dict) -> None:
    """Publica un mensaje al topic wiclouds."""
    info = client.publish(WICLOUDS_TOPIC, json.dumps(payload), qos=1)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        log.warning("Fallo publicando MQTT (rc=%s)", info.rc)


def poll_flow(client: mqtt.Client, from_ts: str, to_ts: str) -> int:
    """Consulta datos de caudal y publica los nuevos. Devuelve nº publicados."""
    records = _fetch_historical(CTRL_FLOW, DEV_FLOW, from_ts, to_ts)
    last_seen = _last_seen["flow"]
    published = 0

    for rec in records:
        if rec["d"] <= last_seen:
            continue

        unix_ts = _iso_to_unix(rec["d"])
        payload = {
            ATTR_FLOW: rec["v"],
            "hist": [{"n": ATTR_FLOW, "t": unix_ts, "v": rec["v"]}],
        }
        _publish(client, payload)
        last_seen = rec["d"]
        published += 1

    _last_seen["flow"] = last_seen
    return published


def poll_soil(client: mqtt.Client, from_ts: str, to_ts: str) -> int:
    """Consulta datos de humedad y publica los nuevos. Devuelve nº publicados."""
    records = _fetch_historical(CTRL_SOIL, DEV_SOIL, from_ts, to_ts)
    last_seen = _last_seen["soil"]
    published = 0

    for rec in records:
        if rec["d"] <= last_seen:
            continue

        unix_ts = _iso_to_unix(rec["d"])
        payload = {
            "soil_moisture": rec["v"],
            "hist": [{"n": "soil_moisture", "t": unix_ts, "v": rec["v"]}],
        }
        _publish(client, payload)
        last_seen = rec["d"]
        published += 1

    _last_seen["soil"] = last_seen
    return published


def poll_irrigation(client: mqtt.Client, from_ts: str, to_ts: str) -> int:
    """
    Consulta programas de riego y deriva el estado de la válvula (ev40).
    Si hay algún programa cuyo rango [inicio, fin] incluye `now` → ev40=1.
    """
    programs = _fetch_irrigation(CTRL_FLOW, from_ts, to_ts)
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    valve_open = False
    for prog in programs:
        inicio = prog.get("inicioProgramaRiego", "")
        fin = prog.get("finProgramaRiego", "")
        if not inicio or not fin:
            continue

        inicio_dt = datetime.fromisoformat(inicio.replace("Z", "+00:00"))
        fin_dt = datetime.fromisoformat(fin.replace("Z", "+00:00"))

        if inicio_dt <= now <= fin_dt:
            valve_open = True
            break

    status = 1 if valve_open else 0
    unix_ts = int(now.timestamp())
    payload = {
        ATTR_VALVE: status,
        "hist": [{"n": ATTR_VALVE, "t": unix_ts, "v": status}],
    }
    _publish(client, payload)

    _last_seen["irrigation"] = now_iso
    return 1  # siempre publica 1 mensaje (el estado actual)


def poll_all(client: mqtt.Client) -> None:
    """Ejecuta un ciclo completo de polling."""
    now = datetime.now(timezone.utc)
    to_ts = now.isoformat()

    # Usamos la ventana desde el último poll (o now - POLL_INTERVAL al inicio)
    # con un margen de 60s para no perder datos por desfase de reloj
    from_ts = (
        datetime.fromisoformat(_last_seen["flow"].replace("Z", "+00:00"))
        - timedelta(seconds=60)
    ).isoformat()

    # Para irrigación, consultamos una ventana más amplia (últimos 2 días)
    # para capturar programas que empezaron antes pero siguen activos
    irr_from = (now - timedelta(days=2)).isoformat()

    n_flow = poll_flow(client, from_ts, to_ts)
    n_soil = poll_soil(client, from_ts, to_ts)
    n_irr  = poll_irrigation(client, irr_from, to_ts)

    log.info(
        "Poll completado: flow=%d, soil=%d, valve=ev40 → %s",
        n_flow, n_soil, "OPEN" if n_irr and True else "publicado",
    )


# ── MQTT ──────────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("Conectado a MQTT %s:%s", BROKER_HOST, BROKER_PORT)
    else:
        log.error("Conexión MQTT fallida (rc=%s)", rc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("API Bridge starting")
    log.info("  API base:  %s", API_BASE)
    log.info("  MQTT topic: %s", WICLOUDS_TOPIC)
    log.info("  Poll interval: %ds (%d min)", POLL_INTERVAL, POLL_INTERVAL // 60)
    log.info("  Flow:  ctrl=%s dev=%s", CTRL_FLOW, DEV_FLOW)
    log.info("  Soil:  ctrl=%s dev=%s", CTRL_SOIL, DEV_SOIL)

    if not API_TOKEN:
        log.error("WICLOUDS_API_TOKEN no configurado — el servicio no puede arrancar")
        return

    _init_last_seen()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"api-bridge-{os.getpid()}",
    )
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect = on_connect
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    # Esperar conexión MQTT
    time.sleep(2)

    tick = 0
    try:
        while True:
            tick += 1
            log.info("── Tick %d ──", tick)
            try:
                poll_all(client)
            except Exception:
                log.exception("Error en el ciclo de polling")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        log.info("API Bridge stopped")
    finally:
        client.loop_stop()
        client.disconnect()


def test_apis():
    """
    Modo test: consulta cada API una vez, imprime los resultados y sale.
    No necesita MQTT.

    Uso:
      python app.py --test
      python app.py --test flow
      python app.py --test soil
      python app.py --test irrigation
    """
    import sys

    target = sys.argv[2] if len(sys.argv) > 2 else "all"
    now = datetime.now(timezone.utc)
    from_ts = (now - timedelta(minutes=30)).isoformat()
    to_ts = now.isoformat()

    if target in ("all", "flow"):
        print(f"\n{'='*60}")
        print(f"FLOW — ctrl={CTRL_FLOW} dev={DEV_FLOW}")
        print(f"  Ventana: {from_ts} → {to_ts}")
        records = _fetch_historical(CTRL_FLOW, DEV_FLOW, from_ts, to_ts)
        print(f"  Registros: {len(records)}")
        for r in records[:5]:
            print(f"    {r['d']}  →  {r['v']}")
        if len(records) > 5:
            print(f"    ... y {len(records) - 5} más")

    if target in ("all", "soil"):
        print(f"\n{'='*60}")
        print(f"SOIL — ctrl={CTRL_SOIL} dev={DEV_SOIL}")
        print(f"  Ventana: {from_ts} → {to_ts}")
        records = _fetch_historical(CTRL_SOIL, DEV_SOIL, from_ts, to_ts)
        print(f"  Registros: {len(records)}")
        for r in records:
            print(f"    {r['d']}  →  {r['v']}")

    if target in ("all", "irrigation"):
        print(f"\n{'='*60}")
        irr_from = (now - timedelta(days=2)).isoformat()
        print(f"IRRIGATION — ctrl={CTRL_FLOW}")
        print(f"  Ventana: {irr_from} → {to_ts}")
        programs = _fetch_irrigation(CTRL_FLOW, irr_from, to_ts)
        print(f"  Programas: {len(programs)}")
        for p in programs[-5:]:
            inicio = p.get("inicioProgramaRiego", "?")
            fin = p.get("finProgramaRiego", "?")
            prog_id = p.get("idPrograma", "?")
            print(f"    Prog {prog_id}: {inicio} → {fin}")

        # Estado actual de la válvula
        valve_open = False
        for p in programs:
            inicio = p.get("inicioProgramaRiego", "")
            fin = p.get("finProgramaRiego", "")
            if inicio and fin:
                i = datetime.fromisoformat(inicio.replace("Z", "+00:00"))
                f = datetime.fromisoformat(fin.replace("Z", "+00:00"))
                if i <= now <= f:
                    valve_open = True
                    break
        print(f"\n  Válvula ahora: {'ABIERTA' if valve_open else 'CERRADA'}")

    print(f"\n{'='*60}")
    print("Test completado.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_apis()
    else:
        main()
