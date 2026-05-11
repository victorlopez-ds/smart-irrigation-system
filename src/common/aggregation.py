"""
src/common/aggregation.py — Agregación diaria de lecturas MQTT en features_daily.

Separado de mqtt_subscriber.py para evitar la dependencia de paho-mqtt
en contextos donde solo se necesita la lógica de agregación (ej. Airflow).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data/stream"))
DB_PATH  = DATA_DIR / "irridea.duckdb"


def aggregate_daily(target_date: str | None = None, db_path: Path = DB_PATH) -> None:
    """
    Agrega las lecturas 15-min del día `target_date` (YYYY-MM-DD) en
    features_daily. Si target_date es None, usa ayer.
    """
    import datetime as dt

    if target_date is None:
        target_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    con = duckdb.connect(str(db_path))

    theta = con.execute(f"""
        SELECT AVG(soil_moisture)
        FROM raw_soil
        WHERE CAST(ts AS DATE) = '{target_date}'
    """).fetchone()[0]

    water_used = con.execute(f"""
        SELECT COALESCE(SUM(vol_diff), 0)
        FROM raw_flow
        WHERE CAST(ts AS DATE) = '{target_date}' AND vol_diff > 0
    """).fetchone()[0] or 0.0

    log.info("Agregación diaria %s: θ=%.1f%%, agua=%.0f L", target_date, theta or 0, water_used)

    con.execute("""
        INSERT OR REPLACE INTO features_daily (date, theta, water_used_l)
        VALUES (?, ?, ?)
    """, [target_date, theta, water_used])
    con.close()
