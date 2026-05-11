"""
scripts/seed_db.py — Pobla irridea.duckdb con datos ficticios realistas.

Genera 90 días de lecturas cada 15 minutos para:
  - raw_soil   : humedad del suelo + temperatura
  - raw_flow   : volumen acumulado del contador de agua
  - raw_valve  : estado de la electroválvula (0/1)
  - features_daily : agregación diaria (resultado del DAG ingest)

Uso:
    python scripts/seed_db.py [--days 90] [--db data/stream/irridea.duckdb]
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

# ── Parámetros del generador ───────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

INTERVAL_MIN  = 15          # minutos entre lecturas
RIEGO_HORA    = 7           # hora a la que se riega (07:00)
RIEGO_DIAS    = [0, 2, 4]   # lunes, miércoles, viernes (weekday)


def generar_temperatura(dt: datetime) -> float:
    """Temperatura ambiente con ciclo diario y estacional."""
    dia_del_año = dt.timetuple().tm_yday
    hora        = dt.hour + dt.minute / 60

    base_temp   = 20 + 10 * math.sin(2 * math.pi * (dia_del_año - 80) / 365)
    ciclo_dia   = 7 * math.sin(2 * math.pi * (hora - 6) / 24)
    ruido       = random.gauss(0, 0.5)

    return round(base_temp + ciclo_dia + ruido, 2)


def generar_humedad(humedad_prev: float, riego: bool, temp: float) -> float:
    """Humedad del suelo con evapotranspiración y efecto del riego."""
    et  = 0.02 + 0.003 * max(0, temp - 15)          # evapotranspiración (cuarto de hora)
    inc = 3.5 if riego else 0.0                       # incremento por riego
    ruido = random.gauss(0, 0.15)
    nueva = humedad_prev - et + inc + ruido
    return round(max(15.0, min(55.0, nueva)), 2)      # clamp físico


def generar_vol_acumulado(vol_prev: float, valvula_abierta: bool) -> float:
    """Contador de agua acumulado (litros). Solo sube cuando la válvula está abierta."""
    if valvula_abierta:
        caudal = random.gauss(12.5, 0.8)              # ~12.5 L/min × 15 min ≈ 187 L
        return round(vol_prev + caudal * INTERVAL_MIN, 1)
    return vol_prev


def main(days: int, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    # Limpiar datos previos de prueba
    for tabla in ("raw_soil", "raw_flow", "raw_valve", "features_daily"):
        con.execute(f"DELETE FROM {tabla}")
    print(f"Tablas limpiadas. Generando {days} días de datos...")

    ahora      = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    inicio     = ahora - timedelta(days=days)
    ts         = inicio

    humedad    = random.uniform(28.0, 35.0)
    vol        = random.uniform(5000, 8000)

    soil_rows  = []
    flow_rows  = []
    valve_rows = []

    while ts < ahora:
        temp         = generar_temperatura(ts)
        es_dia_riego = ts.weekday() in RIEGO_DIAS
        es_hora_riego = ts.hour == RIEGO_HORA and ts.minute < INTERVAL_MIN
        valvula      = 1 if (es_dia_riego and es_hora_riego) else 0

        humedad = generar_humedad(humedad, valvula == 1, temp)
        vol     = generar_vol_acumulado(vol, valvula == 1)

        soil_rows.append((ts, humedad, temp))
        flow_rows.append((ts, vol, None))
        valve_rows.append((ts, valvula))

        ts += timedelta(minutes=INTERVAL_MIN)

    # Insertar en bloque
    con.executemany("INSERT INTO raw_soil  VALUES (?, ?, ?)", soil_rows)
    con.executemany("INSERT INTO raw_flow  VALUES (?, ?, ?)", flow_rows)
    con.executemany("INSERT INTO raw_valve VALUES (?, ?)",   valve_rows)

    print(f"  raw_soil  : {len(soil_rows):>6} filas")
    print(f"  raw_flow  : {len(flow_rows):>6} filas")
    print(f"  raw_valve : {len(valve_rows):>6} filas")

    # Calcular vol_diff (diferencia entre lecturas consecutivas)
    con.execute("""
        UPDATE raw_flow
        SET vol_diff = vol_l - LAG(vol_l) OVER (ORDER BY ts)
        WHERE vol_diff IS NULL
    """)

    # Agregar features_daily para cada día completo
    fecha = inicio.date()
    hoy   = ahora.date()
    daily_rows = 0

    while fecha < hoy:
        fecha_str = fecha.isoformat()

        theta = con.execute(f"""
            SELECT AVG(soil_moisture) FROM raw_soil
            WHERE CAST(ts AS DATE) = '{fecha_str}'
        """).fetchone()[0]

        water = con.execute(f"""
            SELECT COALESCE(SUM(vol_diff), 0) FROM raw_flow
            WHERE CAST(ts AS DATE) = '{fecha_str}' AND vol_diff > 0
        """).fetchone()[0]

        con.execute("""
            INSERT OR REPLACE INTO features_daily (date, theta, water_used_l)
            VALUES (?, ?, ?)
        """, [fecha_str, theta, water])

        daily_rows += 1
        fecha += timedelta(days=1)

    print(f"  features_daily: {daily_rows:>4} filas")
    con.close()
    print("\n✅ Base de datos poblada correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--db",   type=str, default="data/stream/irridea.duckdb")
    args = parser.parse_args()
    main(args.days, Path(args.db))
