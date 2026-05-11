"""
src/anomaly/detect_window.py — Detección de anomalías sobre la última ventana.

Pensado para ejecutarse desde el DAG `detect_anomaly` cada 15 minutos. Hace:

  1. Lee de DuckDB las últimas `window` filas de raw_valve (status) y la última
     fila de raw_flow (vol_diff actual).
  2. Carga el modelo ONNX de producción y el StandardScaler.
  3. Llama a AnomalyDetector.detect → genera AnomalyResult.
  4. Si es anomalía, inserta una fila en la tabla anomaly_alerts.

Imprime el resultado por stdout (lo recoge el DAG).

Variables de entorno:
  DATA_DIR        — directorio que contiene irridea.duckdb (default /app/data)
  MODEL_PATH      — ruta al modelo .onnx
  SCALER_PATH     — ruta al .pkl del scaler
  ANOMALY_WINDOW  — tamaño de ventana (default 30)
  ANOMALY_THRESH  — umbral de error (default 0.25)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

DATA_DIR    = Path(os.getenv("DATA_DIR", "/app/data"))
DB_PATH     = DATA_DIR / "irridea.duckdb"
MODEL_PATH  = os.getenv("MODEL_PATH", "/app/models/anomaly_cnn_lstm.onnx")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/models/anomaly_scaler.pkl")
WINDOW      = int(os.getenv("ANOMALY_WINDOW", "30"))
THRESHOLD   = float(os.getenv("ANOMALY_THRESH", "0.25"))


def main() -> int:
    if not Path(MODEL_PATH).exists():
        print(json.dumps({"status": "skipped", "reason": f"model not found at {MODEL_PATH}"}))
        return 0

    con = duckdb.connect(str(DB_PATH))

    # Tabla de alertas (se crea en la primera ejecución)
    con.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            ts            TIMESTAMP,
            error         DOUBLE,
            threshold     DOUBLE,
            predicted_l   DOUBLE,
            actual_l      DOUBLE
        )
    """)

    # Última ventana de status (electroválvula)
    rows = con.execute(f"""
        SELECT ts, status FROM raw_valve
        ORDER BY ts DESC LIMIT {WINDOW}
    """).fetchall()

    if len(rows) < WINDOW:
        print(json.dumps({
            "status": "skipped",
            "reason": f"insufficient history ({len(rows)}/{WINDOW})",
        }))
        con.close()
        return 0

    rows.reverse()  # cronológico
    status_window = np.array([r[1] for r in rows], dtype=np.float32).reshape(WINDOW, 1)
    last_ts = rows[-1][0]

    # vol_diff actual (alineado por timestamp con la última lectura de válvula)
    vol_row = con.execute(
        "SELECT vol_diff FROM raw_flow WHERE ts = ? LIMIT 1", [last_ts]
    ).fetchone()
    if vol_row is None or vol_row[0] is None:
        print(json.dumps({
            "status": "skipped",
            "reason": f"no vol_diff for ts={last_ts}",
        }))
        con.close()
        return 0
    actual_vol = float(vol_row[0])

    # Inferencia
    from src.anomaly.infer import AnomalyDetector
    detector = AnomalyDetector(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        threshold=THRESHOLD,
        window=WINDOW,
    )
    actual_scaled = float(detector.scaler.transform([[actual_vol]])[0, 0])
    result = detector.detect(status_window, actual_scaled)

    payload = {
        "ts":            str(last_ts),
        "is_anomaly":    result.is_anomaly,
        "error":         result.error,
        "threshold":     result.threshold,
        "predicted_l":   result.predicted_volume,
        "actual_l":      result.actual_volume,
    }

    if result.is_anomaly:
        con.execute(
            "INSERT INTO anomaly_alerts VALUES (?, ?, ?, ?, ?)",
            [last_ts, result.error, result.threshold,
             result.predicted_volume, result.actual_volume],
        )
        payload["status"] = "ANOMALY"
    else:
        payload["status"] = "ok"

    con.close()
    print(json.dumps(payload, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
