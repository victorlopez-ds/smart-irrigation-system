"""
src/anomaly/pipeline.py — Pipeline completo de reentrenamiento (CLI).

Orquesta en un único proceso:
  1. train         — entrena el challenger y guarda test set
  2. export_onnx   — convierte challenger .keras → .onnx
  3. evaluate      — MAE de champion vs challenger sobre el mismo test set
  4. promote       — si MAE_challenger < MAE_champion × (1 - margin),
                     copia challenger sobre el modelo de producción

Uso:
    python -m src.anomaly.pipeline \
        --raw-watermeter /app/data/raw/contador_agua_albudeite.json \
        --raw-solenoid   /app/data/raw/electrovalvula_albudeite.json \
        --models-dir     /app/models \
        --improvement-margin 0.01 \
        --run-name anomaly_retrain_2026-04-25

Imprime al final un JSON con el resumen del run; lo recoge el DAG de Airflow.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def _onnx_mae(onnx_path: str, X_test: np.ndarray, y_test: np.ndarray) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    name = sess.get_inputs()[0].name
    y_pred = sess.run(None, {name: X_test.astype(np.float32)})[0].reshape(-1)
    return float(np.mean(np.abs(y_pred - y_test)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-watermeter",     required=True)
    ap.add_argument("--raw-solenoid",       required=True)
    ap.add_argument("--models-dir",         required=True)
    ap.add_argument("--improvement-margin", type=float, default=0.01)
    ap.add_argument("--run-name",           default="anomaly_retrain")
    ap.add_argument("--mlflow-uri",         default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    ap.add_argument("--model-name",         default="anomaly-detector")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    new_keras  = str(models_dir / "anomaly_cnn_lstm_new.keras")
    new_onnx   = str(models_dir / "anomaly_cnn_lstm_new.onnx")
    prod_onnx  = str(models_dir / "anomaly_cnn_lstm.onnx")
    scaler     = str(models_dir / "anomaly_scaler.pkl")
    testset    = str(models_dir / "anomaly_testset.npz")

    # ── 1) TRAIN ──────────────────────────────────────────────────────────────
    print("\n== STEP 1/4: train ==", flush=True)
    from src.anomaly.train import train as _train
    metrics = _train(
        raw_watermeter=args.raw_watermeter,
        raw_solenoid=args.raw_solenoid,
        model_output=new_keras,
        scaler_output=scaler,
        testset_output=testset,
    )
    print(f"Challenger entrenado. MAE interno: {metrics['mae']:.4f}", flush=True)

    # ── 2) EXPORT ONNX ────────────────────────────────────────────────────────
    print("\n== STEP 2/4: export_onnx ==", flush=True)
    from src.anomaly.export import export_keras_to_onnx
    export_keras_to_onnx(new_keras, new_onnx)

    # ── 3) EVALUATE (champion vs challenger sobre el mismo test set) ──────────
    print("\n== STEP 3/4: evaluate ==", flush=True)
    data = np.load(testset)
    X_test, y_test = data["X_test"], data["y_test"]
    challenger_mae = _onnx_mae(new_onnx, X_test, y_test)
    champion_mae   = _onnx_mae(prod_onnx, X_test, y_test) if os.path.exists(prod_onnx) else None
    print(f"champion_mae   = {champion_mae}", flush=True)
    print(f"challenger_mae = {challenger_mae:.4f}", flush=True)

    # ── 4) PROMOTE ────────────────────────────────────────────────────────────
    print("\n== STEP 4/4: promote ==", flush=True)
    if champion_mae is None:
        promoted = True
        reason = "no hay champion previo"
    else:
        threshold = champion_mae * (1 - args.improvement_margin)
        promoted = challenger_mae < threshold
        reason = (
            f"challenger MAE {challenger_mae:.4f} "
            f"{'<' if promoted else '>='} {threshold:.4f}"
        )

    if promoted:
        shutil.copy2(new_onnx, prod_onnx)
        print(f"✅ Promovido ({reason}): {prod_onnx}", flush=True)

        # Registro en MLflow
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            with mlflow.start_run(run_name=args.run_name):
                mlflow.log_metric("challenger_mae", challenger_mae)
                if champion_mae is not None:
                    mlflow.log_metric("champion_mae", champion_mae)
                mlflow.log_artifact(prod_onnx, artifact_path="model")
        except Exception as exc:
            print(f"Advertencia MLflow: {exc}", flush=True)
    else:
        print(f"❌ NO promovido ({reason})", flush=True)

    summary = {
        "promoted":       promoted,
        "reason":         reason,
        "champion_mae":   champion_mae,
        "challenger_mae": challenger_mae,
    }
    # Línea final con el JSON: el DAG la parsea
    print("\n__PIPELINE_RESULT__=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
