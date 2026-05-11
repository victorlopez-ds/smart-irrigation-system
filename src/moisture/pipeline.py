"""
src/moisture/pipeline.py — Pipeline completo de reentrenamiento (CLI).

Orquesta en un único proceso:
  1. build_dataset — reconstruye data/processed/dataset_diario.csv desde raw
                     (si no se pasa --skip-build)
  2. train         — entrena el challenger (MLP) y guarda test set
  3. export_onnx   — convierte challenger .keras → .onnx
  4. evaluate      — MAE de champion vs challenger sobre el mismo test set
  5. promote       — si MAE_challenger < MAE_champion × (1 - margin),
                     copia challenger sobre el modelo de producción

Uso:
    python -m src.moisture.pipeline \
        --raw-dir       /app/data/raw \
        --processed-dir /app/data/processed \
        --models-dir    /app/models \
        --improvement-margin 0.01 \
        --run-name      moisture_retrain_2026-04-25

Imprime al final un JSON con el resumen del run; lo recoge el DAG de Airflow.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np


def _onnx_mae(
    onnx_path: str,
    scaler_x_path: str,
    scaler_y_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Evalúa MAE de un MLP exportado a ONNX.

    El test set se almacena SIN ESCALAR (en unidades reales). Para evaluar:
      X_scaled = scaler_x.transform(X_test)
      y_pred_scaled = onnx.run(X_scaled)
      y_pred = scaler_y.inverse_transform(y_pred_scaled)
    """
    import onnxruntime as ort

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    sess = ort.InferenceSession(onnx_path)
    name = sess.get_inputs()[0].name

    X_scaled = scaler_x.transform(X_test).astype(np.float32)
    y_pred_scaled = sess.run(None, {name: X_scaled})[0]
    y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
    return float(np.mean(np.abs(y_pred - y_test)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir",       default="/app/data/raw")
    ap.add_argument("--processed-dir", default="/app/data/processed")
    ap.add_argument("--models-dir",    required=True)
    ap.add_argument("--skip-build",    action="store_true",
                    help="No reconstruir dataset_diario.csv (usar el existente)")
    ap.add_argument("--improvement-margin", type=float, default=0.01)
    ap.add_argument("--run-name",      default="moisture_retrain")
    ap.add_argument("--mlflow-uri",    default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    ap.add_argument("--model-name",    default="moisture-predictor")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = str(processed_dir / "dataset_diario.csv")

    # Modelo en producción (champion) — convención: nombres "limpios"
    prod_keras       = str(models_dir / "moisture_mlp.keras")
    prod_onnx        = str(models_dir / "moisture_mlp.onnx")
    prod_scaler_x    = str(models_dir / "moisture_scaler_x.pkl")
    prod_scaler_y    = str(models_dir / "moisture_scaler_y.pkl")

    # Challenger (sufijo _new)
    new_keras        = str(models_dir / "moisture_mlp_new.keras")
    new_onnx         = str(models_dir / "moisture_mlp_new.onnx")
    new_scaler_x     = str(models_dir / "moisture_scaler_x_new.pkl")
    new_scaler_y     = str(models_dir / "moisture_scaler_y_new.pkl")
    testset          = str(models_dir / "moisture_testset.npz")

    # Apuntar MLflow
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri

    # ── 0) BUILD DATASET ──────────────────────────────────────────────────────
    if args.skip_build:
        print(f"\n== STEP 0/5: build_dataset SKIP (uso {dataset_csv}) ==", flush=True)
    else:
        print("\n== STEP 0/5: build_dataset ==", flush=True)
        from src.common.data_loader import build_daily_dataset
        build_daily_dataset(raw_dir=args.raw_dir, output_path=dataset_csv)

    # ── 1) TRAIN ──────────────────────────────────────────────────────────────
    print("\n== STEP 1/4: train ==", flush=True)
    from src.moisture.train import train as _train
    metrics = _train(
        data_path=dataset_csv,
        model_output=new_keras,
        scaler_x_output=new_scaler_x,
        scaler_y_output=new_scaler_y,
        testset_output=testset,
    )
    print(
        f"Challenger entrenado. R²={metrics['r2']:.4f}  "
        f"MAE={metrics['mae']:.4f}  epochs={metrics['epochs']}",
        flush=True,
    )

    # ── 2) EXPORT ONNX ────────────────────────────────────────────────────────
    print("\n== STEP 2/4: export_onnx ==", flush=True)
    from src.moisture.export import export_mlp_to_onnx
    export_mlp_to_onnx(new_keras, new_onnx)

    # ── 3) EVALUATE (champion vs challenger sobre el mismo test set) ──────────
    print("\n== STEP 3/4: evaluate ==", flush=True)
    data = np.load(testset)
    X_test, y_test = data["X_test"], data["y_test"]
    challenger_mae = _onnx_mae(new_onnx, new_scaler_x, new_scaler_y, X_test, y_test)
    if (
        os.path.exists(prod_onnx)
        and os.path.exists(prod_scaler_x)
        and os.path.exists(prod_scaler_y)
    ):
        champion_mae = _onnx_mae(prod_onnx, prod_scaler_x, prod_scaler_y, X_test, y_test)
    else:
        champion_mae = None
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
        shutil.copy2(new_onnx,     prod_onnx)
        shutil.copy2(new_keras,    prod_keras)
        shutil.copy2(new_scaler_x, prod_scaler_x)
        shutil.copy2(new_scaler_y, prod_scaler_y)
        print(f"✅ Promovido ({reason}): {prod_onnx}", flush=True)

        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            with mlflow.start_run(run_name=args.run_name):
                mlflow.log_metric("challenger_mae", challenger_mae)
                mlflow.log_metric("challenger_r2",  metrics["r2"])
                if champion_mae is not None:
                    mlflow.log_metric("champion_mae", champion_mae)
                mlflow.log_artifact(prod_onnx,     artifact_path="model")
                mlflow.log_artifact(prod_scaler_x, artifact_path="model")
                mlflow.log_artifact(prod_scaler_y, artifact_path="model")
        except Exception as exc:
            print(f"Advertencia MLflow: {exc}", flush=True)
    else:
        print(f"❌ NO promovido ({reason})", flush=True)

    summary = {
        "promoted":       promoted,
        "reason":         reason,
        "champion_mae":   champion_mae,
        "challenger_mae": challenger_mae,
        "challenger_r2":  metrics["r2"],
        "epochs":         metrics["epochs"],
    }
    print("\n__PIPELINE_RESULT__=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
