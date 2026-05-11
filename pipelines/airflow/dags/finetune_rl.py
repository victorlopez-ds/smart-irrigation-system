"""
DAG: finetune_rl — Fine-tune mensual del agente TQC con replay híbrido.

Programación: 1er día de cada mes a las 04:00.

Tareas:
  1. dump_real_transitions  — exporta DuckDB.rl_transitions a parquet en
                               /app/data/tmp/real_transitions.parquet
                               (consultando ingest-svc /rl/export).
  2. finetune_pipeline      — DockerOperator que ejecuta src.rl.finetune en
                               la imagen `irridea-rl-finetune`. Carga el .zip
                               actual, mezcla reales + sim con --real-ratio
                               y, si Δreturn ≥ margen, reescribe el .zip y
                               re-exporta el .onnx.
  3. notify_distributor     — el sync-agent del edge se actualiza solo en su
                               próximo tick horario; aquí solo reportamos.

Requisitos:
  - Imagen `irridea-rl-finetune` construida en el host.
  - HOST_PROJECT_DIR en el contenedor de Airflow.
  - best_model_rl.zip presente en /opt/airflow/project/models.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from airflow.decorators import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

DEFAULT_ARGS = {
    "owner": "irridea",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
    "email_on_failure": False,
}

HOST_PROJECT_DIR = os.getenv("HOST_PROJECT_DIR", "/opt/airflow/project")
HOST_MODELS = f"{HOST_PROJECT_DIR}/models"
HOST_DATA   = f"{HOST_PROJECT_DIR}/data"

INGEST_URL = "http://ingest-svc:8000"

# Ruta dentro del contenedor de Airflow ⇄ contenedor de fine-tune
PARQUET_REL_PATH = "tmp/rl_real_transitions.parquet"


@dag(
    dag_id="finetune_rl",
    default_args=DEFAULT_ARGS,
    schedule="0 4 1 * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rl", "finetune", "irridea"],
    description="Fine-tune mensual del actor TQC con replay híbrido sim+real",
)
def finetune_rl():

    @task
    def dump_real_transitions() -> dict:
        """Descarga las transiciones completas y las vuelca a parquet."""
        r = requests.get(f"{INGEST_URL}/rl/export", timeout=60)
        r.raise_for_status()
        rows = r.json().get("rows", [])
        df = pd.DataFrame(rows)
        out_path = f"/opt/airflow/project/data/{PARQUET_REL_PATH}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Volcadas {len(df)} transiciones a {out_path}")
        return {"path": out_path, "n": len(df)}

    finetune_pipeline = DockerOperator(
        task_id="finetune_pipeline",
        image="irridea-rl-finetune:latest",
        api_version="auto",
        auto_remove=True,
        force_pull=False,
        docker_url="unix://var/run/docker.sock",
        network_mode="irridea-net",
        mount_tmp_dir=False,
        mounts=[
            Mount(source=HOST_MODELS, target="/app/models", type="bind"),
            Mount(source=HOST_DATA,   target="/app/data",   type="bind", read_only=False),
            Mount(source="deployment_mlflow-artifacts", target="/mlflow-artifacts", type="volume"),
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "PYTHONUNBUFFERED": "1",
        },
        command=[
            "python", "-m", "src.rl.finetune",
            "--base-model",     "/app/models/best_model_rl.zip",
            "--real-parquet",   f"/app/data/{PARQUET_REL_PATH}",
            "--moisture-model", "/app/models/moisture_mlp.keras",
            "--moisture-sx",    "/app/models/moisture_scaler_x.pkl",
            "--moisture-sy",    "/app/models/moisture_scaler_y.pkl",
            "--dataset",        "/app/data/processed/dataset_diario.csv",
            "--output-zip",     "/app/models/best_model_rl.zip",
            "--output-onnx",    "/app/models/rl_actor.onnx",
            "--total-steps",    "50000",
            "--real-ratio",     "0.5",
            "--eval-episodes",  "20",
            "--improvement-margin", "0.01",
            "--warmup-episodes", "50",
        ],
        do_xcom_push=True,
    )

    @task
    def report(container_logs: str | None = None) -> dict:
        summary = {"promoted": False, "reason": "no result line"}
        if container_logs:
            for line in container_logs.splitlines():
                if "__FINETUNE_RESULT__=" in line:
                    summary = json.loads(line.split("__FINETUNE_RESULT__=", 1)[1])
                    break
        print(f"Resumen fine-tune: {summary}")
        if summary.get("promoted"):
            print("✅ Nuevo actor promovido. El sync-agent del edge "
                  "lo recogerá en su próximo tick (cron horario :07).")
        else:
            print("ℹ️  Sin promoción; modelo en producción intacto.")
        return summary

    dump = dump_real_transitions()
    dump >> finetune_pipeline
    report(finetune_pipeline.output)


finetune_rl()
