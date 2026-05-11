"""
DAG: retrain_moisture — Reentrenamiento mensual del predictor MLP de humedad.

Programación: 1er día de cada mes a las 03:00.

Tareas:
  1. train_pipeline   — DockerOperator que ejecuta `src.moisture.pipeline`
                        (build_dataset → train → export_onnx → evaluate →
                        promote) en una imagen dedicada
                        (`irridea-train-moisture`). Airflow no necesita
                        TF/sklearn/onnx instalados.
  2. notify_reload    — POST a moisture-api/reload tras una promoción.

Requisitos:
  - Imagen `irridea-train-moisture` construida en el host.
  - Variable de entorno HOST_PROJECT_DIR en el contenedor de Airflow.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import requests
from airflow.decorators import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

DEFAULT_ARGS = {
    "owner": "irridea",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

HOST_PROJECT_DIR = os.getenv("HOST_PROJECT_DIR", "/opt/airflow/project")
HOST_MODELS = f"{HOST_PROJECT_DIR}/models"
HOST_DATA   = f"{HOST_PROJECT_DIR}/data"

MOISTURE_API_URL = os.getenv("MOISTURE_API_URL", "http://moisture-api:8002")


@dag(
    dag_id="retrain_moisture",
    default_args=DEFAULT_ARGS,
    schedule="0 3 1 * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["retrain", "moisture", "irridea"],
    description="Reentrenamiento mensual del predictor MLP de humedad",
)
def retrain_moisture():

    train_pipeline = DockerOperator(
        task_id="train_pipeline",
        image="irridea-train-moisture:latest",
        api_version="auto",
        auto_remove=True,
        force_pull=False,
        docker_url="unix://var/run/docker.sock",
        network_mode="irridea-net",
        mount_tmp_dir=False,
        mounts=[
            Mount(source=HOST_MODELS, target="/app/models", type="bind"),
            Mount(source=HOST_DATA,   target="/app/data",   type="bind"),
            Mount(source="deployment_mlflow-artifacts", target="/mlflow-artifacts", type="volume"),
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "PYTHONUNBUFFERED": "1",
        },
        command=[
            "python", "-m", "src.moisture.pipeline",
            "--raw-dir",       "/app/data/raw",
            "--processed-dir", "/app/data/processed",
            "--models-dir",    "/app/models",
            "--improvement-margin", "0.01",
            "--run-name",      "moisture_retrain_{{ ds }}",
        ],
        do_xcom_push=True,
    )

    @task
    def notify_reload(container_logs: str | None = None) -> dict:
        """Parsea __PIPELINE_RESULT__=… y, si hubo promoción, llama /reload."""
        summary = {"promoted": False, "reason": "no result line"}
        if container_logs:
            for line in container_logs.splitlines():
                if "__PIPELINE_RESULT__=" in line:
                    summary = json.loads(line.split("__PIPELINE_RESULT__=", 1)[1])
                    break
        print(f"Resumen pipeline: {summary}")

        if summary.get("promoted"):
            try:
                r = requests.post(f"{MOISTURE_API_URL}/reload", timeout=10)
                r.raise_for_status()
                print(f"moisture-api recargado: {r.json()}")
            except Exception as exc:
                print(f"Advertencia: fallo al recargar moisture-api: {exc}")
        else:
            print("No hubo promoción; no se recarga la API.")
        return summary

    notify_reload(train_pipeline.output)


retrain_moisture()
