# ──────────────────────────────────────────────────────────────────────────────
# airflow.Dockerfile — Imagen base de Airflow con dependencias pre-instaladas.
#
# Sustituye al patrón `_PIP_ADDITIONAL_REQUIREMENTS` (que reinstala los
# paquetes en cada arranque de contenedor): aquí se hace una vez al build.
#
# Paquetes:
#   - duckdb, pandas              → consumidos por ingest_dag / build_rl_transitions
#   - docker, providers-docker    → necesarios por DockerOperator
#
# mlflow NO se incluye: los DAGs no lo importan; los pipelines de entrenamiento
# que sí lo usan corren en imágenes propias (irridea-train-*).
# ──────────────────────────────────────────────────────────────────────────────

FROM apache/airflow:2.9.3-python3.11

RUN pip install --no-cache-dir \
        duckdb==1.0.0 \
        pandas==2.2.2 \
        docker==7.1.0 \
        apache-airflow-providers-docker==3.12.0
