FROM ghcr.io/mlflow/mlflow:v2.15.1

RUN pip install --no-cache-dir psycopg2-binary
