"""
DAG: ingest_dag — Agregación diaria de datos MQTT a features_daily.

Programación: 23:55 todos los días.

Tareas:
  1. fetch_weather — POST http://ingest-svc:8000/weather/fetch
  2. aggregate    — POST http://ingest-svc:8000/aggregate?date=ds
  3. validate     — comprueba que la fila insertada tiene valores coherentes.

Ya no necesitamos parar/arrancar el worker: el ingest-svc es ahora el único
proceso que toca DuckDB y lo expone vía HTTP.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "irridea",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

INGEST_URL = "http://ingest-svc:8000"


@dag(
    dag_id="ingest_dag",
    default_args=DEFAULT_ARGS,
    schedule="55 23 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ingest", "irridea"],
    description="Agrega lecturas MQTT diarias en features_daily",
)
def ingest_dag():

    @task
    def fetch_weather() -> dict:
        import requests
        r = requests.post(f"{INGEST_URL}/weather/fetch", timeout=30)
        r.raise_for_status()
        result = r.json()
        print(f"AEMET fetch: {result}")
        return result

    @task
    def aggregate(ds: str) -> dict:
        import requests
        r = requests.post(f"{INGEST_URL}/aggregate", params={"date": ds}, timeout=60)
        r.raise_for_status()
        result = r.json()
        print(f"Agregación {ds}: {result}")
        return result

    @task
    def validate(result: dict) -> None:
        theta = result.get("theta")
        water = result.get("water_used_l")

        if theta is not None and not (0 <= theta <= 100):
            raise ValueError(f"Humedad fuera de rango: {theta}")
        if water is not None and water < 0:
            raise ValueError(f"Consumo de agua negativo: {water}")

        if theta is not None:
            print(f"✅ Validación OK: {result['date']}  θ={theta:.1f}%  agua={water:.0f}L")
        else:
            print(f"✅ Validación OK: {result['date']}  θ=N/A  agua={water:.0f}L")

    fetch_weather() >> validate(aggregate())


ingest_dag()
