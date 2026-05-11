"""
DAG: build_rl_transitions — Empareja transiciones crudas RL con θ_{t+1}.

Cada día rl-api loggea (s_t, a_t) en DuckDB.rl_transitions_raw. La recompensa
solo puede calcularse cuando se conoce θ del día siguiente. Este DAG corre a
las 23:55 y empareja:

  rl_transitions_raw[t]   →   rl_transitions_raw[t+δ]   (siguiente del mismo device)

reward = compute_reward(next_theta=obs_{t+1}[0], tau_minutes=action_t).

La operación es idempotente (PRIMARY KEY (ts, device_id) en rl_transitions),
así que reejecutar para el mismo día no rompe nada.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import requests
from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "irridea",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

INGEST_URL = "http://ingest-svc:8000"


@dag(
    dag_id="build_rl_transitions",
    default_args=DEFAULT_ARGS,
    schedule="55 23 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rl", "irridea"],
    description="Empareja s_t con s_{t+1} y calcula reward para fine-tune RL",
)
def build_rl_transitions():

    @task
    def call_build(target_date: str = "{{ ds }}") -> dict:
        r = requests.post(
            f"{INGEST_URL}/rl/build",
            params={"date": target_date},
            timeout=60,
        )
        r.raise_for_status()
        result = r.json()
        print(f"build_rl_transitions resultado: {result}")
        return result

    call_build()


build_rl_transitions()
