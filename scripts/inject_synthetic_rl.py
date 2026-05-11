"""
scripts/inject_synthetic_rl.py — Inyecta transiciones RL sintéticas en
DuckDB.rl_transitions_raw a través de ingest-svc /rl/transition.

Genera N días consecutivos (1 transición/día) consultando rl-api /act para
producir acciones realistas. La obs siguiente se construye:
  next_obs[0] = obs[0] + Δθ  (con Δθ aleatorio reflejando un riego razonable)
  next_obs[1] = action_t     (w_prev del día siguiente = riego de hoy)
  next_obs[2:] = obs[2:]     (E, P se mantienen — campo idealizado)

Uso:
  python scripts/inject_synthetic_rl.py --days 30 --start-date 2025-09-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
from typing import Sequence

import requests

INGEST = "http://localhost:8000"
RL_API = "http://localhost:8003"


def step(obs: Sequence[float]) -> tuple[float, list[float]]:
    """Devuelve (action_tau, next_obs) consultando rl-api."""
    r = requests.post(f"{RL_API}/act", json={"obs": list(obs)}, timeout=5)
    r.raise_for_status()
    body = r.json()
    tau = float(body["tau_minutes"])

    # Δθ heurístico para sintético: humedad sube con riego, baja con ET, sube con P
    theta_t = obs[0]
    et      = obs[2]
    pt      = obs[4]
    delta = 0.04 * tau - 0.6 * et + 0.4 * pt + random.uniform(-0.5, 0.5)
    theta_next = max(0.0, min(100.0, theta_t + delta))
    next_obs = [theta_next, tau, obs[3], obs[2], obs[5], obs[4]]
    return tau, next_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",       type=int, default=30)
    ap.add_argument("--start-date", default="2025-09-01")
    ap.add_argument("--device-id",  default="edge-synthetic")
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start_date).replace(tzinfo=dt.timezone.utc)
    obs = [28.0, 0.0, 6.5, 6.0, 0.0, 0.0]

    for d in range(args.days):
        ts = (start + dt.timedelta(days=d, hours=6)).isoformat()
        tau, next_obs = step(obs)
        body = {
            "ts": ts,
            "obs": list(obs),
            "action": tau,
            "device_id": args.device_id,
        }
        r = requests.post(f"{INGEST}/rl/transition", json=body, timeout=5)
        r.raise_for_status()
        print(f"[{d+1}/{args.days}] ts={ts} τ={tau:.1f}min θ_t={obs[0]:.1f} → θ_t+1={next_obs[0]:.1f}")
        obs = next_obs


if __name__ == "__main__":
    main()
