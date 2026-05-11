"""
services/rl_api/app.py — API de inferencia del agente TQC (ONNX).

El modelo ONNX YA incluye:
  - Normalización VecNormalize (mean/var congelados durante export)
  - Forward determinista del actor (latent_pi → mu → tanh)
  - Des-escalado de la acción a unidades reales [0, tau_max]

Por tanto rl-api solo recibe la observación cruda y devuelve los minutos
de riego (τ).

Endpoints:
  GET  /health       — liveness + estado del modelo + dim de observación
  POST /act          — inferencia. Acepta dos formas equivalentes:
                       a) {"obs":[θ, w_prev, E_t, ..., E_{t+N-1}, P_t, ..., P_{t+N-1}]}
                       b) {"theta":θ, "w_prev":w, "E":[..,..], "P":[..,..]}
                       responde: {"tau_minutes":τ, "liters":τ·q}
  POST /reload       — recarga el ONNX desde disco

Variables de entorno:
  MODEL_PATH   ruta al .onnx (default /app/models/rl_actor.onnx)
  TAU_MAX      límite superior τ (default 180.0)  — solo para info /health
  Q_LH         litros/min de la electroválvula (default 12.57)
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("rl-api")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/rl_actor.onnx")
TAU_MAX    = float(os.getenv("TAU_MAX", "180.0"))
Q_LH       = float(os.getenv("Q_LH",    "12.57"))

# Ingest-svc para loggear transiciones (replay buffer del fine-tune).
# En el edge real esto apuntaría a la URL del central. Si está vacío, se omite.
INGEST_URL = os.getenv("INGEST_URL", "").rstrip("/")
DEVICE_ID  = os.getenv("DEVICE_ID", "edge-default")
_LOG_POOL  = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rl-log")


def _log_transition_async(ts_iso: str, obs: list[float], action: float) -> None:
    """Fire-and-forget. No bloquea /act ni propaga errores al cliente."""
    if not INGEST_URL:
        return
    def _send():
        try:
            requests.post(
                f"{INGEST_URL}/rl/transition",
                json={"ts": ts_iso, "obs": obs, "action": action,
                      "device_id": DEVICE_ID},
                timeout=2.0,
            )
        except Exception as exc:   # noqa: BLE001 — best-effort logging
            log.warning("No se pudo loggear transición: %s", exc)
    _LOG_POOL.submit(_send)


class ModelHolder:
    def __init__(self):
        self._lock = threading.RLock()
        self._sess: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._obs_dim: int | None = None
        self._version: str | None = None
        self.load()

    def load(self) -> bool:
        with self._lock:
            if not Path(MODEL_PATH).exists():
                log.warning("Modelo no encontrado: %s", MODEL_PATH)
                self._sess = None
                return False
            self._sess = ort.InferenceSession(MODEL_PATH)
            inp = self._sess.get_inputs()[0]
            self._input_name = inp.name
            # shape suele ser ['batch', obs_dim]
            self._obs_dim = int(inp.shape[1]) if isinstance(inp.shape[1], int) else None
            stat = Path(MODEL_PATH).stat()
            self._version = f"{int(stat.st_mtime)}-{stat.st_size}"
            log.info("Modelo cargado (version=%s, obs_dim=%s)",
                     self._version, self._obs_dim)
            return True

    @property
    def ready(self) -> bool:
        return self._sess is not None

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def obs_dim(self) -> int | None:
        return self._obs_dim

    def act(self, obs: np.ndarray) -> float:
        if not self.ready:
            raise RuntimeError("model not loaded")
        x = obs.astype(np.float32).reshape(1, -1)
        action = self._sess.run(None, {self._input_name: x})[0]
        # Salvaguarda: clip por si llegan obs fuera del rango entrenado
        return float(np.clip(action[0, 0], 0.0, TAU_MAX))


model = ModelHolder()
app = Flask(__name__)


def _build_obs_from_named(body: dict) -> np.ndarray:
    """
    Construye la observación a partir de campos con nombre:
        theta, w_prev, E:[E_t, E_{t+1}, ...], P:[P_t, P_{t+1}, ...]
    Ordenado como en RiegoEnv._get_obs():
        [θ, w_prev, E_t, ..., E_{t+N-1}, P_t, ..., P_{t+N-1}]
    """
    theta  = float(body["theta"])
    w_prev = float(body["w_prev"])
    E = list(map(float, body["E"]))
    P = list(map(float, body["P"]))
    if len(E) != len(P):
        raise ValueError("E y P deben tener la misma longitud (N)")
    return np.array([theta, w_prev, *E, *P], dtype=np.float32)


@app.get("/health")
def health():
    return jsonify(
        status="ok" if model.ready else "not_ready",
        model_version=model.version,
        model_path=MODEL_PATH,
        obs_dim=model.obs_dim,
        tau_max=TAU_MAX,
        q_lh=Q_LH,
    )


@app.post("/act")
def act():
    if not model.ready:
        return jsonify(error="model not loaded"), 503
    body = request.get_json(force=True)
    try:
        if "obs" in body:
            obs = np.asarray(body["obs"], dtype=np.float32)
        else:
            obs = _build_obs_from_named(body)
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify(error=f"bad input: {exc}"), 400

    if model.obs_dim is not None and obs.shape[0] != model.obs_dim:
        return jsonify(
            error=f"obs has {obs.shape[0]} dims, expected {model.obs_dim}"
        ), 400

    tau = model.act(obs)
    obs_list = obs.tolist()
    ts_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    _log_transition_async(ts_iso, obs_list, tau)

    return jsonify(
        tau_minutes=tau,
        liters=tau * Q_LH,
        obs=obs_list,
        ts=ts_iso,
    )


@app.post("/reload")
def reload_model():
    ok = model.load()
    return jsonify(
        status="ok" if ok else "not_ready",
        model_version=model.version,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8003)
