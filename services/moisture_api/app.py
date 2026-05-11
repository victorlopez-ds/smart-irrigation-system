"""
services/moisture_api/app.py — API de inferencia del predictor de humedad (MLP).

El predictor estima ΔV_t = V_{t+1} − V_t. Sirve como herramienta de
inspección y para consultas "what-if" del simulador (drydown, etc.).
No participa en el lazo de control en tiempo real.

Endpoints:
  GET  /health        — liveness + estado del modelo
  POST /predict       — inferencia sin estado
                        body: {"vt":..,"rt":..,"rt_lag":..,"pt":..,"et":..}
                        respuesta: {"delta_v":..,"vt_next":..}
  POST /drydown       — simula descenso de humedad sin riego (sanity check)
                        body: {"vt_init":30,"et_constant":6,"n_days":30}
  POST /reload        — recarga el modelo ONNX y los scalers desde disco

Variables de entorno:
  MODEL_PATH      ruta al .onnx     (default /app/models/moisture_mlp.onnx)
  SCALER_X_PATH   ruta al scaler X  (default /app/models/moisture_scaler_x.pkl)
  SCALER_Y_PATH   ruta al scaler Y  (default /app/models/moisture_scaler_y.pkl)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("moisture-api")

MODEL_PATH    = os.getenv("MODEL_PATH",    "/app/models/moisture_mlp.onnx")
SCALER_X_PATH = os.getenv("SCALER_X_PATH", "/app/models/moisture_scaler_x.pkl")
SCALER_Y_PATH = os.getenv("SCALER_Y_PATH", "/app/models/moisture_scaler_y.pkl")

FEATURES = ["vt", "rt", "rt_lag", "pt", "et"]


class ModelHolder:
    """Mantiene la sesión ONNX + scalers, recargable bajo lock."""

    def __init__(self):
        self._lock = threading.RLock()
        self._sess: ort.InferenceSession | None = None
        self._sx = None
        self._sy = None
        self._input_name: str | None = None
        self._version: str | None = None
        self.load()

    def load(self) -> bool:
        with self._lock:
            for p in (MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH):
                if not Path(p).exists():
                    log.warning("Artefacto faltante: %s", p)
                    self._sess = None
                    return False
            self._sess = ort.InferenceSession(MODEL_PATH)
            self._input_name = self._sess.get_inputs()[0].name
            self._sx = joblib.load(SCALER_X_PATH)
            self._sy = joblib.load(SCALER_Y_PATH)
            stat = Path(MODEL_PATH).stat()
            self._version = f"{int(stat.st_mtime)}-{stat.st_size}"
            log.info("Modelo cargado (version=%s)", self._version)
            return True

    @property
    def ready(self) -> bool:
        return self._sess is not None

    @property
    def version(self) -> str | None:
        return self._version

    def predict_increment(self, vt: float, rt: float, rt_lag: float,
                          pt: float, et: float) -> float:
        if not self.ready:
            raise RuntimeError("model not loaded")
        x = np.array([[vt, rt, rt_lag, pt, et]], dtype=np.float32)
        x_sc = self._sx.transform(x).astype(np.float32)
        y_sc = self._sess.run(None, {self._input_name: x_sc})[0]
        return float(self._sy.inverse_transform(y_sc)[0, 0])


model = ModelHolder()
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(
        status="ok" if model.ready else "not_ready",
        model_version=model.version,
        model_path=MODEL_PATH,
    )


@app.post("/predict")
def predict():
    if not model.ready:
        return jsonify(error="model not loaded"), 503
    body = request.get_json(force=True)
    try:
        kwargs = {k: float(body[k]) for k in FEATURES}
    except KeyError as exc:
        return jsonify(error=f"missing feature: {exc.args[0]}"), 400
    delta_v = model.predict_increment(**kwargs)
    vt_next = float(np.clip(kwargs["vt"] + delta_v, 0.0, 100.0))
    return jsonify(delta_v=delta_v, vt_next=vt_next)


@app.post("/drydown")
def drydown():
    """Simula n días sin riego (sanity check físico)."""
    if not model.ready:
        return jsonify(error="model not loaded"), 503
    body = request.get_json(silent=True) or {}
    vt_init     = float(body.get("vt_init", 30.0))
    et_constant = float(body.get("et_constant", 6.0))
    n_days      = int(body.get("n_days", 30))
    vt = vt_init
    curve = [vt]
    for _ in range(n_days):
        delta = model.predict_increment(vt, 0.0, 0.0, 0.0, et_constant)
        vt = float(np.clip(vt + delta, 0.0, 100.0))
        curve.append(vt)
    return jsonify(curve=curve, n_days=n_days,
                   vt_init=vt_init, et_constant=et_constant)


@app.post("/reload")
def reload_model():
    ok = model.load()
    return jsonify(
        status="ok" if ok else "not_ready",
        model_version=model.version,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)
