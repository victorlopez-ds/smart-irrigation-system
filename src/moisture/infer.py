"""
src/moisture/infer.py — Inferencia del predictor MLP de incrementos de humedad.

Este módulo es utilizado internamente por RiegoEnv como función de
transición T(s,a)→s'. No se despliega directamente en el edge.

Interfaz con RiegoEnv:
    El entorno carga el modelo Keras directamente (model = keras.load_model(...))
    y lo llama como callable: model(x_scaled, training=False).
    Este módulo expone también una clase MoisturePredictor de mayor nivel
    para uso fuera del entorno (p.ej. simulaciones independientes).
"""

from __future__ import annotations

import joblib
import numpy as np
from tensorflow import keras


class MoisturePredictor:
    """
    Predictor de incremento diario de humedad del suelo (ΔV_t) basado en MLP.

    Parameters
    ----------
    model_path    : ruta al fichero .keras del MLP
    scaler_x_path : ruta al StandardScaler ajustado sobre features X
    scaler_y_path : ruta al StandardScaler ajustado sobre target ΔV
    """

    FEATURES = ["Vt", "Rt", "Rt_lag", "Pt", "Et"]

    def __init__(
        self,
        model_path: str,
        scaler_x_path: str,
        scaler_y_path: str,
    ):
        self.model = keras.models.load_model(model_path)
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)

    def predict_increment(
        self,
        vt: float,
        rt: float,
        rt_lag: float,
        pt: float,
        et: float,
    ) -> float:
        """
        Predice el incremento de humedad ΔV para el día siguiente.

        Parameters
        ----------
        vt     : humedad actual del suelo (%)
        rt     : agua de riego hoy (litros)
        rt_lag : agua de riego ayer (litros)
        pt     : precipitación hoy (mm)
        et     : evapotranspiración hoy (mm)

        Returns
        -------
        delta_v : float — incremento de humedad (positivo = aumento)
        """
        x = np.array([[vt, rt, rt_lag, pt, et]], dtype=np.float32)
        x_scaled = self.scaler_x.transform(x)
        delta_scaled = self.model.predict(x_scaled, verbose=0)
        delta = float(self.scaler_y.inverse_transform(delta_scaled)[0, 0])
        return delta

    def predict_next_moisture(
        self,
        vt: float,
        rt: float,
        rt_lag: float,
        pt: float,
        et: float,
    ) -> float:
        """
        Predice la humedad absoluta del día siguiente V_{t+1}.

        Returns
        -------
        vt_next : float — humedad del suelo mañana (%)
        """
        delta = self.predict_increment(vt, rt, rt_lag, pt, et)
        return float(np.clip(vt + delta, 0.0, 100.0))

    def simulate_drydown(
        self,
        vt_init: float = 30.0,
        et_constant: float = 6.0,
        n_days: int = 30,
    ) -> list[float]:
        """
        Simula la curva de descenso de humedad durante n_days sin riego.
        Comprobación de sentido físico: la humedad debe disminuir monótonamente.

        Returns lista de valores de humedad día a día.
        """
        vt = vt_init
        curve = [vt]
        for _ in range(n_days):
            vt = self.predict_next_moisture(vt, 0.0, 0.0, 0.0, et_constant)
            curve.append(vt)
        return curve
