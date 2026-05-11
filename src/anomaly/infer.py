"""
src/anomaly/infer.py — Inferencia del detector de anomalías hidráulicas.

Expone:
  - AnomalyDetector : clase que encapsula modelo ONNX + scaler + umbral
  - detect          : función de conveniencia para una sola ventana

El modelo se carga en formato ONNX para ser ejecutado en la Raspberry Pi
con onnxruntime sin necesidad de TensorFlow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    from tensorflow import keras as _keras
    _KERAS_AVAILABLE = True
except ImportError:
    _KERAS_AVAILABLE = False


@dataclass
class AnomalyResult:
    is_anomaly: bool
    error: float
    threshold: float
    predicted_volume: float   # litros (escala original)
    actual_volume: float      # litros (escala original)


class AnomalyDetector:
    """
    Detector de anomalías hidráulicas basado en error de reconstrucción.

    Carga el modelo en formato ONNX (producción) o Keras (.keras) como
    fallback para desarrollo.

    Parameters
    ----------
    model_path  : ruta al fichero .onnx o .keras
    scaler_path : ruta al StandardScaler ajustado sobre vol_diff
    threshold   : umbral de error normalizado por encima del cual se declara anomalía
    window      : longitud de la ventana temporal de entrada (debe coincidir con entrenamiento)
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        threshold: float = 0.25,
        window: int = 30,
    ):
        self.threshold = threshold
        self.window = window
        self.scaler = joblib.load(scaler_path)

        if model_path.endswith(".onnx"):
            if not _ONNX_AVAILABLE:
                raise ImportError("onnxruntime no está instalado. Instálalo con: pip install onnxruntime")
            self._session = ort.InferenceSession(model_path)
            self._input_name = self._session.get_inputs()[0].name
            self._backend = "onnx"
        elif model_path.endswith(".keras"):
            if not _KERAS_AVAILABLE:
                raise ImportError("tensorflow/keras no está disponible")
            self._model = _keras.models.load_model(model_path)
            self._backend = "keras"
        else:
            raise ValueError(f"Formato de modelo no soportado: {model_path}")

    def predict_raw(self, status_window: np.ndarray) -> np.ndarray:
        """
        Predice el volumen (normalizado) a partir de una ventana de status.

        Parameters
        ----------
        status_window : array de forma (window, 1) con los valores de estado
                        de la electroválvula (minutos activos por intervalo de 15 min)

        Returns
        -------
        pred_scaled : float — volumen predicho en escala normalizada
        """
        x = status_window.reshape(1, self.window, 1).astype(np.float32)
        if self._backend == "onnx":
            result = self._session.run(None, {self._input_name: x})
            return float(result[0][0, 0])
        else:
            return float(self._model.predict(x, verbose=0)[0, 0])

    def detect(
        self,
        status_window: np.ndarray,
        actual_volume_scaled: float,
    ) -> AnomalyResult:
        """
        Detecta si el punto actual es una anomalía.

        Parameters
        ----------
        status_window          : array (window, 1) — historial de status de válvula
        actual_volume_scaled   : volumen real escalado (salida del StandardScaler)

        Returns
        -------
        AnomalyResult
        """
        pred_scaled = self.predict_raw(status_window)
        error = abs(pred_scaled - actual_volume_scaled)

        # Invertir escala para valores interpretables
        pred_original = float(self.scaler.inverse_transform([[pred_scaled]])[0, 0])
        actual_original = float(self.scaler.inverse_transform([[actual_volume_scaled]])[0, 0])

        return AnomalyResult(
            is_anomaly=error > self.threshold,
            error=error,
            threshold=self.threshold,
            predicted_volume=pred_original,
            actual_volume=actual_original,
        )

    def detect_batch(
        self,
        status_series: np.ndarray,
        volume_series: np.ndarray,
    ) -> list[AnomalyResult]:
        """
        Detecta anomalías sobre toda una serie temporal.

        Parameters
        ----------
        status_series : array (T,) — serie temporal de estados de válvula
        volume_series : array (T,) — serie temporal de volúmenes (escala original)

        Returns
        -------
        Lista de AnomalyResult (longitud T - window)
        """
        volume_scaled = self.scaler.transform(volume_series.reshape(-1, 1)).flatten()
        results = []
        for i in range(len(status_series) - self.window):
            window = status_series[i : i + self.window].reshape(self.window, 1)
            actual_v = volume_scaled[i + self.window]
            results.append(self.detect(window, actual_v))
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Función de conveniencia
# ──────────────────────────────────────────────────────────────────────────────

def detect(
    model_path: str,
    scaler_path: str,
    status_window: np.ndarray,
    actual_volume: float,
    threshold: float = 0.25,
    window: int = 30,
) -> AnomalyResult:
    """Función de conveniencia para detección en un solo punto."""
    detector = AnomalyDetector(model_path, scaler_path, threshold, window)
    actual_scaled = float(
        detector.scaler.transform([[actual_volume]])[0, 0]
    )
    return detector.detect(status_window, actual_scaled)
