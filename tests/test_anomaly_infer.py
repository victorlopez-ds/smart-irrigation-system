"""
tests/test_anomaly_infer.py — Tests del detector de anomalías hidráulicas.

Ejecutar: pytest tests/test_anomaly_infer.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestAnomalyResult:
    def test_anomaly_result_fields(self):
        from src.anomaly.infer import AnomalyResult
        r = AnomalyResult(
            is_anomaly=True,
            error=0.5,
            threshold=0.25,
            predicted_volume=100.0,
            actual_volume=200.0,
        )
        assert r.is_anomaly is True
        assert r.error == 0.5


class TestAnomalyDetectorONNX:
    """Tests con modelo ONNX mockeado (no requiere fichero real)."""

    def _make_detector(self):
        from src.anomaly.infer import AnomalyDetector
        from sklearn.preprocessing import StandardScaler
        import joblib
        import tempfile
        import os

        # Scaler temporal
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 1))
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        joblib.dump(scaler, tmp.name)
        scaler_path = tmp.name

        # Mock del session ONNX
        with patch("src.anomaly.infer.ort") as mock_ort, \
             patch("src.anomaly.infer._ONNX_AVAILABLE", True):
            mock_session = MagicMock()
            mock_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_session.run.return_value = [np.array([[0.1]])]
            mock_ort.InferenceSession.return_value = mock_session

            detector = AnomalyDetector.__new__(AnomalyDetector)
            detector.threshold = 0.25
            detector.window = 30
            detector.scaler = scaler
            detector._session = mock_session
            detector._backend = "onnx"

        os.unlink(scaler_path)
        return detector

    def test_no_anomaly_below_threshold(self):
        detector = self._make_detector()
        # error = |0.1 - 0.05| = 0.05 < 0.25 → no anomalía
        result = detector.detect(
            status_window=np.zeros((30, 1), dtype=np.float32),
            actual_volume_scaled=0.05,
        )
        assert result.is_anomaly is False

    def test_anomaly_above_threshold(self):
        detector = self._make_detector()
        # error = |0.1 - 0.8| = 0.7 > 0.25 → anomalía
        result = detector.detect(
            status_window=np.zeros((30, 1), dtype=np.float32),
            actual_volume_scaled=0.8,
        )
        assert result.is_anomaly is True

    def test_error_value(self):
        detector = self._make_detector()
        result = detector.detect(
            status_window=np.zeros((30, 1), dtype=np.float32),
            actual_volume_scaled=0.5,
        )
        assert abs(result.error - 0.4) < 1e-5


class TestAnomalyDetectorBatch:
    def test_batch_length(self):
        from src.anomaly.infer import AnomalyDetector
        detector = MagicMock(spec=AnomalyDetector)
        detector.window = 30
        detector.scaler = MagicMock()
        detector.scaler.transform = lambda x: x

        n = 50
        status = np.zeros(n, dtype=np.float32)
        volume = np.random.randn(n).astype(np.float32)

        # Llamamos al método real con el objeto mockeado
        detector.detect = lambda w, v: MagicMock(is_anomaly=False)
        results = AnomalyDetector.detect_batch(detector, status, volume)
        assert len(results) == n - 30
