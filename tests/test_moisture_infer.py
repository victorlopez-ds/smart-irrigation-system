"""
tests/test_moisture_infer.py — Tests del predictor MLP de humedad del suelo.

Ejecutar: pytest tests/test_moisture_infer.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestMoisturePredictorInterface:
    def _make_predictor(self):
        from src.moisture.infer import MoisturePredictor

        predictor = MoisturePredictor.__new__(MoisturePredictor)

        # Mock del modelo Keras
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.5]])  # ΔV escalado

        # Mock de los scalers
        mock_sx = MagicMock()
        mock_sx.transform.return_value = np.zeros((1, 5), dtype=np.float32)

        mock_sy = MagicMock()
        mock_sy.inverse_transform.return_value = np.array([[2.5]])  # ΔV real

        predictor.model = mock_model
        predictor.scaler_x = mock_sx
        predictor.scaler_y = mock_sy
        return predictor

    def test_predict_increment_returns_float(self):
        predictor = self._make_predictor()
        result = predictor.predict_increment(
            vt=30.0, rt=500.0, rt_lag=300.0, pt=0.0, et=4.0
        )
        assert isinstance(result, float)
        assert result == pytest.approx(2.5)

    def test_predict_next_moisture_adds_increment(self):
        predictor = self._make_predictor()
        next_v = predictor.predict_next_moisture(
            vt=30.0, rt=500.0, rt_lag=300.0, pt=0.0, et=4.0
        )
        assert next_v == pytest.approx(32.5)

    def test_predict_clamps_to_100(self):
        predictor = self._make_predictor()
        predictor.scaler_y.inverse_transform.return_value = np.array([[80.0]])
        next_v = predictor.predict_next_moisture(
            vt=90.0, rt=500.0, rt_lag=0.0, pt=50.0, et=0.0
        )
        assert next_v == pytest.approx(100.0)

    def test_predict_clamps_to_0(self):
        predictor = self._make_predictor()
        predictor.scaler_y.inverse_transform.return_value = np.array([[-50.0]])
        next_v = predictor.predict_next_moisture(
            vt=10.0, rt=0.0, rt_lag=0.0, pt=0.0, et=8.0
        )
        assert next_v == pytest.approx(0.0)

    def test_scaler_x_applied_before_model(self):
        """Verifica que se aplica el scaler antes de llamar al modelo."""
        predictor = self._make_predictor()
        predictor.predict_increment(vt=25.0, rt=100.0, rt_lag=50.0, pt=2.0, et=3.5)
        # scaler_x.transform debe haber sido llamado con el input correcto
        call_input = predictor.scaler_x.transform.call_args[0][0]
        assert call_input[0, 0] == pytest.approx(25.0)  # Vt
        assert call_input[0, 1] == pytest.approx(100.0) # Rt
        assert call_input[0, 2] == pytest.approx(50.0)  # Rt_lag
        assert call_input[0, 3] == pytest.approx(2.0)   # Pt
        assert call_input[0, 4] == pytest.approx(3.5)   # Et


class TestSimulateDrydown:
    def test_returns_list_of_correct_length(self):
        from src.moisture.infer import MoisturePredictor
        predictor = MoisturePredictor.__new__(MoisturePredictor)
        predictor.scaler_x = MagicMock()
        predictor.scaler_x.transform.return_value = np.zeros((1, 5))
        predictor.scaler_y = MagicMock()
        # Simula descenso de 1% por día
        predictor.scaler_y.inverse_transform.return_value = np.array([[-1.0]])
        predictor.model = MagicMock()
        predictor.model.predict.return_value = np.array([[-0.5]])

        curve = predictor.simulate_drydown(vt_init=30.0, et_constant=6.0, n_days=10)
        assert len(curve) == 11  # día 0 + 10 días

    def test_monotone_decrease_without_rain(self):
        """Sin riego, la humedad debe disminuir (o quedar en 0)."""
        from src.moisture.infer import MoisturePredictor
        predictor = MoisturePredictor.__new__(MoisturePredictor)
        predictor.scaler_x = MagicMock()
        predictor.scaler_x.transform.return_value = np.zeros((1, 5))
        predictor.scaler_y = MagicMock()
        predictor.scaler_y.inverse_transform.return_value = np.array([[-2.0]])
        predictor.model = MagicMock()
        predictor.model.predict.return_value = np.array([[0.0]])

        curve = predictor.simulate_drydown(vt_init=30.0, n_days=5)
        for i in range(1, len(curve)):
            assert curve[i] <= curve[i - 1] + 1e-6, \
                f"La humedad no decrece en el día {i}: {curve[i-1]} → {curve[i]}"
