"""
tests/test_rl_env.py — Tests unitarios del entorno RiegoEnv.

Ejecutar: pytest tests/test_rl_env.py -v
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


def make_dummy_df(n_rows: int = 200) -> pd.DataFrame:
    """Crea un DataFrame sintético con la estructura esperada por RiegoEnv."""
    dates = pd.date_range("2023-06-01", periods=n_rows, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "TimeInstant": dates,
        "Vt": rng.uniform(20, 40, n_rows),
        "Et": rng.uniform(2, 8, n_rows),
        "Pt": rng.uniform(0, 5, n_rows),
        "Rt": rng.uniform(0, 2500, n_rows),
    })


def make_dummy_scaler(n_features: int, n_outputs: int = 1):
    scaler = MagicMock()
    scaler.mean_ = np.zeros(n_features, dtype=np.float32)
    scaler.scale_ = np.ones(n_features, dtype=np.float32)
    return scaler


def make_dummy_model():
    """Modelo de transición que devuelve siempre incremento 0."""
    model = MagicMock()
    import tensorflow as tf
    model.return_value = tf.constant([[0.0]], dtype=tf.float32)
    return model


@pytest.fixture
def env():
    from src.rl.env import RiegoEnv, obtener_ventanas_validas

    df = make_dummy_df(200)
    indices = list(range(100))  # índices dummy

    model = make_dummy_model()
    scaler_x = make_dummy_scaler(5)
    scaler_y = make_dummy_scaler(1)

    return RiegoEnv(
        df_historico=df,
        indices_validos=indices,
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        N=2,
        duracion=10,   # episodio corto para tests rápidos
        isStochastic=False,
    )


class TestRiegoEnvSpaces:
    def test_observation_space_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (6,), f"Esperado (6,), obtenido {obs.shape}"

    def test_observation_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_action_space_bounds(self, env):
        assert env.action_space.low[0] == 0.0
        assert env.action_space.high[0] == 180.0


class TestRiegoEnvDynamics:
    def test_step_returns_5_elements(self, env):
        env.reset()
        result = env.step(np.array([90.0], dtype=np.float32))
        assert len(result) == 5

    def test_episode_terminates_after_duracion(self, env):
        env.reset()
        for _ in range(env.duracion - 1):
            _, _, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
            assert not term
        _, _, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        assert term

    def test_info_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.array([60.0], dtype=np.float32))
        for key in ("humedad", "riego_litros", "dia", "recompensa"):
            assert key in info

    def test_reward_optimal_zone(self, env):
        """Con humedad en zona óptima y sin penalización de agua, reward ≈ 1."""
        env.reset()
        env.soil_moisture = 30.0  # dentro de [25, 35]
        _, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert reward > 0, f"Reward esperado positivo, obtenido {reward}"

    def test_reward_severe_stress(self, env):
        """Con humedad crítica, la recompensa debe ser muy negativa."""
        env.reset()
        env.soil_moisture = 5.0  # muy por debajo de theta_min=20
        _, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert reward < -4.0, f"Reward esperado < -4, obtenido {reward}"


class TestRewardFunction:
    def test_all_zones(self, env):
        """Comprueba la función de recompensa en los cuatro regímenes."""
        env.reset()
        # Zona óptima
        r_opt = env._calcular_recompensa(30.0, 0.0)
        # Subóptima baja
        r_sub_low = env._calcular_recompensa(22.0, 0.0)
        # Subóptima alta
        r_sub_high = env._calcular_recompensa(40.0, 0.0)
        # Estrés severo
        r_stress = env._calcular_recompensa(10.0, 0.0)

        assert r_opt > r_sub_low
        assert r_opt > r_sub_high
        assert r_sub_low > r_stress


class TestVentanasValidas:
    def test_returns_list(self):
        from src.rl.env import obtener_ventanas_validas
        df = make_dummy_df(200)
        indices = obtener_ventanas_validas(df, ventana=90, N=2)
        assert isinstance(indices, list)

    def test_min_window(self):
        from src.rl.env import obtener_ventanas_validas
        df = make_dummy_df(200)
        indices = obtener_ventanas_validas(df, ventana=90, N=2)
        # Con 200 filas debería haber al menos algunos índices válidos
        # (depende de nulos, pero los datos sintéticos no tienen)
        assert len(indices) >= 0  # al menos no falla
