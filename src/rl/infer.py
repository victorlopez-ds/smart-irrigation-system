"""
src/rl/infer.py — Inferencia diaria del agente TQC de riego.

Expone:
  - RLInferenceEngine : clase que encapsula carga de modelo + predicción
  - evaluate_agent    : evaluación del agente sobre un VecEnv (métricas)
  - predict_action    : función de conveniencia para un único paso
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from gymnasium import spaces
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.env import RiegoEnv


# ──────────────────────────────────────────────────────────────────────────────
# Motor de inferencia
# ──────────────────────────────────────────────────────────────────────────────

class RLInferenceEngine:
    """
    Carga el modelo TQC y las estadísticas VecNormalize para inferencia
    sin necesidad de GPU ni de reentrenar.

    Parameters
    ----------
    model_path    : ruta al fichero best_model.zip
    vecnorm_path  : ruta al fichero vecnormalize.pkl
    N             : horizonte de previsión (debe coincidir con el entrenamiento)
    w_max         : volumen máximo de riego (litros)
    tau_max       : tiempo máximo de activación (minutos)
    """

    def __init__(
        self,
        model_path: str,
        vecnorm_path: str,
        N: int = 2,
        w_max: float = 2500.0,
        tau_max: float = 180.0,
    ):
        self.N = N
        self.tau_max = tau_max
        self.w_max = w_max

        # Espacios con las dimensiones correctas para cargar el modelo
        obs_space = spaces.Box(
            low=np.zeros(2 + 2 * N, dtype=np.float32),
            high=np.array([100, w_max] + N * [20] + N * [100], dtype=np.float32),
        )
        act_space = spaces.Box(low=0.0, high=tau_max, shape=(1,), dtype=np.float32)

        # Entorno dummy solo para que SB3 pueda cargar el modelo
        dummy_env = DummyVecEnv([lambda: _DummyEnv(obs_space, act_space)])
        self._vec_env = VecNormalize.load(vecnorm_path, dummy_env)
        self._vec_env.training = False
        self._vec_env.norm_reward = False

        self.model = TQC.load(
            model_path,
            env=self._vec_env,
            custom_objects={
                "observation_space": obs_space,
                "action_space": act_space,
            },
        )

    def predict(
        self,
        theta_t: float,
        w_prev: float,
        et_forecast: list[float],
        pt_forecast: list[float],
        deterministic: bool = True,
    ) -> float:
        """
        Devuelve τ (minutos de riego) dado el estado actual del suelo y
        la previsión meteorológica.

        Parameters
        ----------
        theta_t       : humedad del suelo actual (%)
        w_prev        : volumen de riego del día anterior (litros)
        et_forecast   : lista de N valores de ET para los próximos N días
        pt_forecast   : lista de N valores de precipitación para los próximos N días

        Returns
        -------
        tau : float — minutos de activación de la electroválvula ∈ [0, tau_max]
        """
        obs = np.array(
            [theta_t, w_prev] + list(et_forecast) + list(pt_forecast),
            dtype=np.float32,
        ).reshape(1, -1)

        obs_norm = self._vec_env.normalize_obs(obs)
        action, _ = self.model.predict(obs_norm, deterministic=deterministic)
        tau = float(np.clip(action[0][0], 0.0, self.tau_max))
        return tau


# ──────────────────────────────────────────────────────────────────────────────
# Evaluación sobre entorno vectorizado
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_agent(
    model,
    eval_vec_env: VecNormalize,
    n_episodes: int = 20,
    theta_opt_low: float = 25.0,
    theta_opt_high: float = 35.0,
    theta_min: float = 20.0,
    theta_max: float = 45.0,
) -> dict:
    """
    Evalúa un agente SB3 sobre un entorno vectorizado con VecNormalize.

    Returns
    -------
    dict con métricas agregadas (mean/std) y datos crudos por episodio.
    """
    ep_rewards, ep_pct_optimo = [], []
    ep_agua_total, ep_dias_estres, ep_std_humedad = [], [], []

    for _ in range(n_episodes):
        obs = eval_vec_env.reset()
        humedades, riegos, rewards = [], [], []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, info = eval_vec_env.step(action)
            humedades.append(info[0]["humedad"])
            riegos.append(info[0]["riego_litros"])
            rewards.append(info[0]["recompensa"])
            done = done_arr[0]

        dias_opt = sum(1 for h in humedades if theta_opt_low <= h <= theta_opt_high)
        dias_str = sum(1 for h in humedades if h < theta_min or h > theta_max)

        ep_rewards.append(sum(rewards))
        ep_pct_optimo.append(dias_opt / len(humedades) * 100)
        ep_agua_total.append(sum(riegos))
        ep_dias_estres.append(dias_str)
        ep_std_humedad.append(float(np.std(humedades)))

    return {
        "reward_mean":      np.mean(ep_rewards),
        "reward_std":       np.std(ep_rewards),
        "pct_optimo_mean":  np.mean(ep_pct_optimo),
        "pct_optimo_std":   np.std(ep_pct_optimo),
        "agua_total_mean":  np.mean(ep_agua_total),
        "agua_total_std":   np.std(ep_agua_total),
        "dias_estres_mean": np.mean(ep_dias_estres),
        "dias_estres_std":  np.std(ep_dias_estres),
        "std_humedad_mean": np.mean(ep_std_humedad),
        "raw_rewards":      ep_rewards,
        "raw_pct_optimo":   ep_pct_optimo,
        "raw_agua_total":   ep_agua_total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entorno dummy (solo para cargar el modelo)
# ──────────────────────────────────────────────────────────────────────────────

class _DummyEnv:
    """Entorno mínimo para inicializar VecNormalize sin datos reales."""

    def __init__(self, obs_space: spaces.Box, act_space: spaces.Box):
        self.observation_space = obs_space
        self.action_space = act_space
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}
        self.spec = None

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}
