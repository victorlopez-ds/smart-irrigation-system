"""
RiegoEnv — Entorno Gymnasium para optimización de riego mediante RL.

Paso de tiempo : 1 día
Episodio       : una temporada de `duracion` días (por defecto 90)
Observación    : [θ_t, w_{t-1}, E_{t:t+N}, P_{t:t+N}]
Acción         : τ ∈ [0, tau_max] — minutos de activación de la electroválvula
Recompensa     : lógica fuzzy de humedad − λ·(w_t / w_max)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

from .reward import RewardConfig, compute_reward


# ──────────────────────────────────────────────────────────────────────────────
# Entorno principal
# ──────────────────────────────────────────────────────────────────────────────

class RiegoEnv(gym.Env):
    """Entorno de RL para optimización de riego."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        df_historico: pd.DataFrame,
        indices_validos: list[int],
        model,                       # modelo de transición (Keras callable)
        scaler_x,                    # StandardScaler ajustado sobre features X
        scaler_y,                    # StandardScaler ajustado sobre target ΔV
        tau_max: float = 180.0,
        N: int = 2,
        w_max: float = 2500.0,
        theta_min: float = 20.0,
        theta_opt_low: float = 25.0,
        theta_opt_high: float = 35.0,
        theta_max: float = 45.0,
        lambda_agua: float = 0.5,
        q: float = 12.57,
        isStochastic: bool = True,
        duracion: int = 90,
        sd_error: float = 0.67,
    ):
        super().__init__()

        self.df_historico = df_historico
        self.indices_validos = indices_validos
        self.isStochastic = isStochastic
        self.duracion = duracion
        self.N = N

        obs_shape = (2 + 2 * N,)

        self.action_space = spaces.Box(
            low=0, high=tau_max, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.zeros(obs_shape, dtype=np.float32),
            high=np.array([100, w_max] + N * [20] + N * [100], dtype=np.float32),
            dtype=np.float32,
        )

        # Modelo de transición y escaladores
        self.model = model
        self.mean_x = scaler_x.mean_
        self.scale_x = scaler_x.scale_
        self.mean_y = float(scaler_y.mean_[0])
        self.scale_y = float(scaler_y.scale_[0])
        self.sd_error = sd_error

        # Umbrales agronómicos
        self.theta_min = theta_min
        self.theta_opt_low = theta_opt_low
        self.theta_opt_high = theta_opt_high
        self.theta_max = theta_max

        # Parámetros de riego
        self.tau_max = tau_max
        self.lambda_agua = lambda_agua
        self.q = q
        self.w_max = w_max

        # Desviación estándar para estocasticidad de Et
        self.std_Et = float(self.df_historico["Et"].std()) * 0.2

        # Estado interno (inicializado en reset)
        self.datos_episodio: pd.DataFrame | None = None
        self.dia_actual: int = 0
        self.soil_moisture: float = 0.0
        self.last_irrigation: float = 0.0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx_inicio = self.np_random.choice(self.indices_validos)
        len_data = self.duracion + self.N
        self.datos_episodio = (
            self.df_historico
            .iloc[idx_inicio : idx_inicio + len_data]
            .reset_index(drop=True)
            .copy()
        )

        if self.isStochastic:
            noise_Et = self.np_random.normal(0, self.std_Et, len_data)
            self.datos_episodio["Et"] = np.maximum(
                self.datos_episodio["Et"] + noise_Et, 0
            )
            noise_Pt = self.np_random.uniform(0.8, 1.2, len_data)
            self.datos_episodio["Pt"] = noise_Pt * self.datos_episodio["Pt"]

        self.dia_actual = 0
        self.soil_moisture = float(self.datos_episodio.loc[0, "Vt"])
        self.last_irrigation = 0.0

        return self._get_obs(), {}

    def step(self, accion):
        tau_val = float(accion[0])
        irrigation = self._calcular_litros(tau_val)

        fila_actual = self.datos_episodio.iloc[self.dia_actual]
        x_input = np.array(
            [[self.soil_moisture, irrigation, self.last_irrigation,
              fila_actual["Pt"], fila_actual["Et"]]],
            dtype=np.float32,
        )

        Vt_next = self._simulate_transition(x_input)
        reward = self._calcular_recompensa(Vt_next, irrigation)

        self.soil_moisture = Vt_next
        self.dia_actual += 1
        self.last_irrigation = irrigation

        terminated = self.dia_actual >= self.duracion
        info = {
            "humedad": float(self.soil_moisture),
            "riego_litros": float(irrigation),
            "dia": self.dia_actual,
            "recompensa": float(reward),
        }
        return self._get_obs(), reward, terminated, False, info

    # ── Métodos internos ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        proximos = self.datos_episodio.iloc[
            self.dia_actual : self.dia_actual + self.N
        ]
        ets = proximos["Et"].values.astype(np.float32)
        pts = proximos["Pt"].values.astype(np.float32)
        return np.concatenate(
            [[self.soil_moisture, self.last_irrigation], ets, pts]
        ).astype(np.float32)

    def _simulate_transition(self, x_input: np.ndarray) -> float:
        x_scaled = (x_input - self.mean_x) / self.scale_x
        vt_inc_scaled = self.model(x_scaled, training=False)
        vt_inc = float(vt_inc_scaled.numpy()[0, 0]) * self.scale_y + self.mean_y
        Vt_next = self.soil_moisture + vt_inc
        if self.isStochastic:
            Vt_next += self.np_random.normal(0.0, self.sd_error)
        return float(np.clip(Vt_next, 0.0, 100.0))

    def _calcular_litros(self, tau: float) -> float:
        return float(self.q * tau)

    def _calcular_recompensa(self, humedad: float, riego: float) -> float:
        # Delegamos en la función pura compartida con build_rl_transitions.
        # `riego` aquí está en litros (q*tau); reconvertimos a tau_min para
        # mantener la misma semántica de la función pura.
        tau_min = riego / self.q if self.q > 0 else 0.0
        cfg = RewardConfig(
            theta_min=self.theta_min,
            theta_opt_low=self.theta_opt_low,
            theta_opt_high=self.theta_opt_high,
            theta_max=self.theta_max,
            lambda_agua=self.lambda_agua,
            tau_max=self.tau_max,
            q=self.q,
        )
        return compute_reward(next_theta=humedad, tau_minutes=tau_min, cfg=cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def obtener_ventanas_validas(
    df: pd.DataFrame,
    ventana: int = 90,
    N: int = 2,
) -> list[int]:
    """Devuelve los índices de inicio de temporadas válidas (sin nulos, consecutivas)."""
    indices = []
    for i in range(len(df) - (ventana + N) + 1):
        bloque = df.iloc[i : i + ventana + N]
        sin_nulos = not bloque[["Et", "Pt", "Vt"]].isnull().any().any()
        dias_reales = (
            bloque.iloc[-1]["TimeInstant"] - bloque.iloc[0]["TimeInstant"]
        ).days
        if sin_nulos and dias_reales == (ventana + N - 1):
            indices.append(i)
    return indices


class RewardLoggerCallback(BaseCallback):
    """Registra la recompensa media por episodio durante el entrenamiento."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ep_rew_means: list[float] = []
        self.timesteps: list[int] = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            self.ep_rew_means.append(mean_r)
            self.timesteps.append(self.n_calls)
        return True
