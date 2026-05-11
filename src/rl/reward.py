"""
src/rl/reward.py — Función de recompensa pura del problema de riego.

Se extrae de RiegoEnv para poder reutilizarla:
  - dentro del simulador (RiegoEnv.step)
  - sobre transiciones reales (build_rl_transitions DAG): el reward se
    calcula a posteriori, cuando se conoce θ_{t+1} ≡ humedad del día siguiente.

Mantener una sola fuente de verdad garantiza que el actor recibe la misma
señal de optimización en sim y en real → fine-tune estable.

Coeficientes y umbrales son los mismos del entrenamiento offline original:
RewardConfig.DEFAULT.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Parámetros agronómicos y pesos de la recompensa."""
    theta_min:        float = 20.0    # umbral de estrés severo
    theta_opt_low:    float = 25.0    # frontera inferior de la zona óptima
    theta_opt_high:   float = 35.0    # frontera superior de la zona óptima
    theta_max:        float = 45.0    # umbral de encharcamiento
    w_opt:            float = 1.0     # peso por estar en zona óptima
    w_estres:         float = 5.0     # penalización por estrés hídrico
    w_encharc:        float = 5.0     # penalización por encharcamiento
    lambda_agua:      float = 0.5     # penalización por consumo de agua
    tau_max:          float = 180.0   # τ máxima (min) — usado para normalizar
    q:                float = 12.57   # litros/min de la electroválvula


DEFAULT = RewardConfig()


def compute_reward(
    next_theta:    float,
    tau_minutes:   float,
    cfg:           RewardConfig = DEFAULT,
) -> float:
    """
    Recompensa de una transición.

    Parameters
    ----------
    next_theta   : humedad del suelo (%) tras aplicar el riego (s_{t+1})
    tau_minutes  : minutos de activación de la electroválvula (a_t)
    cfg          : parámetros agronómicos. Por defecto RewardConfig.DEFAULT,
                   mismos valores con los que se entrenó el modelo offline.

    Returns
    -------
    reward : float
    """
    h = float(next_theta)

    # Pertenencia a zona óptima (trapezoidal)
    if cfg.theta_opt_low <= h <= cfg.theta_opt_high:
        p_opt = 1.0
    elif cfg.theta_min < h < cfg.theta_opt_low:
        p_opt = (h - cfg.theta_min) / (cfg.theta_opt_low - cfg.theta_min)
    elif cfg.theta_opt_high < h < cfg.theta_max:
        p_opt = (cfg.theta_max - h) / (cfg.theta_max - cfg.theta_opt_high)
    else:
        p_opt = 0.0

    # Estrés hídrico
    if h <= cfg.theta_min:
        p_estres = 1.0
    elif h < cfg.theta_opt_low:
        p_estres = (cfg.theta_opt_low - h) / (cfg.theta_opt_low - cfg.theta_min)
    else:
        p_estres = 0.0

    # Encharcamiento
    if h >= cfg.theta_max:
        p_enc = 1.0
    elif h > cfg.theta_opt_high:
        p_enc = (h - cfg.theta_opt_high) / (cfg.theta_max - cfg.theta_opt_high)
    else:
        p_enc = 0.0

    reward = cfg.w_opt * p_opt - cfg.w_estres * p_estres - cfg.w_encharc * p_enc

    riego_litros = cfg.q * float(tau_minutes)
    riego_norm   = riego_litros / (cfg.q * cfg.tau_max)
    reward      -= cfg.lambda_agua * riego_norm

    return float(reward)
