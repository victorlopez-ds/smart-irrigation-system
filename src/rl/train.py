"""
src/rl/train.py — Entrenamiento del agente TQC para control de riego.

Uso:
    python -m src.rl.train [--lambda_agua 1.5] [--N 2] [--timesteps 300000]

El modelo entrenado se registra automáticamente en MLflow si
MLFLOW_TRACKING_URI está definido en el entorno.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import OrderedDict
from copy import deepcopy

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import tf_keras as keras
from gymnasium import spaces
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    ProgressBarCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.env import RiegoEnv, RewardLoggerCallback, obtener_ventanas_validas

# ──────────────────────────────────────────────────────────────────────────────
# Configuración por defecto
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    lambda_agua=1.5,
    N=2,
    total_timesteps=300_000,
    seed=42,
    n_eval_episodes=20,
    log_dir="models/logs_rl/",
    data_path="data/processed/dataset_diario.csv",
    model_nn_path="models/modelo_nn_humedad.keras",
    scaler_x_path="models/scaler_x.pkl",
    scaler_y_path="models/scaler_y.pkl",
)

ALGO_KWARGS = dict(
    learning_rate=1e-4,
    ent_coef=0.1,
    batch_size=256,
    learning_starts=500,
    gamma=0.99,
    policy_kwargs=dict(net_arch=[256, 256]),
)

ENV_DEFAULTS = dict(
    tau_max=180.0,
    w_max=2500.0,
    theta_min=20.0,
    theta_opt_low=25.0,
    theta_opt_high=35.0,
    theta_max=45.0,
    q=12.57,
    duracion=90,
)


# ──────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ──────────────────────────────────────────────────────────────────────────────

def train(
    lambda_agua: float = DEFAULTS["lambda_agua"],
    N: int = DEFAULTS["N"],
    total_timesteps: int = DEFAULTS["total_timesteps"],
    seed: int = DEFAULTS["seed"],
    n_eval_episodes: int = DEFAULTS["n_eval_episodes"],
    log_dir: str = DEFAULTS["log_dir"],
    data_path: str = DEFAULTS["data_path"],
    model_nn_path: str = DEFAULTS["model_nn_path"],
    scaler_x_path: str = DEFAULTS["scaler_x_path"],
    scaler_y_path: str = DEFAULTS["scaler_y_path"],
) -> dict:
    """
    Entrena un agente TQC y devuelve las métricas de evaluación.

    Returns:
        dict con claves: reward_mean, pct_optimo_mean, agua_total_mean,
                         timesteps_used, log_dir, vecnorm_path
    """
    os.makedirs(log_dir, exist_ok=True)

    # ── Carga de datos y modelos de transición ────────────────────────────────
    df = pd.read_csv(data_path, parse_dates=["TimeInstant"])
    model_nn = keras.models.load_model(model_nn_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    indices_validos = obtener_ventanas_validas(df, ventana=ENV_DEFAULTS["duracion"], N=N)
    print(f"Temporadas válidas para N={N}: {len(indices_validos)}")

    env_kwargs = dict(
        df_historico=df,
        indices_validos=indices_validos,
        model=model_nn,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        N=N,
        lambda_agua=lambda_agua,
        **ENV_DEFAULTS,
    )

    # ── Entornos vectorizados ─────────────────────────────────────────────────
    train_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(RiegoEnv(**env_kwargs))]),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99,
    )
    eval_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(RiegoEnv(**env_kwargs))]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99,
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = TQC(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
        tensorboard_log=os.path.join(log_dir, "tb"),
        **ALGO_KWARGS,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=10, verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=5,
        best_model_save_path=log_dir,
        log_path=log_dir,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=1,
    )
    reward_logger = RewardLoggerCallback()
    callbacks = CallbackList([ProgressBarCallback(), reward_logger, eval_cb])

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"TQC_lambda{lambda_agua}_N{N}"):
        mlflow.log_params({"lambda_agua": lambda_agua, "N": N,
                           "total_timesteps": total_timesteps, "seed": seed})
        mlflow.log_params(ALGO_KWARGS)

        model.learn(total_timesteps=total_timesteps, callback=callbacks)

        # Sincronizar estadísticas VecNormalize al entorno de evaluación
        eval_env.obs_rms = deepcopy(train_env.obs_rms)
        eval_env.ret_rms = deepcopy(train_env.ret_rms)
        eval_env.training = False

        # Guardar estadísticas VecNormalize
        vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")
        train_env.save(vecnorm_path)

        # ── Evaluación final ──────────────────────────────────────────────────
        from src.rl.infer import evaluate_agent
        metrics = evaluate_agent(
            model, eval_env,
            n_episodes=n_eval_episodes,
            theta_opt_low=ENV_DEFAULTS["theta_opt_low"],
            theta_opt_high=ENV_DEFAULTS["theta_opt_high"],
            theta_min=ENV_DEFAULTS["theta_min"],
            theta_max=ENV_DEFAULTS["theta_max"],
        )
        metrics["timesteps_used"] = model.num_timesteps
        metrics["log_dir"] = log_dir
        metrics["vecnorm_path"] = vecnorm_path

        mlflow.log_metrics({
            "reward_mean": metrics["reward_mean"],
            "pct_optimo_mean": metrics["pct_optimo_mean"],
            "agua_total_mean": metrics["agua_total_mean"],
            "dias_estres_mean": metrics["dias_estres_mean"],
        })
        mlflow.log_artifact(os.path.join(log_dir, "best_model.zip"))
        mlflow.log_artifact(vecnorm_path)

    print(f"\n✅ Entrenamiento completado. Reward media: {metrics['reward_mean']:.2f}")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento agente RL de riego")
    parser.add_argument("--lambda_agua", type=float, default=DEFAULTS["lambda_agua"])
    parser.add_argument("--N", type=int, default=DEFAULTS["N"])
    parser.add_argument("--timesteps", type=int, default=DEFAULTS["total_timesteps"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--log_dir", type=str, default=DEFAULTS["log_dir"])
    args = parser.parse_args()

    train(
        lambda_agua=args.lambda_agua,
        N=args.N,
        total_timesteps=args.timesteps,
        seed=args.seed,
        log_dir=args.log_dir,
    )
