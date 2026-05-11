"""
src/rl/finetune.py — Fine-tune online del agente TQC con replay buffer híbrido.

Carga el agente champion (rl_actor.zip), construye un replay buffer que
combina transiciones REALES (recolectadas por rl-api en producción) con
transiciones SIMULADAS (generadas durante el rollout) en una proporción
controlable, ejecuta `.learn()` continuando desde los pesos actuales, y
evalúa challenger vs champion sobre el simulador. Si Δreturn ≥ margen
configurado, exporta el nuevo .zip y deja un actor ONNX listo para
distribución vía model-distributor.

Uso típico (vía DockerOperator):
    python -m src.rl.finetune \
        --base-model       /app/models/best_model_rl.zip \
        --real-parquet     /tmp/real_transitions.parquet \
        --moisture-model   /app/models/moisture_mlp.keras \
        --moisture-sx      /app/models/moisture_scaler_x.pkl \
        --moisture-sy      /app/models/moisture_scaler_y.pkl \
        --dataset          /app/data/processed/dataset_diario.csv \
        --output-zip       /app/models/best_model_rl.zip \
        --output-onnx      /app/models/rl_actor.onnx \
        --total-steps      50000 \
        --real-ratio       0.5 \
        --eval-episodes    20 \
        --improvement-margin 0.01 \
        --warmup-episodes  50

Imprime al final:
    __FINETUNE_RESULT__={...JSON...}
para que el DAG de Airflow parsee y decida.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch as th
import tf_keras as keras
from sb3_contrib import TQC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.env import RiegoEnv, obtener_ventanas_validas


# ──────────────────────────────────────────────────────────────────────────────
# HybridReplayBuffer — dos buffers internos (real, sim) + sampler ponderado
# ──────────────────────────────────────────────────────────────────────────────

class HybridReplayBuffer(ReplayBuffer):
    """
    Drop-in replacement del ReplayBuffer de SB3 que mantiene dos sub-buffers:

      - `_buf_real` : transiciones reales pre-cargadas. NO se modifica en
                      rollout (no se sobrescribe ni con add() ni con LRU).
      - `_buf_sim`  : transiciones simuladas, añadidas durante .learn().

    `sample(batch_size)` devuelve un batch con `batch_size * real_ratio`
    transiciones reales y el resto simuladas. Si una de las particiones está
    vacía, cae en la otra para evitar fallar en el primer step.

    Limitaciones:
      - real_ratio fijo durante todo el .learn() de una llamada.
      - No soporta n_envs > 1 en el sub-buffer real (lo creamos siempre con 1).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: str = "cpu",
        n_envs: int = 1,
        real_ratio: float = 0.5,
        real_capacity: int = 100_000,
        **kwargs,
    ):
        # Buffer "público" (heredado): se usa como buffer SIM. No lo
        # rellenamos con add() porque sobreescribiríamos init.
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            **kwargs,
        )
        self._real_ratio = float(real_ratio)
        self._buf_real = ReplayBuffer(
            buffer_size=real_capacity,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=1,
        )

    # ── Pre-poblado con transiciones reales ──────────────────────────────────

    def load_real_transitions(
        self,
        obs:        np.ndarray,
        actions:    np.ndarray,
        rewards:    np.ndarray,
        next_obs:   np.ndarray,
        dones:      np.ndarray,
    ) -> int:
        """Inserta un lote de transiciones REALES (raw, sin normalizar)."""
        n = len(obs)
        for i in range(n):
            self._buf_real.add(
                obs[i],
                next_obs[i],
                actions[i],
                rewards[i],
                dones[i],
                infos=[{}],
            )
        return n

    @property
    def n_real(self) -> int:
        return int(self._buf_real.size())

    @property
    def real_ratio(self) -> float:
        return self._real_ratio

    @real_ratio.setter
    def real_ratio(self, value: float) -> None:
        self._real_ratio = float(np.clip(value, 0.0, 1.0))

    # ── Sample mezclado ───────────────────────────────────────────────────────

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        n_real_avail = self._buf_real.size()
        n_sim_avail  = super().size()

        if n_real_avail == 0:
            return super().sample(batch_size, env=env)
        if n_sim_avail == 0:
            return self._buf_real.sample(batch_size, env=env)

        n_real = int(round(batch_size * self._real_ratio))
        n_sim  = batch_size - n_real
        n_real = max(0, min(n_real, batch_size))
        n_sim  = batch_size - n_real

        parts = []
        if n_real > 0:
            parts.append(self._buf_real.sample(n_real, env=env))
        if n_sim > 0:
            parts.append(super().sample(n_sim, env=env))

        if len(parts) == 1:
            return parts[0]

        # Concatenar tensores de cada campo
        return ReplayBufferSamples(
            observations      = th.cat([p.observations      for p in parts], dim=0),
            actions           = th.cat([p.actions           for p in parts], dim=0),
            next_observations = th.cat([p.next_observations for p in parts], dim=0),
            dones             = th.cat([p.dones             for p in parts], dim=0),
            rewards           = th.cat([p.rewards           for p in parts], dim=0),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Construcción del entorno (idéntica a export.py — reutilizable)
# ──────────────────────────────────────────────────────────────────────────────

def build_envs(args) -> tuple[VecNormalize, VecNormalize]:
    """Devuelve (train_env, eval_env) con VecNormalize compartiendo stats."""
    print(f"Cargando moisture model: {args.moisture_model}", flush=True)
    model_nn = keras.models.load_model(args.moisture_model)
    scaler_x = joblib.load(args.moisture_sx)
    scaler_y = joblib.load(args.moisture_sy)

    print(f"Cargando dataset: {args.dataset}", flush=True)
    df = pd.read_csv(args.dataset, parse_dates=["TimeInstant"])
    indices_validos = obtener_ventanas_validas(df, ventana=90, N=args.N)
    if not indices_validos:
        raise RuntimeError("No hay temporadas válidas en el dataset")

    env_kwargs = dict(
        df_historico=df,
        indices_validos=indices_validos,
        model=model_nn,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        N=args.N,
    )
    train_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(RiegoEnv(**env_kwargs))]),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99,
    )
    eval_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(RiegoEnv(**env_kwargs))]),
        norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99,
    )
    train_env.training = True
    return train_env, eval_env


def warmup_vecnormalize(model: TQC, vec_env: VecNormalize, episodes: int) -> int:
    """Ejecuta `episodes` con el modelo para calentar obs_rms."""
    obs = vec_env.reset()
    steps = 0
    for _ in range(episodes):
        done = np.array([False])
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)
            steps += 1
    return steps


# ──────────────────────────────────────────────────────────────────────────────
# Carga de transiciones reales desde parquet
# ──────────────────────────────────────────────────────────────────────────────

def load_real_parquet(path: str, expected_obs_dim: int) -> dict:
    df = pd.read_parquet(path)
    if df.empty:
        return {"n": 0}

    obs       = np.stack(df["obs"].apply(np.asarray).values).astype(np.float32)
    next_obs  = np.stack(df["next_obs"].apply(np.asarray).values).astype(np.float32)
    actions   = df["action"].astype(np.float32).values.reshape(-1, 1)
    rewards   = df["reward"].astype(np.float32).values
    dones     = df["done"].astype(np.float32).values

    if obs.shape[1] != expected_obs_dim:
        raise ValueError(
            f"obs_dim del parquet ({obs.shape[1]}) ≠ esperado ({expected_obs_dim}). "
            "¿Se cambió la dimensión del estado entre versiones?"
        )

    return {
        "n":        len(df),
        "obs":      obs,
        "actions":  actions,
        "rewards":  rewards,
        "next_obs": next_obs,
        "dones":    dones,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model",       required=True, help="Champion .zip")
    ap.add_argument("--real-parquet",     required=True)
    ap.add_argument("--moisture-model",   required=True)
    ap.add_argument("--moisture-sx",      required=True)
    ap.add_argument("--moisture-sy",      required=True)
    ap.add_argument("--dataset",          required=True)
    ap.add_argument("--output-zip",       required=True)
    ap.add_argument("--output-onnx",      required=True)
    ap.add_argument("--total-steps",      type=int,   default=50_000)
    ap.add_argument("--real-ratio",       type=float, default=0.5)
    ap.add_argument("--eval-episodes",    type=int,   default=20)
    ap.add_argument("--improvement-margin", type=float, default=0.01)
    ap.add_argument("--warmup-episodes",  type=int,   default=50)
    ap.add_argument("--N",                type=int,   default=2)
    ap.add_argument("--seed",             type=int,   default=42)
    args = ap.parse_args()

    train_env, eval_env = build_envs(args)

    # ── Cargar champion (no se toca; sirve de baseline) ───────────────────────
    print(f"Cargando champion: {args.base_model}", flush=True)
    champion = TQC.load(args.base_model, device="cpu")

    # Warmup VecNormalize con el champion (replica export.py para que las
    # estadísticas obs_rms del fine-tune sean equiparables a las del baseline)
    print(f"Warmup VecNormalize: {args.warmup_episodes} episodios", flush=True)
    warmup_steps = warmup_vecnormalize(champion, train_env, args.warmup_episodes)
    eval_env.obs_rms = deepcopy(train_env.obs_rms)
    eval_env.training = False
    print(f"Warmup: {warmup_steps} steps. obs_mean={train_env.obs_rms.mean}", flush=True)

    # ── Construir challenger sobre los pesos del champion ────────────────────
    print(f"Cargando challenger sobre train_env", flush=True)
    challenger = TQC.load(args.base_model, env=train_env, device="cpu")

    # Reemplazar el replay buffer por el híbrido
    obs_dim = train_env.observation_space.shape[0]
    real = load_real_parquet(args.real_parquet, expected_obs_dim=obs_dim)
    print(f"Transiciones reales en parquet: {real['n']}", flush=True)

    hybrid = HybridReplayBuffer(
        buffer_size=challenger.buffer_size,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        device=challenger.device,
        n_envs=1,
        real_ratio=args.real_ratio,
        real_capacity=max(real["n"], 1024),
    )
    if real["n"] > 0:
        hybrid.load_real_transitions(
            real["obs"], real["actions"], real["rewards"],
            real["next_obs"], real["dones"],
        )
    challenger.replay_buffer = hybrid

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print(f"Fine-tune: {args.total_steps} timesteps "
          f"(real_ratio={args.real_ratio}, n_real={hybrid.n_real})", flush=True)
    challenger.learn(
        total_timesteps=args.total_steps,
        reset_num_timesteps=False,
        progress_bar=False,
    )

    # ── Eval champion vs challenger sobre el simulador ───────────────────────
    print(f"Evaluando champion vs challenger ({args.eval_episodes} eps)…", flush=True)
    champ_mean, _ = evaluate_policy(
        champion,   eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    chall_mean, _ = evaluate_policy(
        challenger, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    rel_gain = (chall_mean - champ_mean) / (abs(champ_mean) + 1e-8)
    promoted = rel_gain >= args.improvement_margin
    print(f"Champion mean reward   = {champ_mean:.4f}", flush=True)
    print(f"Challenger mean reward = {chall_mean:.4f}", flush=True)
    print(f"Mejora relativa        = {rel_gain:+.2%}  (margen {args.improvement_margin:+.2%})", flush=True)

    summary = {
        "champion_mean_reward":   float(champ_mean),
        "challenger_mean_reward": float(chall_mean),
        "rel_gain":               float(rel_gain),
        "improvement_margin":     float(args.improvement_margin),
        "promoted":               bool(promoted),
        "n_real_transitions":     int(real["n"]),
        "real_ratio":             float(args.real_ratio),
        "total_steps":            int(args.total_steps),
    }

    if not promoted:
        print("❌ Sin mejora suficiente → no se promociona.", flush=True)
        print(f"__FINETUNE_RESULT__={json.dumps(summary)}", flush=True)
        return 0

    # ── Guardar challenger y re-exportar ONNX ────────────────────────────────
    Path(args.output_zip).parent.mkdir(parents=True, exist_ok=True)
    challenger.save(args.output_zip)
    print(f"✅ Challenger guardado: {args.output_zip}", flush=True)

    # Reutilizamos export.py via subprocess (independiente, evita duplicar code)
    print("Lanzando export ONNX…", flush=True)
    cmd = [
        sys.executable, "-m", "src.rl.export",
        "--model-zip",       args.output_zip,
        "--moisture-model",  args.moisture_model,
        "--moisture-sx",     args.moisture_sx,
        "--moisture-sy",     args.moisture_sy,
        "--dataset",         args.dataset,
        "--output-onnx",     args.output_onnx,
        "--warmup-episodes", str(args.warmup_episodes),
        "--N",               str(args.N),
        "--seed",            str(args.seed),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.returncode != 0:
        print(res.stderr, flush=True)
        summary["export_error"] = res.stderr[-500:]
        print(f"__FINETUNE_RESULT__={json.dumps(summary)}", flush=True)
        return 1

    summary["output_zip"]  = args.output_zip
    summary["output_onnx"] = args.output_onnx
    print(f"__FINETUNE_RESULT__={json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
