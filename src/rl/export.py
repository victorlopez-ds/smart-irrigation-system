"""
src/rl/export.py — Convierte el actor TQC entrenado (sb3-contrib) a ONNX.

Empaqueta dentro del grafo:
  1. La normalización de la observación de VecNormalize (mean/var congelados)
  2. La forward determinista del actor TQC (latent_pi → mu → tanh)
  3. La des-escalación de la acción al rango original [low, high]

Como NO disponemos del `vecnormalize.pkl` original, se replica el truco de
los notebooks (`create_eval_env_fixed`): se monta un entorno de evaluación
fresco con VecNormalize en modo training=True y se ejecutan N episodios
de "warmup" para que las stats `obs_rms` se estabilicen. Después se
congelan y se hornean en el grafo.

Uso (típicamente vía Docker):
    python -m src.rl.export \
        --model-zip       /app/models/best_model_rl.zip \
        --moisture-model  /app/models/moisture_mlp.keras \
        --moisture-sx     /app/models/moisture_scaler_x.pkl \
        --moisture-sy     /app/models/moisture_scaler_y.pkl \
        --dataset         /app/data/processed/dataset_diario.csv \
        --output-onnx     /app/models/rl_actor.onnx \
        --warmup-episodes 50

Imprime al final un JSON con metadatos del export (obs_dim, action_low/high,
warmup_steps, equivalencia numérica con la policy original).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import tf_keras as keras
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.env import RiegoEnv, obtener_ventanas_validas


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper torch: normalización + actor + unscaling
# ──────────────────────────────────────────────────────────────────────────────

class TQCActorOnnxWrapper(nn.Module):
    """
    forward(obs_raw) → action en unidades reales [low, high]

    obs_raw : (B, obs_dim)  observación SIN normalizar (espacio original del env)
    return  : (B, act_dim)  acción des-escalada (minutos de riego en nuestro caso)
    """

    def __init__(
        self,
        actor: nn.Module,           # TQC.policy.actor
        obs_mean: np.ndarray,       # VecNormalize.obs_rms.mean
        obs_var: np.ndarray,        # VecNormalize.obs_rms.var
        clip_obs: float,            # VecNormalize.clip_obs (e.g. 10.0)
        action_low: np.ndarray,     # env.action_space.low
        action_high: np.ndarray,    # env.action_space.high
        eps: float = 1e-8,
    ):
        super().__init__()
        self.actor = actor
        self.register_buffer("obs_mean",
                             torch.as_tensor(obs_mean,  dtype=torch.float32))
        self.register_buffer("obs_var",
                             torch.as_tensor(obs_var,   dtype=torch.float32))
        self.register_buffer("action_low",
                             torch.as_tensor(action_low,  dtype=torch.float32))
        self.register_buffer("action_high",
                             torch.as_tensor(action_high, dtype=torch.float32))
        self.clip_obs = float(clip_obs)
        self.eps = float(eps)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 1) Normalización VecNormalize (obs_rms congeladas)
        obs_norm = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.eps)
        obs_norm = torch.clamp(obs_norm, -self.clip_obs, self.clip_obs)

        # 2) Forward determinista del actor TQC
        #    En sb3-contrib TQC la acción determinista es tanh(mu(latent_pi(obs)))
        latent = self.actor.latent_pi(obs_norm)
        mean_actions = self.actor.mu(latent)
        action_squashed = torch.tanh(mean_actions)   # ∈ [-1, 1]

        # 3) Des-escalado [-1, 1] → [low, high] (idéntico a Box.scale_action en SB3)
        action = self.action_low + 0.5 * (action_squashed + 1.0) * (
            self.action_high - self.action_low
        )
        return action


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-zip",       required=True)
    ap.add_argument("--moisture-model",  required=True)
    ap.add_argument("--moisture-sx",     required=True)
    ap.add_argument("--moisture-sy",     required=True)
    ap.add_argument("--dataset",         required=True)
    ap.add_argument("--output-onnx",     required=True)
    ap.add_argument("--warmup-episodes", type=int, default=50)
    ap.add_argument("--N",               type=int, default=2,
                    help="Horizonte de pronóstico usado al entrenar")
    ap.add_argument("--seed",            type=int, default=42)
    args = ap.parse_args()

    # ── Carga moisture model + scalers ────────────────────────────────────────
    print(f"Cargando moisture model: {args.moisture_model}", flush=True)
    model_nn = keras.models.load_model(args.moisture_model)
    scaler_x = joblib.load(args.moisture_sx)
    scaler_y = joblib.load(args.moisture_sy)

    # ── Dataset y ventanas válidas ────────────────────────────────────────────
    print(f"Cargando dataset: {args.dataset}", flush=True)
    df = pd.read_csv(args.dataset, parse_dates=["TimeInstant"])
    indices_validos = obtener_ventanas_validas(df, ventana=90, N=args.N)
    print(f"Temporadas válidas (N={args.N}): {len(indices_validos)}", flush=True)
    if not indices_validos:
        raise RuntimeError("No hay temporadas válidas en el dataset")

    # ── Entorno con VecNormalize fresco (modo training=True) ──────────────────
    env_kwargs = dict(
        df_historico=df,
        indices_validos=indices_validos,
        model=model_nn,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        N=args.N,
    )
    vec_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(RiegoEnv(**env_kwargs))]),
        norm_obs=True, norm_reward=False,
        clip_obs=10.0, gamma=0.99,
    )
    vec_env.training = True   # auto-ajusta obs_rms en cada step

    # ── Carga del agente TQC ──────────────────────────────────────────────────
    print(f"Cargando TQC: {args.model_zip}", flush=True)
    model = TQC.load(args.model_zip, env=vec_env, device="cpu")

    # ── Warmup: rellenar obs_rms con N episodios ──────────────────────────────
    print(f"Warmup: {args.warmup_episodes} episodios", flush=True)
    obs = vec_env.reset()
    total_steps = 0
    for ep in range(args.warmup_episodes):
        done = np.array([False])
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)
            total_steps += 1
    print(f"Warmup completado: {total_steps} steps", flush=True)

    # ── Congelar stats ────────────────────────────────────────────────────────
    obs_mean = vec_env.obs_rms.mean.astype(np.float32)
    obs_var  = vec_env.obs_rms.var.astype(np.float32)
    print(f"obs_mean = {obs_mean}", flush=True)
    print(f"obs_var  = {obs_var}",  flush=True)

    # ── Construir wrapper torch ───────────────────────────────────────────────
    actor = model.policy.actor.eval().cpu()

    inner_env = vec_env.envs[0].env  # Monitor → RiegoEnv
    action_low  = inner_env.action_space.low.astype(np.float32)
    action_high = inner_env.action_space.high.astype(np.float32)
    obs_dim     = inner_env.observation_space.shape[0]

    wrapper = TQCActorOnnxWrapper(
        actor=actor,
        obs_mean=obs_mean,
        obs_var=obs_var,
        clip_obs=10.0,
        action_low=action_low,
        action_high=action_high,
    ).eval()

    # ── Test de equivalencia numérica antes de exportar ───────────────────────
    rng = np.random.default_rng(args.seed)
    sample_raw = rng.uniform(
        low=inner_env.observation_space.low,
        high=inner_env.observation_space.high,
        size=(8, obs_dim),
    ).astype(np.float32)

    with torch.no_grad():
        wrapper_action = wrapper(torch.as_tensor(sample_raw)).cpu().numpy()

    # Camino "oficial": normaliza y predice con la policy
    sample_norm = vec_env.normalize_obs(sample_raw)
    sb3_action, _ = model.policy.predict(sample_norm, deterministic=True)
    max_diff = float(np.abs(wrapper_action - sb3_action).max())
    print(f"Diferencia máx wrapper ↔ TQC.predict: {max_diff:.6f}", flush=True)

    # ── Export a ONNX ─────────────────────────────────────────────────────────
    Path(args.output_onnx).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        wrapper, dummy, args.output_onnx,
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    print(f"✅ ONNX exportado: {args.output_onnx}", flush=True)

    # ── Verificación con onnxruntime ──────────────────────────────────────────
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output_onnx)
    onnx_action = sess.run(None, {"obs": sample_raw})[0]
    max_diff_onnx = float(np.abs(wrapper_action - onnx_action).max())
    print(f"Diferencia máx torch ↔ onnx: {max_diff_onnx:.6e}", flush=True)

    summary = {
        "output_onnx":          args.output_onnx,
        "obs_dim":              int(obs_dim),
        "action_low":           action_low.tolist(),
        "action_high":          action_high.tolist(),
        "warmup_episodes":      args.warmup_episodes,
        "warmup_steps":         int(total_steps),
        "obs_mean":             obs_mean.tolist(),
        "obs_var":              obs_var.tolist(),
        "max_diff_wrapper_sb3": max_diff,
        "max_diff_torch_onnx":  max_diff_onnx,
    }
    print("\n__EXPORT_RESULT__=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
