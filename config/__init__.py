"""
prod/config/__init__.py — Carga centralizada de la configuración.

Uso en cualquier módulo de prod/:
    from config import cfg

    model_path = cfg["models"]["moisture_mlp"]
    threshold  = cfg["anomaly"]["threshold"]
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg: dict = _load_config()


def get(key_path: str, default=None):
    """
    Accede a una clave anidada con notación de puntos.
    Ejemplo: get("irrigation_env.theta_min") → 20.0
    """
    keys = key_path.split(".")
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
