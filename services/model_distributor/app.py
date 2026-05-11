"""
services/model_distributor/app.py — Distribuidor HTTP de modelos para el edge.

Sirve los ficheros de /app/models con un manifest indexado por sha256.
El edge (sync-agent) consulta /manifest, compara hashes y descarga los
ficheros que cambien.

Diseño explícitamente no-MLflow: la fuente de verdad es el filesystem
(`prod/models/`), que es lo mismo que montan las APIs de inferencia. No
duplicamos estado en una base de datos de registry.

Endpoints:
  GET  /health                — liveness
  GET  /manifest              — JSON con cada fichero servido y su sha256
  GET  /artifacts/<filename>  — fichero binario
                                (cabeceras: Content-Length, X-Version=sha256)

Variables de entorno:
  MODELS_DIR        directorio servido (default /app/models)
  ALLOWED_FILES     lista CSV de nombres permitidos. Si vacía, se sirve
                    cualquier *.onnx / *.pkl (whitelist por extensión).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, send_from_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("model-distributor")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))

_allowed_csv = os.getenv("ALLOWED_FILES", "").strip()
ALLOWED_FILES: set[str] | None = (
    {f.strip() for f in _allowed_csv.split(",") if f.strip()}
    if _allowed_csv else None
)
ALLOWED_EXTS = {".onnx", ".pkl"}


# ──────────────────────────────────────────────────────────────────────────────
# Caché de hashes (key = (path, mtime_ns, size))
# ──────────────────────────────────────────────────────────────────────────────

_hash_cache: dict[tuple[str, int, int], str] = {}
_hash_lock = threading.Lock()


def _sha256_cached(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _hash_lock:
        cached = _hash_cache.get(key)
        if cached is not None:
            return cached
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    with _hash_lock:
        _hash_cache[key] = digest
    return digest


def _is_servable(name: str) -> bool:
    """¿Está permitido servir este fichero?"""
    if name.startswith(".") or "/" in name or ".." in name:
        return False
    if ALLOWED_FILES is not None:
        return name in ALLOWED_FILES
    return Path(name).suffix.lower() in ALLOWED_EXTS


def _list_files() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(
        p.name for p in MODELS_DIR.iterdir()
        if p.is_file() and _is_servable(p.name)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        models_dir=str(MODELS_DIR),
        n_files=len(_list_files()),
    )


@app.get("/manifest")
def manifest():
    files = {}
    for name in _list_files():
        try:
            files[name] = _sha256_cached(MODELS_DIR / name)
        except OSError as exc:
            log.warning("No se pudo hashear %s: %s", name, exc)
    return jsonify(
        files=files,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/artifacts/<path:filename>")
def artifact(filename: str):
    if not _is_servable(filename):
        abort(404)
    full = MODELS_DIR / filename
    if not full.is_file():
        abort(404)
    response = send_from_directory(
        directory=str(MODELS_DIR),
        path=filename,
        as_attachment=False,
        mimetype="application/octet-stream",
    )
    try:
        response.headers["X-Version"] = _sha256_cached(full)
    except OSError:
        pass
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8004)
