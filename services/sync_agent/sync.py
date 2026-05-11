"""
services/sync_agent/sync.py — Cliente de sincronización de modelos para edge.

Cada ciclo:
  1. GET <CENTRAL_URL>/manifest
  2. Para cada fichero servido:
       - calcula sha256 local
       - si difiere o no existe → descarga a <name>.tmp.<pid>, valida
         hash, hace os.replace()
  3. Si cambió algún fichero de un grupo → POST <reload_url>

Entradas:
  CENTRAL_URL          URL base del model-distributor (http://model-distributor:8004)
  MODELS_DIR           directorio local (default /app/models)
  GROUPS_JSON          JSON con la definición de grupos. Ejemplo:
      {
        "anomaly": {
          "files": ["anomaly_cnn_lstm.onnx", "anomaly_scaler.pkl"],
          "reload_url": "http://anomaly-api:8001/reload"
        },
        "rl": {
          "files": ["rl_actor.onnx"],
          "reload_url": "http://rl-api:8003/reload"
        }
      }
  HTTP_TIMEOUT         segundos (default 30)

CLI:
  python -m sync          # un ciclo, salida JSON
  python -m sync --once   # alias
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("sync-agent")

CENTRAL_URL  = os.getenv("CENTRAL_URL", "http://nginx-central:8080").rstrip("/")
MODELS_DIR   = Path(os.getenv("MODELS_DIR", "/app/models"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
SYNC_TOKEN   = os.getenv("SYNC_TOKEN", "").strip()

_AUTH_HEADERS = {"Authorization": f"Bearer {SYNC_TOKEN}"} if SYNC_TOKEN else {}

# Mapping grupo → {files, reload_url} en JSON serializado
_GROUPS_DEFAULT: dict[str, dict] = {
    "anomaly": {
        "files": ["anomaly_cnn_lstm.onnx", "anomaly_scaler.pkl"],
        "reload_url": "http://anomaly-api:8001/reload",
    },
    "rl": {
        "files": ["rl_actor.onnx"],
        "reload_url": "http://rl-api:8003/reload",
    },
}
_groups_env = os.getenv("GROUPS_JSON")
GROUPS: dict[str, dict] = (
    json.loads(_groups_env) if _groups_env else _GROUPS_DEFAULT
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_atomic(remote_url: str, dest: Path, expected_sha: str | None) -> str:
    """Descarga remote_url a dest de forma atómica. Devuelve sha256 final."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".tmp.", dir=str(dest.parent))
    tmp = Path(tmp_name)
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, requests.get(
            remote_url, stream=True, timeout=HTTP_TIMEOUT, headers=_AUTH_HEADERS
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                out.write(chunk)
                h.update(chunk)
        digest = h.hexdigest()
        if expected_sha and digest != expected_sha:
            raise ValueError(
                f"hash mismatch para {dest.name}: "
                f"esperado {expected_sha[:12]}…, obtenido {digest[:12]}…"
            )
        # mkstemp crea el fichero con 0600; lo dejamos legible para los
        # contenedores de inferencia que comparten el volumen con otro UID.
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
        return digest
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def sync_once() -> dict:
    """Ejecuta un ciclo completo. Devuelve un resumen serializable."""
    summary: dict = {
        "central_url": CENTRAL_URL,
        "started_at":  datetime.now(timezone.utc).isoformat(),
        "downloaded":  [],
        "skipped":     [],
        "missing":     [],
        "reloaded":    [],
        "reload_errors": [],
        "errors":      [],
    }

    # ── 1) Manifest ──────────────────────────────────────────────────────────
    try:
        r = requests.get(f"{CENTRAL_URL}/manifest", timeout=HTTP_TIMEOUT, headers=_AUTH_HEADERS)
        r.raise_for_status()
        manifest = r.json().get("files", {})
    except Exception as exc:
        log.error("Fallo obteniendo manifest: %s", exc)
        summary["errors"].append(f"manifest: {exc}")
        return summary

    log.info("Manifest: %d ficheros disponibles en central", len(manifest))

    # ── 2) Descarga selectiva ────────────────────────────────────────────────
    changed_groups: set[str] = set()

    for group, cfg in GROUPS.items():
        for fname in cfg["files"]:
            remote_sha = manifest.get(fname)
            local_path = MODELS_DIR / fname

            if remote_sha is None:
                summary["missing"].append(fname)
                log.warning("[%s] %s no está en central", group, fname)
                continue

            local_sha: str | None = None
            if local_path.exists():
                try:
                    local_sha = _sha256(local_path)
                except OSError as exc:
                    summary["errors"].append(f"hash local {fname}: {exc}")
                    continue

            if local_sha == remote_sha:
                summary["skipped"].append({"file": fname, "sha": remote_sha[:12]})
                continue

            url = f"{CENTRAL_URL}/artifacts/{fname}"
            log.info("[%s] descargando %s (%s → %s)",
                     group, fname,
                     (local_sha or "ausente")[:12], remote_sha[:12])
            try:
                final_sha = _download_atomic(url, local_path, remote_sha)
            except Exception as exc:
                log.error("[%s] error descargando %s: %s", group, fname, exc)
                summary["errors"].append(f"download {fname}: {exc}")
                continue

            summary["downloaded"].append({"file": fname, "sha": final_sha[:12]})
            changed_groups.add(group)

    # ── 3) Reload de cada grupo afectado ─────────────────────────────────────
    for group in sorted(changed_groups):
        url = GROUPS[group]["reload_url"]
        try:
            r = requests.post(url, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            summary["reloaded"].append({"group": group, "response": r.json()})
            log.info("[%s] /reload OK: %s", group, r.json())
        except Exception as exc:
            log.warning("[%s] /reload falló: %s", group, exc)
            summary["reload_errors"].append({"group": group, "error": str(exc)})

    # ── 4) Persistir estado (solo diagnóstico) ───────────────────────────────
    state_path = MODELS_DIR / ".sync_state.json"
    try:
        state_path.write_text(json.dumps(summary, indent=2))
    except OSError as exc:
        log.warning("No se pudo escribir %s: %s", state_path, exc)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="Compatibilidad; el script siempre ejecuta un ciclo.")
    ap.parse_args()

    summary = sync_once()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
