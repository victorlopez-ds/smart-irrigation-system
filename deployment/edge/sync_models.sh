#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# sync_models.sh — Sincroniza los modelos del edge con MLflow Model Registry
#
# Ejecutado por un systemd timer diariamente a las 01:00 en la Raspberry Pi.
# Descarga las versiones en Production de los modelos registrados si han
# cambiado respecto a la versión local.
#
# Variables de entorno requeridas:
#   MLFLOW_TRACKING_URI   URL del servidor MLflow (ej: http://192.168.1.10:5000)
#   MODELS_DIR            Directorio local donde se guardan los modelos
#                         (por defecto: /app/models)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
MODELS_DIR="${MODELS_DIR:-/app/models}"
VERSION_FILE="${MODELS_DIR}/.versions"

mkdir -p "${MODELS_DIR}"
touch "${VERSION_FILE}"

# ── Función: obtiene la última versión Production de un modelo ────────────────
get_latest_version() {
    local model_name="$1"
    curl -sf \
        "${MLFLOW_URI}/api/2.0/mlflow/registered-models/get-latest-versions" \
        --data-urlencode "name=${model_name}" \
        --data-urlencode "stages=Production" \
        -G | python3 -c "
import sys, json
data = json.load(sys.stdin)
versions = data.get('model_versions', [])
if versions:
    v = versions[0]
    print(v['version'], v['source'])
else:
    print('', '')
"
}

# ── Función: descarga un artefacto desde MLflow ───────────────────────────────
download_artifact() {
    local artifact_uri="$1"
    local output_path="$2"

    # artifact_uri es del tipo: mlflow-artifacts:/1/abc123/artifacts/edge_model/tqc_actor.pt
    # Usamos la API REST de MLflow para descargarlo
    local run_id
    run_id=$(echo "${artifact_uri}" | grep -oP '(?<=/)\w{32}(?=/)' || echo "")

    if [[ -n "${run_id}" ]]; then
        local artifact_path
        artifact_path=$(echo "${artifact_uri}" | sed 's|.*artifacts/||')
        curl -sf \
            "${MLFLOW_URI}/api/2.0/mlflow-artifacts/artifacts/${artifact_path}" \
            -o "${output_path}"
        echo "  ✅ Descargado: ${output_path}"
    else
        echo "  ⚠️  No se pudo parsear artifact_uri: ${artifact_uri}"
    fi
}

# ── Sincronización del modelo RL ──────────────────────────────────────────────
sync_model() {
    local model_name="$1"
    local filename="$2"

    echo "Comprobando ${model_name}..."

    read -r remote_version artifact_source <<< "$(get_latest_version "${model_name}")"

    if [[ -z "${remote_version}" ]]; then
        echo "  ⚠️  No hay versión Production para ${model_name}. Saltando."
        return
    fi

    local local_version
    local_version=$(grep "^${model_name}=" "${VERSION_FILE}" | cut -d= -f2 || echo "")

    if [[ "${remote_version}" == "${local_version}" ]]; then
        echo "  ✅ Ya está actualizado (versión ${remote_version})"
        return
    fi

    echo "  📥 Nueva versión detectada: ${remote_version} (local: ${local_version:-ninguna})"
    download_artifact "${artifact_source}/${filename}" "${MODELS_DIR}/${filename}"

    # Actualizar versión local
    if grep -q "^${model_name}=" "${VERSION_FILE}"; then
        sed -i "s|^${model_name}=.*|${model_name}=${remote_version}|" "${VERSION_FILE}"
    else
        echo "${model_name}=${remote_version}" >> "${VERSION_FILE}"
    fi

    # Reiniciar el servicio de inferencia si está activo
    if systemctl is-active --quiet irridea-edge 2>/dev/null; then
        systemctl restart irridea-edge
        echo "  🔄 Servicio irridea-edge reiniciado"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "=== Sincronización de modelos $(date -Iseconds) ==="
sync_model "rl-agent"         "tqc_actor.pt"
sync_model "anomaly-detector" "anomaly_cnn_lstm.onnx"
echo "=== Sincronización completada ==="
