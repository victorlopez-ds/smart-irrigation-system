# Sistema de riego inteligente

Sistema MLOps de riego inteligente basado en IoT y aprendizaje automático.
Desarrollado como Trabajo Fin de Grado en la Universidad de Murcia.

Tres subsistemas ML trabajan en conjunto:

- **Detector de anomalías hidráulicas** (CNN-LSTM, ONNX) — tiempo real cada 15 min
- **Predictor de incrementos de humedad** (MLP, ONNX) — componente interno del simulador RL
- **Agente de riego por refuerzo** (TQC, ONNX) — decisión diaria a las 06:00

## Arquitectura

```
SERVIDOR CENTRAL (Docker Compose)
  mosquitto ──> ingest-svc ──> DuckDB
                    |
               Airflow DAGs ──> MLflow Registry
                    |                   |
               AEMET API          nginx-central
               (Hargreaves)            |
                                  sync-agent (edge)
EDGE (Raspberry Pi / Docker Compose)
  edge-mqtt-client ──> anomaly-api (ONNX)
  edge-cron 06:00  ──> rl-api (ONNX) ──> electroválvula
  Grafana (:3000)  ──> ingest-svc (métricas)
```

## Estructura del repositorio

```
prod/
├── src/                        Código ML (train, infer, export)
│   ├── anomaly/                  CNN-LSTM anomaly detector
│   ├── moisture/                 MLP soil moisture predictor
│   ├── rl/                       TQC irrigation agent + RiegoEnv
│   └── common/                   Data loading, aggregation
├── services/                   Microservicios (Flask + gunicorn)
│   ├── ingest_svc/               MQTT -> DuckDB + REST API (:8000)
│   ├── anomaly_api/              Anomaly detection (:8001)
│   ├── moisture_api/             Moisture prediction (:8002)
│   ├── rl_api/                   RL inference (:8003)
│   ├── model_distributor/        Model serving (:8004, internal)
│   ├── sync_agent/               Model sync edge <- central
│   ├── edge_mqtt_client/         MQTT subscriber + valve (:8010)
│   └── edge_cron/                Cron scheduler (anomaly + irrigation)
├── pipelines/airflow/dags/     DAGs de Airflow
│   ├── ingest_dag.py             Daily: AEMET + aggregate (23:55)
│   ├── retrain_anomaly.py        Monthly: CNN-LSTM retrain (dia 1)
│   ├── retrain_moisture.py       Monthly: MLP retrain (dia 1)
│   ├── finetune_rl.py            Quarterly: TQC finetune (dia 1)
│   └── build_rl_transitions.py   Daily: build (s,a,r,s') tuples
├── deployment/
│   ├── docker-compose.central.yml   Servidor central
│   ├── docker-compose.edge.yml      Nodo edge
│   ├── grafana/                     Dashboards provisionados
│   ├── mosquitto/                   Config broker MQTT
│   ├── nginx/                       Reverse proxy + bearer auth
│   ├── docker/                      Dockerfiles de training
│   └── .env.example                 Plantilla de variables
├── config/                     Configuración centralizada (YAML)
├── tests/                      Tests unitarios
├── models/                     Artefactos entrenados (gitignored)
└── data/                       Datos raw + processed (gitignored)
```

## Requisitos previos

- Docker Engine >= 24.0 y Docker Compose v2
- Red Docker compartida: `docker network create irridea-net`
- (Opcional) GPU NVIDIA + nvidia-docker para entrenamiento
- (Opcional) API Key de AEMET (gratuita en https://opendata.aemet.es)

## Arranque rapido

### 1. Configurar variables de entorno

```bash
cp deployment/.env.example deployment/.env
# Editar deployment/.env con credenciales reales
```

Variables criticas a rellenar:

| Variable | Descripcion |
|---|---|
| `POSTGRES_PASSWORD` | Password de PostgreSQL (Airflow + MLflow) |
| `MQTT_USER` / `MQTT_PASS` | Credenciales del broker MQTT |
| `SYNC_TOKEN` | Bearer token para descarga de modelos |
| `AIRFLOW_FERNET_KEY` | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `AEMET_API_KEY` | Alta en opendata.aemet.es (opcional, usa defaults sin ella) |
| `HOST_PROJECT_DIR` | Ruta absoluta a `prod/` en el host |

### 2. Levantar el servidor central

```bash
docker network create irridea-net

docker compose --env-file deployment/.env \
    -f deployment/docker-compose.central.yml up -d
```

Servicios disponibles:

| Servicio | URL |
|---|---|
| Airflow | http://localhost:8080 (admin / $AIRFLOW_ADMIN_PASSWORD) |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin / irridea) |
| ingest-svc | http://localhost:8000/health |

### 3. Levantar el nodo edge

```bash
docker compose --env-file deployment/.env \
    -f deployment/docker-compose.edge.yml up -d
```

Servicios edge:

| Servicio | Puerto | Funcion |
|---|---|---|
| anomaly-api | 8001 | Detección de anomalías (ONNX) |
| rl-api | 8003 | Agente de riego (ONNX) |
| edge-mqtt-client | 8010 | Suscriptor MQTT + comando valvula |
| edge-cron | — | Scheduler: anomaly/15min + riego/06:00 |
| sync-agent | — | Descarga modelos desde central cada hora |

### 4. Poblar modelos (primera vez)

Si no hay modelos entrenados en `models/`, el entrenamiento inicial
se lanza con las imagenes de training:

```bash
# Anomaly detector
docker compose --env-file deployment/.env \
    -f deployment/docker-compose.central.yml run --rm \
    irridea-train-anomaly

# Moisture predictor
docker compose --env-file deployment/.env \
    -f deployment/docker-compose.central.yml run --rm \
    irridea-train-moisture

# RL agent export
docker compose --env-file deployment/.env \
    -f deployment/docker-compose.central.yml run --rm \
    irridea-rl-export
```

## Ciclos operacionales

| Ciclo | Frecuencia | Orquestador | Descripcion |
|---|---|---|---|
| Telemetría MQTT | Cada 15 min | mosquitto | Sensores -> ingest-svc -> DuckDB |
| Deteccion anomalías | Cada 15 min | edge-cron | edge-mqtt-client -> anomaly-api |
| Agregación diaria | 23:55 | Airflow | ingest-svc /aggregate |
| Forecast AEMET | 23:55 | Airflow | ingest-svc /weather/fetch (Hargreaves ET0) |
| Riego RL | 06:00 | edge-cron | /rl/obs -> /act -> /valve/command |
| Sync modelos | Cada hora | sync-agent | nginx-central -> models-edge volume |
| Retrain anomaly | Día 1 mensual | Airflow | Champion-challenger + MLflow |
| Retrain moisture | Día 1 mensual | Airflow | Champion-challenger + MLflow |
| Finetune RL | Día 1 trimestral | Airflow | Online RL con transiciones reales |

## Monitorizacion (Grafana)

Grafana se provisiona automáticamente con el dashboard
en http://localhost:3000 (credenciales: admin / irridea).

Paneles:
- **Caudal y anomalías** — línea de vol_diff + puntos rojos en anomalías
- **Humedad del suelo** — serie temporal (%)
- **ET0 y precipitación** — barras AEMET con Hargreaves
- **Riego RL** — acción diaria tau (minutos)
- **Histórico diario** — theta vs agua consumida

Los datos se consultan vía la API REST de ingest-svc (endpoints `/grafana/*`)
usando el plugin Infinity como datasource.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Seguridad

- Las credenciales se gestionan vía `.env` (gitignored)
- El broker MQTT requiere autenticación (user/password)
- La descarga de modelos usa bearer token via nginx reverse proxy
- Las APIs internas no estan expuestas al exterior (solo vía red Docker)

## Licencia

Proyecto académico — Universidad de Murcia, 2025-2026.
