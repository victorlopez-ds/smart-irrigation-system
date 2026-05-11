"""
gunicorn.conf.py — Configuración para ingest-svc.

Importante: workers = 1 porque la conexión a DuckDB y el cliente MQTT
mantienen estado en proceso. Múltiples workers crearían N suscriptores y
N intentos de abrir el mismo fichero DuckDB → conflictos de lock.
"""

bind          = "0.0.0.0:8000"
workers       = 1
worker_class  = "sync"
timeout       = 30
accesslog     = "-"   # stdout
errorlog      = "-"
loglevel      = "info"
preload_app   = False  # importante: cada worker arranca sus hilos en su propio proceso
