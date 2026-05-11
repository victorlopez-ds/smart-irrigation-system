"""gunicorn config para anomaly-api.

1 worker síncrono: el modelo ONNX se carga una sola vez en memoria; con N
workers tendríamos N copias del modelo (RAM × N) sin ganancia (1 petición
cada 15 min).
"""

bind          = "0.0.0.0:8001"
workers       = 1
worker_class  = "sync"
timeout       = 30
accesslog     = "-"
errorlog      = "-"
loglevel      = "info"
preload_app   = False
