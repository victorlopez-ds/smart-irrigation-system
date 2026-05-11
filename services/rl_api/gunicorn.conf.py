"""gunicorn config para rl-api.

1 worker síncrono: una predicción al día en producción, no hay nada que
paralelizar. El modelo ONNX se carga una sola vez en memoria.
"""

bind          = "0.0.0.0:8003"
workers       = 1
worker_class  = "sync"
timeout       = 30
accesslog     = "-"
errorlog      = "-"
loglevel      = "info"
preload_app   = False
