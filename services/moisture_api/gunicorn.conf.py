"""gunicorn config para moisture-api.

1 worker síncrono: las consultas son esporádicas (sanity checks /
inspección), no hay ganancia con varios workers.
"""

bind          = "0.0.0.0:8002"
workers       = 1
worker_class  = "sync"
timeout       = 30
accesslog     = "-"
errorlog      = "-"
loglevel      = "info"
preload_app   = False
