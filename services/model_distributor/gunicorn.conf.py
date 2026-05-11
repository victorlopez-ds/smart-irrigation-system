"""gunicorn config para model-distributor.

2 workers síncronos: el caso típico es 1 GET /manifest cada N min desde
edges; con 2 workers absorbemos un /artifacts (descarga larga) sin
bloquear /health o /manifest.
"""

bind          = "0.0.0.0:8004"
workers       = 2
worker_class  = "sync"
timeout       = 60
accesslog     = "-"
errorlog      = "-"
loglevel      = "info"
preload_app   = False
