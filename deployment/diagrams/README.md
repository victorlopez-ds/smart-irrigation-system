# Diagrama de arquitectura — Sistema de Riego Inteligente (v3.0)

Diagramas de arquitectura del sistema MLOps escritos en **Mermaid.js**. El
código es la fuente de verdad; los SVG/PNG se generan a partir de él.

## Ficheros

| Fichero | Contenido |
|---|---|
| `arquitectura.mmd` | Diagrama principal (3 zonas, 6 sub-zonas, ciclos diario y mensual) |
| `leyenda.mmd`      | Leyenda complementaria (tipos de nodo, flecha y ciclos) |

## Cómo renderizar

### Opción A — Online (más rápido para iterar)

Pega el contenido en https://mermaid.live y exporta SVG/PNG desde la UI.

### Opción B — CLI local (para incluir en LaTeX o CI)

```bash
# Instalar mermaid-cli (una vez)
npm install -g @mermaid-js/mermaid-cli

# Renderizar los dos ficheros
mmdc -i arquitectura.mmd -o arquitectura.svg -w 2400 -H 1400 -b transparent
mmdc -i leyenda.mmd      -o leyenda.svg      -w 1600 -H  900 -b transparent

# Para LaTeX, exportar también a PDF
mmdc -i arquitectura.mmd -o arquitectura.pdf
mmdc -i leyenda.mmd      -o leyenda.pdf
```

### Opción C — VS Code

Instala la extensión **Markdown Preview Mermaid Support** o
**Mermaid Preview**. Abre el `.mmd` y pulsa `Ctrl+Shift+V`.

## Convenciones

### Tipos de nodo (clases CSS)
| Color | Clase | A qué se aplica |
|---|---|---|
| 🟦 Azul claro | `service` | APIs always-on (Flask + ONNX/runtime) |
| 🟧 Naranja | `ephemeral` | Training jobs efímeros (DockerOperator) |
| ⬜ Gris | `storage` | Volúmenes Docker, BBDD, source of truth |
| 🟩 Verde | `hardware` | Sensores, electroválvula |
| 🟨 Amarillo | `external` | APIs de terceros (AEMET) |
| 🟦 Azul oscuro | `gateway` | Brokers y proxies con auth |

### Tipos de flecha
| Estilo Mermaid | Significado |
|---|---|
| `==>` (gruesa) | Flujo en tiempo real / canónico |
| `-->` (fina)   | HTTP síncrono |
| `-.->` (punteada) | Async, read-only, o tracking secundario |

### Color de flecha (por `linkStyle`)
| Color | Significado |
|---|---|
| 🔴 Rojo `#C0392B` | Ciclo diario de operación (pasos 1–6) |
| 🔵 Azul `#2874A6` | Ciclo mensual — orquestación (paso Ⓐ–Ⓑ) |
| 🟣 Morado `#6C3483` | Ciclo mensual — distribución (paso Ⓒ–Ⓓ) |
| 🟡 Amarillo `#B7950B` | Pull diario AEMET (paso 2) |
| ⚪ Gris `#7F8C8D` | Auxiliar (RO mounts, tracking, reload) |

## Mantenimiento

Si añades una flecha nueva:
1. Añádela en el bloque de flechas correspondiente, **al final de su grupo**
   (esto evita renumerar los `linkStyle`).
2. Añade una entrada en `linkStyle` con el índice nuevo. Lleva la cuenta de
   los índices con los comentarios `%% [N]` que hay junto a cada flecha.

Si añades un nodo nuevo:
1. Asígnale una `:::clase` existente. Solo crea una clase nueva si el rol del
   nodo no encaja con ninguna de las 6 actuales.

## Cambios respecto a versiones anteriores del diagrama (draw.io)

Esta versión consolida todas las correcciones discutidas:

- ✅ Añadido `edge-mqtt-client` como punto de entrada de telemetría en el edge
- ✅ Flechas MQTT con sentido correcto (broker → suscriptores)
- ✅ Paso 3 explícito: `edge-cron → ingest-svc GET /rl/obs`
- ✅ Paso 6 etiquetado: `POST /rl/transition (async)`
- ✅ AEMET dibujado como **pull diario orquestado por Airflow**, no push
- ✅ MLflow desconectado de la distribución — solo tracking
- ✅ Volumen `models/` explícito como **source of truth** compartido entre
  training jobs y `model-distributor`
- ✅ NGINX representado como gateway de seguridad con bearer token + whitelist
  de rutas
- ✅ Leyenda separada distingue los dos cilindros (DuckDB ≠ models/)
