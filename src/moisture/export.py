"""
src/moisture/export.py — Convierte el MLP de humedad de Keras a ONNX.

Aunque este modelo solo se usa internamente en el servidor (dentro de
RiegoEnv), exportarlo a ONNX permite:
  - Integración futura en pipelines de inferencia sin TensorFlow
  - Verificación de equivalencia numérica keras ↔ onnx

Uso:
    python -m src.moisture.export \
        --model_path models/modelo_nn_humedad.keras \
        --output_path models/moisture_mlp.onnx

Requiere: tf2onnx  (pip install tf2onnx)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
import tf2onnx
from tensorflow import keras

N_FEATURES = 5  # Vt, Rt, Rt_lag, Pt, Et


def export_mlp_to_onnx(
    model_path: str,
    output_path: str,
    n_features: int = N_FEATURES,
) -> None:
    """
    Convierte el MLP de Keras a formato ONNX.

    Parameters
    ----------
    model_path  : ruta al fichero .keras
    output_path : ruta de salida para el fichero .onnx
    n_features  : número de features de entrada
    """
    model = keras.models.load_model(model_path)
    print(f"Modelo cargado: {model_path}")
    model.summary()

    input_signature = [
        tf2onnx.tf_loader.tf.TensorSpec(
            shape=(None, n_features),
            dtype=tf2onnx.tf_loader.tf.float32,
            name="input",
        )
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=17,
        output_path=output_path,
    )

    print(f"✅ MLP exportado a ONNX: {output_path}")

    # Test de equivalencia numérica
    import onnxruntime as ort

    example_input = np.random.randn(1, n_features).astype(np.float32)
    keras_pred = model.predict(example_input, verbose=0)

    session = ort.InferenceSession(output_path)
    input_name = session.get_inputs()[0].name
    onnx_pred = session.run(None, {input_name: example_input})[0]

    max_diff = float(np.abs(keras_pred - onnx_pred).max())
    assert max_diff < 1e-4, f"❌ Discrepancia keras↔onnx: {max_diff}"
    print(f"✅ Test de equivalencia numérica superado (diff máx: {max_diff:.2e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta MLP de humedad a ONNX")
    parser.add_argument("--model_path", default="models/modelo_nn_humedad.keras")
    parser.add_argument("--output_path", default="models/moisture_mlp.onnx")
    args = parser.parse_args()

    export_mlp_to_onnx(args.model_path, args.output_path)
