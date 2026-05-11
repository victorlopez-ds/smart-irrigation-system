"""
src/anomaly/export.py — Convierte el modelo CNN-LSTM de Keras a ONNX.

Uso:
    python -m src.anomaly.export \
        --model_path models/anomaly_cnn_lstm.keras \
        --output_path models/anomaly_cnn_lstm.onnx \
        --window 30

Requiere: tf2onnx  (pip install tf2onnx)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
import tf2onnx
from tensorflow import keras


def export_keras_to_onnx(
    model_path: str,
    output_path: str,
    window: int = 30,
    n_features: int = 1,
) -> None:
    """
    Convierte un modelo Keras CNN-LSTM a formato ONNX.

    Parameters
    ----------
    model_path  : ruta al fichero .keras
    output_path : ruta de salida para el fichero .onnx
    window      : longitud de la ventana temporal
    n_features  : número de features de entrada por paso temporal
    """
    model = keras.models.load_model(model_path)
    print(f"Modelo cargado: {model_path}")
    model.summary()

    input_signature = [
        tf2onnx.tf_loader.tf.TensorSpec(
            shape=(None, window, n_features),
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

    print(f"✅ Modelo exportado a ONNX: {output_path}")

    # Test de equivalencia numérica
    import onnxruntime as ort

    example_input = np.zeros((1, window, n_features), dtype=np.float32)

    keras_pred = model.predict(example_input, verbose=0)

    session = ort.InferenceSession(output_path)
    input_name = session.get_inputs()[0].name
    onnx_pred = session.run(None, {input_name: example_input})[0]

    max_diff = float(np.abs(keras_pred - onnx_pred).max())
    assert max_diff < 1e-4, f"❌ Discrepancia keras↔onnx: {max_diff}"
    print(f"✅ Test de equivalencia numérica superado (diff máx: {max_diff:.2e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta CNN-LSTM a ONNX")
    parser.add_argument("--model_path", default="models/anomaly_cnn_lstm.keras")
    parser.add_argument("--output_path", default="models/anomaly_cnn_lstm.onnx")
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    export_keras_to_onnx(args.model_path, args.output_path, args.window)
