"""
src/moisture/train.py — Entrenamiento del predictor MLP de incrementos de humedad.

El MLP entrenado se usa como función de transición T(s,a)→s' dentro de
RiegoEnv. Solo se ejecuta en el servidor.

Uso:
    python -m src.moisture.train
"""

from __future__ import annotations

import argparse
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

DEFAULTS = dict(
    data_path="data/processed/dataset_diario.csv",
    model_output="models/modelo_nn_humedad.keras",
    scaler_x_output="models/scaler_x.pkl",
    scaler_y_output="models/scaler_y.pkl",
    testset_output="models/moisture_testset.npz",
    test_size=0.2,
    random_state=42,
)

FEATURES = ["Vt", "Rt", "Rt_lag", "Pt", "Et"]
TARGET = "Vt_inc"

MLP_PARAMS = dict(
    hidden_1=64,
    dropout=0.2,
    hidden_2=32,
    learning_rate=0.001,
    batch_size=16,
    epochs=100,
    validation_split=0.2,
)


def build_mlp(input_dim: int, params: dict = MLP_PARAMS):
    """Construye la arquitectura MLP: Dense(64,relu)→Dropout(0.2)→Dense(32,relu)→Dense(1)."""
    model = models.Sequential([
        layers.Dense(params["hidden_1"], activation="relu", input_shape=(input_dim,)),
        layers.Dropout(params["dropout"]),
        layers.Dense(params["hidden_2"], activation="relu"),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train(
    data_path: str = DEFAULTS["data_path"],
    model_output: str = DEFAULTS["model_output"],
    scaler_x_output: str = DEFAULTS["scaler_x_output"],
    scaler_y_output: str = DEFAULTS["scaler_y_output"],
    testset_output: str | None = DEFAULTS["testset_output"],
    test_size: float = DEFAULTS["test_size"],
    random_state: int = DEFAULTS["random_state"],
) -> dict:
    """
    Entrena el predictor MLP de incrementos de humedad.

    Returns
    -------
    dict con r2, mae, epochs, testset_path
    """
    df = pd.read_csv(data_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Escaladores: imprescindibles para el MLP y requeridos por RiegoEnv
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_sc = scaler_x.fit_transform(X_train)
    X_test_sc = scaler_x.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    model = build_mlp(input_dim=len(FEATURES))

    early_stop = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    with mlflow.start_run(run_name="moisture_mlp"):
        mlflow.log_params(MLP_PARAMS)

        history = model.fit(
            X_train_sc, y_train_sc,
            epochs=MLP_PARAMS["epochs"],
            batch_size=MLP_PARAMS["batch_size"],
            validation_split=MLP_PARAMS["validation_split"],
            callbacks=[early_stop],
            verbose=1,
        )

        # Evaluación sobre test (en escala original)
        y_pred_sc = model.predict(X_test_sc)
        y_pred = scaler_y.inverse_transform(y_pred_sc).flatten()
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        epochs_run = len(history.history["loss"])

        mlflow.log_metrics({"r2": r2, "mae": mae, "epochs": epochs_run})

        os.makedirs(os.path.dirname(model_output) or ".", exist_ok=True)
        model.save(model_output)
        joblib.dump(scaler_x, scaler_x_output)
        joblib.dump(scaler_y, scaler_y_output)

        mlflow.log_artifact(model_output)
        mlflow.log_artifact(scaler_x_output)
        mlflow.log_artifact(scaler_y_output)

        if testset_output:
            os.makedirs(os.path.dirname(testset_output) or ".", exist_ok=True)
            # Guardamos el test set en escala ORIGINAL (sin escalar) para que el
            # champion (que vive como ONNX y trae sus propios scalers) lo pueda
            # consumir sin acoplarse a los scalers del challenger.
            np.savez(
                testset_output,
                X_test=X_test.values.astype(np.float32),
                y_test=y_test.values.astype(np.float32),
            )

    print(f"✅ MLP guardado: {model_output}  R²={r2:.4f}  MAE={mae:.4f}  epochs={epochs_run}")
    return {
        "r2": r2,
        "mae": mae,
        "epochs": epochs_run,
        "testset_path": testset_output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULTS["data_path"])
    parser.add_argument("--model_output", default=DEFAULTS["model_output"])
    args = parser.parse_args()
    train(data_path=args.data_path, model_output=args.model_output)
