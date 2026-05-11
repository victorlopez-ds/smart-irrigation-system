"""
src/anomaly/train.py — Entrenamiento del detector de anomalías hidráulicas (CNN-LSTM).

Uso:
    python -m src.anomaly.train [--window 30] [--epochs 100]

El modelo entrenado (.keras) y el scaler se guardan en `models/` y se
registran en MLflow si MLFLOW_TRACKING_URI está configurado.
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

DEFAULTS = dict(
    raw_watermeter="data/raw/contador_agua_albudeite.json",
    raw_solenoid="data/raw/electrovalvula_albudeite.json",
    model_output="models/anomaly_cnn_lstm.keras",
    scaler_output="models/anomaly_scaler.pkl",
    testset_output="models/anomaly_testset.npz",
    window=30,
    epochs=100,
    batch_size=128,
    val_split=0.1,
    test_split=0.2,
    downsample_ratio=0.1,
    threshold=0.25,
)

# Configuración del modelo ganador (ID2 CNN-LSTM Standard del experimento)
BEST_CONFIG = dict(
    cnn_layers=[32],
    lstm_layers=[64],
    dropout=0.1,
    learning_rate=0.001,
    use_pooling=False,
)


# ──────────────────────────────────────────────────────────────────────────────
# Carga y preprocesamiento de datos
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_data(watermeter_path: str, solenoid_path: str) -> pd.DataFrame:
    """Carga los JSON de contador y electroválvula, los une y resamplea a 15 min."""
    with open(watermeter_path) as f:
        wm = json.load(f)
    with open(solenoid_path) as f:
        sol = json.load(f)

    df_wm = pd.DataFrame(wm)
    df_sol = pd.DataFrame(sol)

    df_wm["TimeInstant"] = pd.to_datetime(df_wm["TimeInstant"])
    df_sol["TimeInstant"] = pd.to_datetime(df_sol["TimeInstant"])

    df_wm.set_index("TimeInstant", inplace=True)
    df_sol.set_index("TimeInstant", inplace=True)

    # Diferencia de volumen (litros por intervalo)
    df_wm["vol_diff"] = df_wm["vol"].diff()
    df_wm.loc[
        (df_wm["vol_diff"] < 0) | (df_wm["vol_diff"] > 4000), "vol_diff"
    ] = np.nan

    # Electroválvula → 15 min
    df_sol_15 = df_sol.resample("1min").ffill().resample("15min").sum()
    df_wm_15 = df_wm.resample("15min").agg({"vol_diff": "sum"})

    df = df_wm_15.join(df_sol_15[["status"]], how="inner").dropna()
    return df


def create_sequences(X: np.ndarray, y: np.ndarray, window: int):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i : i + window])
        ys.append(y[i + window])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def downsample_inactive(X: np.ndarray, y: np.ndarray, ratio: float = 0.1):
    """Reduce el desbalance entre ventanas con y sin actividad de riego."""
    epsilon = 1e-4
    umbral = np.min(X[:, :, 0]) + epsilon
    active = [i for i in range(len(X)) if np.any(X[i, :, 0] > umbral)]
    inactive = [i for i in range(len(X)) if i not in active]

    np.random.seed(42)
    n_keep = int(len(inactive) * ratio)
    inactive_keep = np.random.choice(inactive, size=n_keep, replace=False)
    idx = np.concatenate([active, inactive_keep])
    np.random.shuffle(idx)
    return X[idx], y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Construcción del modelo
# ──────────────────────────────────────────────────────────────────────────────

def build_model(config: dict, input_shape: tuple) -> Sequential:
    model = Sequential()

    for i, filters in enumerate(config.get("cnn_layers", [])):
        kwargs = dict(filters=filters, kernel_size=3, padding="same", activation="relu")
        if i == 0:
            kwargs["input_shape"] = input_shape
        model.add(Conv1D(**kwargs))
        if config.get("use_pooling", False):
            model.add(MaxPooling1D(pool_size=2))

    lstm_layers = config.get("lstm_layers", [64])
    for i, units in enumerate(lstm_layers):
        is_last = i == len(lstm_layers) - 1
        first = not config.get("cnn_layers") and i == 0
        lstm_kwargs = dict(return_sequences=not is_last)
        if first:
            lstm_kwargs["input_shape"] = input_shape
        model.add(LSTM(units, **lstm_kwargs))
        if config.get("dropout", 0) > 0:
            model.add(Dropout(config["dropout"]))

    model.add(Dense(1))
    model.compile(
        optimizer=Adam(learning_rate=config.get("learning_rate", 0.001)),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ──────────────────────────────────────────────────────────────────────────────

def train(
    raw_watermeter: str = DEFAULTS["raw_watermeter"],
    raw_solenoid: str = DEFAULTS["raw_solenoid"],
    model_output: str = DEFAULTS["model_output"],
    scaler_output: str = DEFAULTS["scaler_output"],
    testset_output: str = DEFAULTS["testset_output"],
    window: int = DEFAULTS["window"],
    epochs: int = DEFAULTS["epochs"],
    batch_size: int = DEFAULTS["batch_size"],
    val_split: float = DEFAULTS["val_split"],
    test_split: float = DEFAULTS["test_split"],
    downsample_ratio: float = DEFAULTS["downsample_ratio"],
    threshold: float = DEFAULTS["threshold"],
) -> dict:
    df = load_raw_data(raw_watermeter, raw_solenoid)

    # Escalado
    scaler = StandardScaler()
    df["vol_diff_scaled"] = scaler.fit_transform(df[["vol_diff"]])

    # Partición temporal: test_split del final
    n_test = int(len(df) * test_split)
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]

    X_train, y_train = create_sequences(
        train_df[["status"]].values, train_df["vol_diff_scaled"].values, window
    )
    X_test, y_test = create_sequences(
        test_df[["status"]].values, test_df["vol_diff_scaled"].values, window
    )
    X_train, y_train = downsample_inactive(X_train, y_train, downsample_ratio)

    model = build_model(BEST_CONFIG, input_shape=(window, 1))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    ]

    with mlflow.start_run(run_name="anomaly_cnn_lstm"):
        mlflow.log_params({**BEST_CONFIG, "window": window, "threshold": threshold})

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluación sobre test
        y_pred = model.predict(X_test).reshape(-1)
        errors = np.abs(y_pred - y_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        anomaly_rate = float(np.mean(errors > threshold))

        mlflow.log_metrics({"test_mae": mae, "anomaly_rate": anomaly_rate})

        # Guardar artefactos
        os.makedirs(os.path.dirname(model_output) or ".", exist_ok=True)
        model.save(model_output)
        joblib.dump(scaler, scaler_output)

        # Guardar test set para champion-challenger en el DAG
        np.savez(testset_output, X_test=X_test, y_test=y_test)

        mlflow.log_artifact(model_output)
        mlflow.log_artifact(scaler_output)
        mlflow.log_artifact(testset_output)

    print(f"✅ Modelo guardado en: {model_output}  (MAE test: {mae:.4f})")
    return {
        "mae": mae,
        "anomaly_rate": anomaly_rate,
        "epochs": len(history.history["loss"]),
        "testset_path": testset_output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=DEFAULTS["window"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    args = parser.parse_args()
    train(window=args.window, epochs=args.epochs, threshold=args.threshold)
