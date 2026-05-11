"""
src/common/data_loader.py — Carga y preprocesamiento de datos históricos.

Expone funciones reutilizadas por los módulos de entrenamiento de los
tres subsistemas (anomalía, humedad, RL).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyet


# ──────────────────────────────────────────────────────────────────────────────
# Carga de ficheros raw
# ──────────────────────────────────────────────────────────────────────────────

def read_json(filepath: str | Path) -> dict | list:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def load_daily_dataset(csv_path: str = "data/processed/dataset_diario.csv") -> pd.DataFrame:
    """
    Carga el dataset diario preprocesado con columnas:
    Vt, Rt, Rt_lag, Pt, Et, Vt_inc, TimeInstant
    """
    df = pd.read_csv(csv_path, parse_dates=["TimeInstant"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Preprocesamiento completo (desde ficheros raw)
# ──────────────────────────────────────────────────────────────────────────────

def build_daily_dataset(
    raw_dir: str = "data/raw",
    output_path: str | None = "data/processed/dataset_diario.csv",
    elevation: float = 150.0,
    lat: float = 38.028,
) -> pd.DataFrame:
    """
    Construye el dataset diario completo desde los ficheros raw.

    Incluye:
    - Datos climáticos (temperatura, HR, viento, radiación)
    - Cálculo de ET₀ por FAO-56 Penman-Monteith (pyet)
    - Humedad del suelo (sensor diario)
    - Riego (electroválvula + contador)
    - Precipitación

    Returns
    -------
    DataFrame con columnas: Vt, Rt, Rt_lag, Pt, Et, Vt_inc, TimeInstant
    """
    raw = Path(raw_dir)

    # ── Climatología ──────────────────────────────────────────────────────────
    clim_data = read_json(raw / "climatologia_albudeite.json")
    col_names = [f["id"] for f in clim_data["fields"]]
    df_clim = pd.DataFrame(clim_data["records"], columns=col_names)
    df_clim["TimeInstant"] = pd.to_datetime(df_clim["TimeInstant"])
    df_clim.set_index("TimeInstant", inplace=True)

    cols_clim = ["apparentTemperature", "relativeHumidity", "windSpeed", "solarRadiation", "barometricPressure"]
    df_clim = df_clim[cols_clim].copy()

    # Eliminar outliers
    df_clim.loc[(df_clim["barometricPressure"] > 110) | (df_clim["barometricPressure"] < 90), "barometricPressure"] = np.nan
    df_clim.loc[(df_clim["windSpeed"] < 0) | (df_clim["windSpeed"] > 30), "windSpeed"] = np.nan
    df_clim.loc[df_clim["relativeHumidity"] < 10, "relativeHumidity"] = np.nan

    df_diario = df_clim.resample("D").agg({
        "apparentTemperature": ["max", "min", "mean"],
        "relativeHumidity": "mean",
        "windSpeed": "mean",
        "solarRadiation": "mean",
    })
    df_diario.columns = ["Tmax", "Tmin", "Tmean", "RH", "u2", "Rs"]

    # ── ET₀ (FAO-56) ─────────────────────────────────────────────────────────
    df_diario["Rs"] = df_diario["Rs"] * 0.0864  # W/m² → MJ/m²/día
    df_diario["ET0"] = pyet.pm_fao56(
        tmean=df_diario["Tmean"],
        wind=df_diario["u2"],
        rs=df_diario["Rs"],
        tmax=df_diario["Tmax"],
        tmin=df_diario["Tmin"],
        rh=df_diario["RH"],
        elevation=elevation,
        lat=lat,
    )

    # ── Humedad del suelo ─────────────────────────────────────────────────────
    sm_data = read_json(raw / "sensor_humedad_albudeite.json")
    sm_col = [f["id"] for f in sm_data["fields"]]
    df_sm = pd.DataFrame(sm_data["records"], columns=sm_col)
    df_sm["TimeInstant"] = pd.to_datetime(df_sm["TimeInstant"])
    df_sm.set_index("TimeInstant", inplace=True)
    df_suelo = df_sm[["soilMoisture"]].resample("D").asfreq()

    # ── Riego ─────────────────────────────────────────────────────────────────
    sol_data = read_json(raw / "electrovalvula_albudeite.json")
    wm_data = read_json(raw / "contador_agua_albudeite.json")

    df_sol = pd.DataFrame(sol_data)
    df_wm = pd.DataFrame(wm_data)
    df_sol["TimeInstant"] = pd.to_datetime(df_sol["TimeInstant"])
    df_wm["TimeInstant"] = pd.to_datetime(df_wm["TimeInstant"])
    df_sol.set_index("TimeInstant", inplace=True)
    df_wm.set_index("TimeInstant", inplace=True)

    df_wm["vol_diff"] = df_wm["vol"].diff()
    df_wm.loc[(df_wm["vol_diff"] < 0) | (df_wm["vol_diff"] > 4000), "vol_diff"] = np.nan

    df_sol_15 = df_sol.resample("1min").ffill().resample("15min").sum()
    df_wm_15 = df_wm.resample("15min").agg({"vol_diff": "sum"})
    df_irr = df_wm_15.join(df_sol_15[["status"]], how="inner")

    # Eliminar períodos con status constante (sensor congelado)
    umbral_rep = 4 * 24 * 7
    es_rep = df_irr["status"].diff() == 0
    gid = (df_irr["status"].diff() != 0).cumsum()
    conteo = df_irr.groupby(gid)["status"].transform("count")
    df_irr.loc[es_rep & (conteo >= umbral_rep), ["status", "vol_diff"]] = np.nan

    df_irr_daily = df_irr.resample("D").sum()

    # ── Precipitación ─────────────────────────────────────────────────────────
    df_prec = pd.read_csv(raw / "precipitaciones_mula.csv", sep=";", decimal=",")
    df_prec["FECHA"] = pd.to_datetime(df_prec["FECHA"], format="%d/%m/%y")
    df_prec.set_index("FECHA", inplace=True)
    # Algunas exportaciones meten doble espacio en los nombres ("PREC  (mm)"):
    # normalizamos a un único espacio.
    df_prec.columns = df_prec.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    drop_cols = [c for c in df_prec.columns if c not in ["PREC (mm)"]]
    df_prec.drop(columns=drop_cols, inplace=True, errors="ignore")
    df_prec = df_prec.resample("D").max().fillna(0)

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = (
        df_diario
        .join(df_suelo[["soilMoisture"]])
        .join(df_irr_daily[["vol_diff"]])
        .join(df_prec[["PREC (mm)"]])
    )
    df["soilMoistureNextDay"] = df["soilMoisture"].shift(-1)

    # Eliminar rachas largas de NaN (>14 días)
    es_nan = df["Tmax"].isna()
    grupos = (es_nan != es_nan.shift()).cumsum()
    conteo_nan = df.groupby(grupos)["Tmax"].transform("size")
    df = df[~(es_nan & (conteo_nan > 14))].copy()
    df = df.interpolate(method="linear").bfill()

    df = df.rename(columns={
        "soilMoisture": "Vt",
        "soilMoistureNextDay": "Vt_next",
        "vol_diff": "Rt",
        "PREC (mm)": "Pt",
        "ET0": "Et",
    })
    df = df[["Vt", "Rt", "Pt", "Et", "Vt_next"]].copy()
    df["Rt_lag"] = df["Rt"].shift(1).fillna(0)
    df["Vt_inc"] = df["Vt_next"] - df["Vt"]
    df = df.dropna(subset=["Vt_inc"])
    df["TimeInstant"] = df.index

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset guardado en: {output_path}  ({len(df)} filas)")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Carga desde DuckDB (stream en tiempo real)
# ──────────────────────────────────────────────────────────────────────────────

def load_features_from_db(
    db_path: str = "data/raw_stream/irridea.duckdb",
    n_days: int | None = None,
) -> pd.DataFrame:
    """
    Carga los features diarios desde la tabla features_daily de DuckDB.

    Parameters
    ----------
    db_path : ruta al fichero DuckDB
    n_days  : si se indica, devuelve solo los últimos n_days días

    Returns
    -------
    DataFrame con columnas: date, theta, water_used_l, precip_mm, et0_mm
    """
    con = duckdb.connect(db_path, read_only=True)
    query = "SELECT * FROM features_daily ORDER BY date"
    if n_days:
        query += f" WHERE date >= CURRENT_DATE - INTERVAL '{n_days} days'"
    df = con.execute(query).df()
    con.close()
    return df
