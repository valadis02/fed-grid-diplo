"""
cloud_evaluation_service.py
============================
Αξιολογεί το global model μετά από κάθε FL aggregation round.
Υποστηρίζει και LCL data (αρχεία απευθείας στο /app/data/) και
συνθετικά data (house_1/input_data.csv δομή).
"""
import os
import json
import time
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from cloud.communication.cloud_resources_paths import CloudResourcesPaths
from shared.logging_config import logger
from shared import normalization as norm

# ── Configuration ─────────────────────────────────────────────
SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH', 48))
BATCH_SIZE      = 256
WINDOWS         = [3, 6, 12, 24]

# Ίδιο train/eval παράθυρο με τα edges (βλ. model_training_service).
# Διόρθωση: παλιότερα το test set ήταν οι γραμμές 20*48..29*48 (λογική του
# 1-μηνιαίου dataset) — με 12 μήνες αυτές πέφτουν ΜΕΣΑ στο training period,
# και αξιολογούνταν μόνο τα house_1..10.
TRAINING_DATE = os.getenv('TRAINING_DATE', '2012-02-01')
TRAIN_DAYS    = int(os.getenv('TRAIN_DAYS', 274))
EVAL_DAYS     = int(os.getenv('EVAL_DAYS', 92))

FEATURE_COLS = [
    "value_diff",
    "value_rolling_mean_3",  "value_volatility_3",  "value_ewm_3",
    "value_rolling_mean_6",  "value_volatility_6",  "value_ewm_6",
    "value_rolling_mean_12", "value_volatility_12", "value_ewm_12",
    "value_rolling_mean_24", "value_volatility_24", "value_ewm_24",
    "drift_flag", "time_since_last_spike",
]

RESULTS_FILE = os.path.join(
    CloudResourcesPaths.STATUS_FOLDER_PATH.value,
    "fl_global_model_metrics.json"
)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"timestamp": "datetime", "consumption_kwh": "value"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "value"])
    df["value"] = df["value"].astype(float)

    for w in WINDOWS:
        df[f"value_rolling_mean_{w}"] = (
            df["value"].rolling(w).mean().interpolate().ffill().bfill()
        )

    df["value_diff"] = df["value"].diff().fillna(0)
    for w in WINDOWS:
        df[f"value_ewm_{w}"]        = df["value"].ewm(span=w, adjust=False).mean()
        df[f"value_volatility_{w}"] = df["value"].rolling(w).std().fillna(0)

    baseline_vol     = df["value"].rolling(6).std().fillna(0)
    df["drift_flag"] = (df["value"].diff().abs() > baseline_vol * 2).astype(int)

    counter, time_since = 0, []
    for flag in df["drift_flag"]:
        counter = 0 if flag == 1 else counter + 1
        time_since.append(counter)
    df["time_since_last_spike"] = time_since

    return df


def _find_csv_files(base: str) -> list:
    """Όλα τα house_N/input_data.csv (ταξινομημένα κατά N), αλλιώς *.csv."""
    paths = glob.glob(os.path.join(base, "house_*", "input_data.csv"))

    def _num(p):
        try:
            return int(os.path.basename(os.path.dirname(p)).split("_")[1])
        except Exception:
            return 10 ** 9
    paths = sorted(paths, key=_num)
    if not paths:
        paths = sorted(glob.glob(os.path.join(base, "*.csv")))
    return paths


def _windows():
    t0 = pd.to_datetime(TRAINING_DATE)
    tr = (t0, t0 + pd.Timedelta(days=TRAIN_DAYS))
    ev = (t0 + pd.Timedelta(days=TRAIN_DAYS + 1),
          t0 + pd.Timedelta(days=TRAIN_DAYS + 1 + EVAL_DAYS))
    return tr, ev


def _load_houses() -> list:
    """Επιστρέφει [(name, X_eval_norm, y_eval_norm, stats)] για κάθε σπίτι."""
    (tr0, tr1), (ev0, ev1) = _windows()
    out = []
    for base in ["/app/data", "./data"]:
        if not os.path.isdir(base):
            continue
        csv_files = _find_csv_files(base)
        if not csv_files:
            continue
        logger.info("Cloud eval: found %d houses in %s | eval window %s -> %s",
                    len(csv_files), base, ev0.date(), ev1.date())
        for path in csv_files:
            name = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
            try:
                raw = pd.read_csv(path)
                ts = pd.to_datetime(raw["timestamp"], errors="coerce")
                tr_df = _preprocess(raw[(ts >= tr0) & (ts < tr1)].copy())
                ev_df = _preprocess(raw[(ts >= ev0) & (ts < ev1)].copy())
                if len(ev_df) <= SEQUENCE_LENGTH:
                    logger.warning("Cloud eval: %s has only %d eval rows, skipping.", name, len(ev_df))
                    continue
                stats = norm.fit_stats(tr_df) if norm.enabled() else None
                ev_n = norm.transform_df(ev_df, stats)
                ev_n = ev_n.dropna(subset=FEATURE_COLS + ["value"])
                X = ev_n[FEATURE_COLS].astype("float32").values
                y = ev_n["value"].astype("float32").values
                out.append((name, X, y, stats))
            except Exception as e:
                logger.error("Cloud eval: failed to load %s: %s", path, e)
        if out:
            break
    if not out:
        logger.warning("Cloud eval: no test data found — skipping evaluation.")
    return out


def _predict(model, X, y):
    ds = timeseries_dataset_from_array(
        data=X, targets=y, sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE, shuffle=False,
    )
    yt, yp = [], []
    for xb, yb in ds:
        yp.append(model.predict(xb, verbose=0).flatten())
        yt.append(yb.numpy())
    return np.concatenate(yt), np.concatenate(yp)


def _evaluate(model, houses: list) -> dict:
    yt_kwh, yp_kwh, yt_n, yp_n, per_house = [], [], [], [], {}
    for name, X, y, stats in houses:
        t, p = _predict(model, X, y)
        tk, pk = norm.inverse_target(t, stats), norm.inverse_target(p, stats)
        yt_kwh.append(tk); yp_kwh.append(pk); yt_n.append(t); yp_n.append(p)
        per_house[name] = {
            "mse": float(mean_squared_error(tk, pk)),
            "r2": float(r2_score(tk, pk)),
        }
    if not yt_kwh:
        return {}
    tk, pk = np.concatenate(yt_kwh), np.concatenate(yp_kwh)
    tn, pn = np.concatenate(yt_n), np.concatenate(yp_n)
    r2s = [v["r2"] for v in per_house.values()]
    return {
        # pooled, σε kWh (ίδια μονάδα με τα παλιά runs)
        "mse": float(mean_squared_error(tk, pk)),
        "mae": float(mean_absolute_error(tk, pk)),
        "r2":  float(r2_score(tk, pk)),
        # pooled στον κανονικοποιημένο χώρο
        "mse_norm": float(mean_squared_error(tn, pn)),
        "r2_norm":  float(r2_score(tn, pn)),
        # ανά σπίτι
        "r2_house_median": float(np.median(r2s)),
        "r2_house_min":    float(np.min(r2s)),
        "n_houses": len(per_house),
        "per_house": per_house,
    }


def evaluate_global_model(round_num: int):
    model_path = CloudResourcesPaths.CLOUD_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        logger.warning("Cloud eval: model not found at %s.", model_path)
        return

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        logger.error("Cloud eval: failed to load model: %s", e)
        return

    houses = _load_houses()
    if not houses:
        return

    metrics = _evaluate(model, houses)
    if not metrics:
        return

    metrics["round"] = round_num
    metrics["timestamp"] = int(time.time())

    logger.info(
        "Cloud eval [Round %d]: MSE=%.4f | MAE=%.4f | R²=%.4f (kWh, pooled) | "
        "R²_norm=%.4f | median house R²=%.4f | min house R²=%.4f | houses=%d",
        round_num, metrics["mse"], metrics["mae"], metrics["r2"],
        metrics["r2_norm"], metrics["r2_house_median"], metrics["r2_house_min"],
        metrics["n_houses"],
    )

    _append_metrics(metrics)


def _append_metrics(metrics: dict):
    try:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

        existing = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                existing = json.load(f)

        existing.append(metrics)

        with open(RESULTS_FILE, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info("Cloud eval: metrics saved to %s", RESULTS_FILE)

    except Exception as e:
        logger.error("Cloud eval: failed to save metrics: %s", e)

        