"""
cloud_evaluation_service.py
============================
Αξιολογεί το global model μετά από κάθε FL aggregation round.
Χρησιμοποιεί ένα κοινό test set από όλα τα σπίτια για σύγκριση
με την κεντρικοποιημένη εκπαίδευση.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from cloud.communication.cloud_resources_paths import CloudResourcesPaths
from shared.logging_config import logger

# ── Configuration ─────────────────────────────────────────────
SEQUENCE_LENGTH = 144
BATCH_SIZE      = 32
WINDOWS         = [3, 6, 12, 24]
# Ημέρες που χρησιμοποιούνται ως test set (τελευταίες 20 μέρες)
TEST_DAYS_START = 70
TEST_DAYS_END   = 90

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
    """Ίδιο preprocessing με edge/model/data_preprocessing.py"""
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


def _load_test_data() -> pd.DataFrame:
    """
    Φορτώνει το test set από όλα τα σπίτια.
    Ψάχνει στο /app/data/ (Docker mount) ή στο ./data/ (local).
    """
    points_per_day = 144
    start_idx = TEST_DAYS_START * points_per_day
    end_idx   = TEST_DAYS_END   * points_per_day

    dfs = []
    for base in ["/app/data", "./data"]:
        if not os.path.isdir(base):
            continue
        for house_id in range(1, 6):
            path = os.path.join(base, f"house_{house_id}", "input_data.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df = _preprocess(df)
            dfs.append(df.iloc[start_idx:end_idx])

        if dfs:
            break

    if not dfs:
        logger.warning("Cloud eval: no test data found — skipping evaluation.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def _evaluate(model, df: pd.DataFrame) -> dict:
    """Αξιολογεί το μοντέλο στο test set."""
    df = df.dropna(subset=FEATURE_COLS + ["value"])
    if len(df) <= SEQUENCE_LENGTH:
        logger.warning("Cloud eval: not enough test rows (%d).", len(df))
        return {}

    X = df[FEATURE_COLS].astype("float32").values
    y = df["value"].astype("float32").values

    ds = timeseries_dataset_from_array(
        data=X, targets=y,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    steps = max(1, int(np.ceil((len(df) - SEQUENCE_LENGTH) / BATCH_SIZE)))

    y_true, y_pred = [], []
    for X_batch, y_batch in ds.take(steps):
        preds = model.predict(X_batch, verbose=0)
        y_true.append(y_batch.numpy())
        y_pred.append(preds.flatten())

    if not y_true:
        return {}

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2":  float(r2_score(y_true, y_pred)),
    }


def evaluate_global_model(round_num: int):
    """
    Καλείται μετά από κάθε aggregation round.
    Φορτώνει το cloud model, αξιολογεί στο test set και αποθηκεύει τα metrics.
    """
    model_path = CloudResourcesPaths.CLOUD_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        logger.warning("Cloud eval: model not found at %s.", model_path)
        return

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        logger.error("Cloud eval: failed to load model: %s", e)
        return

    test_df = _load_test_data()
    if test_df.empty:
        return

    metrics = _evaluate(model, test_df)
    if not metrics:
        return

    metrics["round"] = round_num
    metrics["timestamp"] = int(time.time())

    logger.info(
        "Cloud eval [Round %d]: MSE=%.4f | MAE=%.4f | R²=%.4f",
        round_num, metrics["mse"], metrics["mae"], metrics["r2"]
    )

    # Αποθήκευση — append στο JSON αρχείο
    _append_metrics(metrics)


def _append_metrics(metrics: dict):
    """Προσθέτει τα metrics στο αρχείο αποτελεσμάτων."""
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