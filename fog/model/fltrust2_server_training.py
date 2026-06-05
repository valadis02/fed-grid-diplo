"""
FLTrust2 Server Training — Fog-side independent server model.

Εκπαιδεύει τοπικά ένα server model στο fog χρησιμοποιώντας
ένα clean reference dataset (π.χ. house_1 CSV), ανεξάρτητα
από τους edge clients.

Αν αποτύχει οποιοδήποτε βήμα → fallback σε FedAvg.
Δεν υπάρχει fallback στον FLTRUST_SERVER_NODE client.
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from shared.logging_config import logger

# ── Env vars ──────────────────────────────────────────────────────────────────
FLTRUST_ROOT_DATA_PATH = os.getenv('FLTRUST_ROOT_DATA_PATH', '')
FLTRUST_LR             = float(os.getenv('FLTRUST_LR', '0.001'))
FLTRUST_EPOCHS         = int(os.getenv('FLTRUST_EPOCHS', '1'))
SEQUENCE_LENGTH        = int(os.getenv('SEQUENCE_LENGTH', '48'))
BATCH_SIZE             = int(os.getenv('BATCH_SIZE', '32'))
TRAIN_DAYS             = int(os.getenv('TRAIN_DAYS', '20'))

try:
    from shared.utils import required_columns
except ImportError:
    required_columns = None
    logger.warning("FLTrust2 server: could not import required_columns from shared.utils")


# ── Preprocessing (ίδιο pipeline με edge) ────────────────────────────────────

def _extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['year']    = df['datetime'].dt.year
    df['month']   = df['datetime'].dt.month
    df['day']     = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday
    df['month_sin']   = np.sin((df['month'] - 1) * (2. * np.pi / 12))
    df['month_cos']   = np.cos((df['month'] - 1) * (2. * np.pi / 12))
    df['weekday_sin'] = np.sin((df['weekday'] - 1) * (2. * np.pi / 7))
    df['weekday_cos'] = np.cos((df['weekday'] - 1) * (2. * np.pi / 7))
    return df


def _extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['hour']   = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['hour_sin']   = np.sin(df['hour'] * (2. * np.pi / 24))
    df['hour_cos']   = np.cos(df['hour'] * (2. * np.pi / 24))
    df['minute_sin'] = np.sin(df['minute'] * (2. * np.pi / 60))
    df['minute_cos'] = np.cos(df['minute'] * (2. * np.pi / 60))
    return df


def _add_rolling_features(df: pd.DataFrame, windows=(3, 6, 12, 24)) -> pd.DataFrame:
    value = df['value']
    for w in windows:
        df[f'value_rolling_mean_{w}'] = (
            value.rolling(window=w).mean()
            .interpolate(method='linear').ffill().bfill()
        )
    return df


def _add_advanced_features(df: pd.DataFrame, windows=(3, 6, 12, 24)) -> pd.DataFrame:
    value = df['value']
    df['value_diff'] = value.diff().fillna(0)
    for w in windows:
        df[f'value_ewm_{w}']        = value.ewm(span=w, adjust=False).mean()
        df[f'value_volatility_{w}'] = value.rolling(window=w).std().fillna(0)
    baseline_vol = value.rolling(window=6).std().fillna(0)
    df['drift_flag'] = (value.diff().abs() > baseline_vol * 2).astype(int)
    return df


def _add_time_since_last_spike(df: pd.DataFrame) -> pd.DataFrame:
    time_since, counter = [], 0
    for flag in df['drift_flag']:
        counter = 0 if flag == 1 else counter + 1
        time_since.append(counter)
    df['time_since_last_spike'] = time_since
    return df


def _preprocess_server_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'timestamp' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'timestamp': 'datetime'})
    if 'consumption_kwh' in df.columns and 'value' not in df.columns:
        df = df.rename(columns={'consumption_kwh': 'value'})

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.replace('Null', np.nan)
    df = df.dropna(subset=['value'])
    df = df.astype({'value': 'float'})

    df = _extract_date_features(df)
    df = _extract_time_features(df)
    df = _add_rolling_features(df)
    df = _add_advanced_features(df)
    df = _add_time_since_last_spike(df)

    if required_columns is not None:
        cols = [c for c in required_columns if c in df.columns]
        df = df[cols]

    return df


# ── Dataset builder ───────────────────────────────────────────────────────────

def _make_tf_dataset(df: pd.DataFrame, sequence_length: int,
                     batch_size: int) -> tf.data.Dataset:
    if required_columns is not None:
        feature_cols = [c for c in required_columns if c != 'value' and c in df.columns]
    else:
        feature_cols = [c for c in df.columns if c != 'value']

    df = df.dropna(subset=['value'])
    X = df[feature_cols].astype('float32').values
    y = df['value'].astype('float32').values

    if len(df) <= sequence_length:
        raise ValueError(
            f"FLTrust2 server: not enough rows ({len(df)}) "
            f"for sequence_length={sequence_length}"
        )

    return timeseries_dataset_from_array(
        data=X,
        targets=y,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def train_server_model(fog_model_path: str) -> list | None:
    """
    Εκπαιδεύει τοπικά το server model του FLTrust2 στο fog.

    Returns:
        list of numpy arrays (weights), ή None αν αποτύχει.
        Σε περίπτωση None, ο caller κάνει fallback σε FedAvg.
    """
    if not FLTRUST_ROOT_DATA_PATH:
        logger.warning("FLTrust2: FLTRUST_ROOT_DATA_PATH not set — cannot train server model.")
        return None

    if not os.path.exists(FLTRUST_ROOT_DATA_PATH):
        logger.warning(f"FLTrust2: data file not found at '{FLTRUST_ROOT_DATA_PATH}'.")
        return None

    if not os.path.exists(fog_model_path):
        logger.warning(f"FLTrust2: fog model not found at '{fog_model_path}'.")
        return None

    logger.info(
        f"FLTrust2 server training: data='{FLTRUST_ROOT_DATA_PATH}' | "
        f"epochs={FLTRUST_EPOCHS} | lr={FLTRUST_LR} | seq_len={SEQUENCE_LENGTH}"
    )

    try:
        df_raw = pd.read_csv(FLTRUST_ROOT_DATA_PATH, low_memory=False)
        df = _preprocess_server_df(df_raw.copy())

        # Κρατάμε μόνο TRAIN_DAYS μέρες
        if 'datetime' in df_raw.columns:
            df_raw['datetime'] = pd.to_datetime(df_raw['datetime'], errors='coerce')
            min_date = df_raw['datetime'].min()
            max_train_date = min_date + pd.Timedelta(days=TRAIN_DAYS)
            mask = (df_raw['datetime'] <= max_train_date).values
            df = df.iloc[:len(mask)][mask[:len(df)]]

        logger.info(f"FLTrust2 server: preprocessed shape={df.shape}")

        server_model = tf.keras.models.load_model(fog_model_path, compile=False)
        server_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FLTRUST_LR),
            loss=tf.keras.losses.Huber(),
        )

        dataset = _make_tf_dataset(df, SEQUENCE_LENGTH, BATCH_SIZE)
        steps   = max(1, int(np.ceil(max(0, len(df) - SEQUENCE_LENGTH) / BATCH_SIZE)))

        t0 = time.perf_counter()
        server_model.fit(
            dataset.repeat(),
            epochs=FLTRUST_EPOCHS,
            steps_per_epoch=steps,
            verbose=0,
        )
        elapsed = round(time.perf_counter() - t0, 3)

        w_server = server_model.get_weights()
        logger.info(
            f"FLTrust2 server training: done | time={elapsed}s | layers={len(w_server)}"
        )
        return w_server

    except Exception as e:
        logger.error(f"FLTrust2 server training failed: {e}", exc_info=True)
        return None