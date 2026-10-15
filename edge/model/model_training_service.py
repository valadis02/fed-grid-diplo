import os
import time
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from edge.model.data_preprocessing import preprocess_data
from edge.model.data_selection import filter_data_by_interval_date
from edge.communication.edge_resources_paths import EdgeResourcesPaths
from shared.logging_config import logger
from shared.utils import required_columns
from shared import normalization as norm
from shared import forecasting as fc

# ── FedProx configuration ────────────────────────────────────────────────────
FL_ALGORITHM = os.getenv('FL_ALGORITHM', 'fedavg').lower()
FEDPROX_MU   = float(os.getenv('FEDPROX_MU', '0.1'))

logger.info(f"FL Algorithm: {FL_ALGORITHM.upper()}"
            + (f" | mu={FEDPROX_MU}" if FL_ALGORITHM == 'fedprox' else ""))
# ─────────────────────────────────────────────────────────────────────────────


# =========================
# Metrics (defensive)
# =========================
def compute_metrics(y_true, y_pred):
    if (
        y_true is None
        or y_pred is None
        or len(y_true) == 0
        or len(y_pred) == 0
        or np.isnan(y_true).any()
        or np.isnan(y_pred).any()
    ):
        logger.warning("Invalid values detected in metrics. Skipping evaluation.")
        return {}

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "logcosh": float(np.mean(np.log(np.cosh(y_pred - y_true)))),
        "huber": float(tf.keras.losses.Huber()(y_true, y_pred).numpy()),
        "msle": float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)),
    }


def compute_metrics_pair(y_true_n, y_pred_n, norm_stats):
    """Metrics σε kWh (για αναφορά/σύγκριση) + στον κανονικοποιημένο χώρο."""
    if len(y_true_n) == 0:
        return {}, {}
    y_true_kwh = norm.inverse_target(y_true_n, norm_stats)
    y_pred_kwh = norm.inverse_target(y_pred_n, norm_stats)
    kwh = compute_metrics(y_true_kwh, y_pred_kwh)
    if norm_stats is None:
        return kwh, {}
    normed = {
        "mse": float(mean_squared_error(y_true_n, y_pred_n)),
        "mae": float(mean_absolute_error(y_true_n, y_pred_n)),
        "r2": float(r2_score(y_true_n, y_pred_n)),
    }
    return kwh, normed


# =========================
# FedProx custom loss
# =========================
def make_fedprox_loss(global_weights, mu):
    global_weights_tensors = [
        tf.constant(w, dtype=tf.float32) for w in global_weights
    ]

    @tf.function
    def fedprox_loss(y_true, y_pred):
        task_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return task_loss

    return fedprox_loss, global_weights_tensors


class FedProxModel(tf.keras.Model):
    def __init__(self, base_model, global_weights, mu):
        super().__init__()
        self.base_model = base_model
        self.mu = mu
        self.global_weights_vars = [
            tf.Variable(w.astype(np.float32), trainable=False, name=f"global_w_{i}")
            for i, w in enumerate(global_weights)
        ]

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            task_loss = tf.reduce_mean(tf.square(
                tf.cast(y, tf.float32) - tf.squeeze(y_pred)
            ))

            prox_term = tf.constant(0.0)
            trainable_weights = self.base_model.trainable_variables
            for w, w_global in zip(trainable_weights, self.global_weights_vars):
                prox_term += tf.reduce_sum(tf.square(w - w_global))
            prox_term = (self.mu / 2.0) * prox_term

            total_loss = task_loss + prox_term

        gradients = tape.gradient(total_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        return {
            "loss": total_loss,
            "task_loss": task_loss,
            "prox_term": prox_term,
        }

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        task_loss = tf.reduce_mean(tf.square(
            tf.cast(y, tf.float32) - tf.squeeze(y_pred)
        ))
        return {"loss": task_loss}


# =========================
# RAM Monitor
# =========================
class PeakRamMonitor:
    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._peak_mb = 0.0
        self._stop_event = threading.Event()
        self._process = psutil.Process(os.getpid())
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                ram_mb = self._process.memory_info().rss / (1024 ** 2)
                if ram_mb > self._peak_mb:
                    self._peak_mb = ram_mb
            except psutil.NoSuchProcess:
                break
            time.sleep(self._interval)

    def start(self):
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        self._thread.join()
        return round(self._peak_mb, 2)


# =========================
# Streaming CSV generator
# =========================
def data_generator(file_path, feature_columns, target_column, sequence_length):
    df = pd.read_csv(file_path)

    missing = [c for c in feature_columns + [target_column] if c not in df.columns]
    if missing:
        logger.error(f"Missing columns in {file_path}: {missing}")
        return

    df = df.dropna(subset=[target_column])

    if len(df) <= sequence_length:
        logger.warning(
            f"Not enough rows ({len(df)}) for sequence_length={sequence_length} in {file_path}"
        )
        return

    X = df[feature_columns].astype("float32").values
    y = df[target_column].astype("float32").values

    dataset = timeseries_dataset_from_array(
        data=X,
        targets=y,
        sequence_length=sequence_length,
        batch_size=32,
        shuffle=False,
    )

    for batch in dataset:
        yield batch


# =========================
# Main training function
# =========================
def train_local_edge_model(
    training_date: str,
    sequence_length: int = int(os.getenv('SEQUENCE_LENGTH', 48)),
    batch_size: int = int(os.getenv('BATCH_SIZE', 32)),
):
    # -------- Hyperparameters from env --------
    train_days = int(os.getenv('TRAIN_DAYS', 2))
    eval_days  = int(os.getenv('EVAL_DAYS', 2))

    if fc.dayahead():
        return _train_dayahead(training_date, train_days, eval_days, batch_size)

    # -------- Dates --------
    start_dt = pd.to_datetime(training_date, errors="coerce")
    if pd.isna(start_dt):
        raise ValueError(f"Invalid training_date: {training_date}")

    training_day1   = start_dt.strftime("%Y-%m-%d")
    training_day2   = (start_dt + pd.Timedelta(days=train_days)).strftime("%Y-%m-%d")
    evaluation_day1 = (start_dt + pd.Timedelta(days=train_days + 1)).strftime("%Y-%m-%d")
    evaluation_day2 = (start_dt + pd.Timedelta(days=train_days + 1 + eval_days)).strftime("%Y-%m-%d")

    logger.info(
        f"Training days {training_day1} → {training_day2} ({train_days} days), "
        f"Evaluation days {evaluation_day1} → {evaluation_day2} ({eval_days} days)"
    )

    training_data_path   = EdgeResourcesPaths.TRAINING_DAYS_DATA_PATH.value
    evaluation_data_path = EdgeResourcesPaths.EVALUATION_DAYS_DATA_PATH.value

    # -------- Training data --------
    filter_data_by_interval_date(
        EdgeResourcesPaths.INPUT_DATA_PATH.value,
        "timestamp",
        training_day1,
        training_day2,
        training_data_path,
    )

    preprocess_data(training_data_path, "timestamp", "consumption_kwh")
    train_df = pd.read_csv(training_data_path)
    logger.info(f"Training data shape: {train_df.shape}")

    # -------- Evaluation data --------
    filter_data_by_interval_date(
        EdgeResourcesPaths.INPUT_DATA_PATH.value,
        "timestamp",
        evaluation_day1,
        evaluation_day2,
        evaluation_data_path,
    )

    fallback_eval = False
    if not os.path.exists(evaluation_data_path) or os.path.getsize(evaluation_data_path) == 0:
        logger.warning("Evaluation data empty. Falling back to training data.")
        evaluation_data_path = training_data_path
        fallback_eval = True

    if not fallback_eval:
        preprocess_data(evaluation_data_path, "timestamp", "consumption_kwh")

    # -------- Per-house normalization (stats ΜΟΝΟ από training split) --------
    norm_stats = None
    if norm.enabled():
        norm_stats = norm.fit_stats(train_df)
        norm.transform_csv(training_data_path, norm_stats)
        if not fallback_eval:
            norm.transform_csv(evaluation_data_path, norm_stats)
        norm.save_stats(
            norm_stats,
            os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH.value, "norm_stats.json"),
        )
        train_df = pd.read_csv(training_data_path)
        logger.info(
            f"Normalization: per-house | mu={norm_stats['mu']:.4f} kWh | "
            f"sd={norm_stats['sd']:.4f} kWh"
        )
    else:
        logger.info("Normalization: disabled (NORMALIZATION=none)")

    eval_df = pd.read_csv(evaluation_data_path)

    # -------- Features --------
    feature_columns = [c for c in required_columns if c != "value"]
    num_features = len(feature_columns)

    for col in feature_columns + ["value"]:
        if col not in train_df.columns:
            raise ValueError(f"Column '{col}' missing from training data after preprocessing.")

    # -------- Dataset factory --------
    def make_dataset(csv_path):
        return tf.data.Dataset.from_generator(
            lambda p=csv_path: data_generator(p, feature_columns, "value", sequence_length),
            output_signature=(
                tf.TensorSpec(shape=(None, sequence_length, num_features), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
        )

    def calc_steps(n_rows):
        sequences = max(0, n_rows - sequence_length)
        return max(1, int(np.ceil(sequences / batch_size)))

    train_steps = calc_steps(len(train_df))
    eval_steps  = calc_steps(len(eval_df))

    # -------- Load model --------
    model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model not found: {model_path}")

    base_model = tf.keras.models.load_model(model_path, compile=False)

    # -------- Metrics BEFORE training --------
    logger.info("Computing metrics before training...")
    y_true_before, y_pred_before = [], []
    for X_batch, y_batch in make_dataset(evaluation_data_path).take(eval_steps):
        preds = base_model.predict(X_batch, verbose=0)
        y_true_before.append(y_batch.numpy())
        y_pred_before.append(preds.flatten())

    before_metrics, before_norm = compute_metrics_pair(
        np.concatenate(y_true_before) if y_true_before else np.array([]),
        np.concatenate(y_pred_before) if y_pred_before else np.array([]),
        norm_stats,
    )
    logger.info(f"Metrics before training (kWh): {before_metrics}")
    if before_norm:
        logger.info(f"Metrics before training (normalized): {before_norm}")

    # -------- Configure model (FedAvg ή FedProx) --------
    if FL_ALGORITHM == 'fedprox':
        logger.info(f"Using FedProx (mu={FEDPROX_MU}) — adding proximal regularization")
        global_weights = [w.numpy() for w in base_model.trainable_variables]
        model = FedProxModel(base_model, global_weights, mu=FEDPROX_MU)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001))
    else:
        logger.info("Using FedAvg (standard training)")
        model = base_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.Huber(),
        )

    # -------- Training --------
    logger.info(f"Starting local edge training ({FL_ALGORITHM.upper()})...")
    ram_monitor = PeakRamMonitor(interval=0.5)
    ram_monitor.start()
    t_train_start = time.perf_counter()

    model.fit(
        make_dataset(training_data_path).repeat(),
        epochs=int(os.getenv("EPOCHS", 40)),
        steps_per_epoch=train_steps,
        validation_data=make_dataset(evaluation_data_path).repeat(),
        validation_steps=eval_steps,
        verbose=1,
    )

    training_time_secs = round(time.perf_counter() - t_train_start, 2)
    peak_ram_mb = ram_monitor.stop()

    logger.info(f"Training time: {training_time_secs}s | Peak RAM: {peak_ram_mb} MB")

    # -------- Metrics AFTER training --------
    inference_model = base_model if FL_ALGORITHM == 'fedprox' else model

    logger.info("Computing metrics after training...")
    y_true_after, y_pred_after = [], []
    for X_batch, y_batch in make_dataset(evaluation_data_path).take(eval_steps):
        preds = inference_model.predict(X_batch, verbose=0)
        y_true_after.append(y_batch.numpy())
        y_pred_after.append(preds.flatten())

    after_metrics, after_norm = compute_metrics_pair(
        np.concatenate(y_true_after) if y_true_after else np.array([]),
        np.concatenate(y_pred_after) if y_pred_after else np.array([]),
        norm_stats,
    )
    logger.info(f"Metrics after training ({FL_ALGORITHM.upper()}, kWh): {after_metrics}")
    if after_norm:
        logger.info(f"Metrics after training ({FL_ALGORITHM.upper()}, normalized): {after_norm}")

    # -------- Save model --------
    Path(EdgeResourcesPaths.MODELS_FOLDER_PATH.value).mkdir(parents=True, exist_ok=True)
    inference_model.save(
        EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value,
        include_optimizer=False,
    )

    logger.info(f"Local edge training ({FL_ALGORITHM.upper()}) completed successfully.")

    return {
        "fl_algorithm": FL_ALGORITHM,
        "fedprox_mu": FEDPROX_MU if FL_ALGORITHM == 'fedprox' else None,
        "before_training": before_metrics,
        "after_training": after_metrics,
        "before_training_norm": before_norm,
        "after_training_norm": after_norm,
        "norm_stats": norm_stats,
        "training_time_secs": training_time_secs,
        "peak_ram_mb": peak_ram_mb,
    }


# =========================
# Day-ahead (FORECAST_MODE=dayahead)
# =========================
def _train_dayahead(training_date: str, train_days: int, eval_days: int, batch_size: int):
    """
    Εκπαίδευση day-ahead multi-output μοντέλου (βλ. shared/forecasting.py).
      train: [T0, T1), στόχοι από T0 + 14 μέρες (ιστορικό)
      eval : στόχοι [T1, T1 + EVAL_DAYS), πρόβλεψη μία φορά τη μέρα στις 00:00
    Μετρικές σε kWh στο eval του σπιτιού (πριν/μετά την τοπική εκπαίδευση).
    """
    import gc
    tf.keras.backend.clear_session(); gc.collect()      # κάθε γύρος φορτώνει νέο μοντέλο — όχι συσσώρευση μνήμης
    t0, t1, e1 = fc.windows(training_date, train_days, eval_days)
    logger.info(f"[dayahead] train data {t0.date()} → {t1.date()} | eval targets {t1.date()} → {e1.date()}")

    src = EdgeResourcesPaths.INPUT_DATA_PATH.value
    tr = fc.load_series(src, t0, t1)
    ev = fc.load_series(src, t1 - pd.Timedelta(days=fc.HISTORY_DAYS), e1)
    if len(tr) <= fc.HISTORY_DAYS * 48 + fc.HORIZON:
        raise ValueError(f"[dayahead] not enough training data ({len(tr)} rows) in {t0.date()}–{t1.date()}")

    norm_stats = fc.fit_stats(tr) if norm.enabled() else None
    if norm_stats is not None:
        norm.save_stats(norm_stats, os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH.value, "norm_stats.json"))
        logger.info(f"Normalization: per-house | mu={norm_stats['mu']:.4f} kWh | sd={norm_stats['sd']:.4f} kWh")

    A_tr = fc.build_arrays(tr, norm_stats)
    A_ev = fc.build_arrays(ev, norm_stats, target_start=t1, target_end=e1, issue_at_midnight=True)
    logger.info(f"[dayahead] train samples={A_tr['n']} | eval days={A_ev.get('n', 0)}")

    model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model not found: {model_path}")
    base_model = tf.keras.models.load_model(model_path, compile=False)

    def _eval(m):
        if A_ev.get("n", 0) == 0:
            return {}
        return fc.metrics(A_ev["y_kwh"], fc.predict_kwh(m, A_ev))

    before_metrics = _eval(base_model)
    logger.info(f"Metrics before training (kWh, day-ahead): {before_metrics}")

    if FL_ALGORITHM == 'fedprox':
        logger.info(f"Using FedProx (mu={FEDPROX_MU})")
        global_weights = [w.numpy() for w in base_model.trainable_variables]
        model = FedProxModel(base_model, global_weights, mu=FEDPROX_MU)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001))
    else:
        model = base_model
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')

    epochs = int(os.getenv("EPOCHS", 40))
    logger.info(f"Starting local day-ahead training ({FL_ALGORITHM.upper()}) | epochs={epochs} | batch={batch_size}")
    ram_monitor = PeakRamMonitor(interval=0.5)
    ram_monitor.start()
    t_train_start = time.perf_counter()
    model.fit([A_tr["Xh"], A_tr["Xf"]], A_tr["y"], epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)
    training_time_secs = round(time.perf_counter() - t_train_start, 2)
    peak_ram_mb = ram_monitor.stop()
    logger.info(f"Training time: {training_time_secs}s | Peak RAM: {peak_ram_mb} MB")

    inference_model = base_model if FL_ALGORITHM == 'fedprox' else model
    after_metrics = _eval(inference_model)
    logger.info(f"Metrics after training (kWh, day-ahead): {after_metrics}")

    Path(EdgeResourcesPaths.MODELS_FOLDER_PATH.value).mkdir(parents=True, exist_ok=True)
    # Αποθήκευση ΧΩΡΙΣ optimizer state (το Keras 3 αγνοεί include_optimizer): ~0,6 MB αντί ~1,8 MB προς το fog
    clean = tf.keras.Model.from_config(inference_model.get_config())     # ασυμπίεστο, χωρίς compile/optimizer
    clean.set_weights(inference_model.get_weights())
    clean.save(EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value)
    logger.info("Local edge day-ahead training completed successfully.")

    return {
        "fl_algorithm": FL_ALGORITHM,
        "fedprox_mu": FEDPROX_MU if FL_ALGORITHM == 'fedprox' else None,
        "forecast_mode": "dayahead",
        "before_training": before_metrics,
        "after_training": after_metrics,
        "before_training_norm": {},
        "after_training_norm": {},
        "norm_stats": norm_stats,
        "training_time_secs": training_time_secs,
        "peak_ram_mb": peak_ram_mb,
    }

