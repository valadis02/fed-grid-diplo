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

# ── FedProx configuration ────────────────────────────────────────────────────
# Διαβάζεται από env variable: FL_ALGORITHM=fedprox ή fedavg (default)
FL_ALGORITHM = os.getenv('FL_ALGORITHM', 'fedavg').lower()
# Proximal term μ (mu) — τυπικές τιμές: 0.01, 0.1, 1.0
# Μεγαλύτερο μ = πιο κοντά στο global model, λιγότερο client drift
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


# =========================
# FedProx custom loss
# =========================
def make_fedprox_loss(global_weights, mu):
    """
    Δημιουργεί custom loss που προσθέτει proximal term:
        L_fedprox = L_task + (mu/2) * ||w - w_global||²

    Αυτό αποτρέπει το client drift αναγκάζοντας τα τοπικά βάρη
    να παραμένουν κοντά στο καθολικό μοντέλο.

    Args:
        global_weights: λίστα από numpy arrays (βάρη global model)
        mu: proximal term coefficient (τυπικά 0.01 - 1.0)
    """
    # Μετατρέπουμε σε tensors μία φορά (εκτός του loop εκπαίδευσης)
    global_weights_tensors = [
        tf.constant(w, dtype=tf.float32) for w in global_weights
    ]

    @tf.function
    def fedprox_loss(y_true, y_pred):
        # Βασικό loss (MSE — ίδιο με το υπάρχον σύστημα)
        task_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return task_loss

    # Η proximal regularization γίνεται μέσω custom training step
    # (επιστρέφουμε το loss function και τα global weights ξεχωριστά)
    return fedprox_loss, global_weights_tensors


class FedProxModel(tf.keras.Model):
    """
    Wrapper γύρω από το υπάρχον Keras model που προσθέτει
    proximal regularization στο train_step.

    Ο proximal term υπολογίζεται ως:
        prox = (mu/2) * Σ_l ||w_l - w_global_l||²
    όπου l τρέχει σε όλα τα trainable layers.
    """
    def __init__(self, base_model, global_weights, mu):
        super().__init__()
        self.base_model = base_model
        self.mu = mu
        # Αποθηκεύουμε τα global weights ως non-trainable variables
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

            # Task loss (MSE)
            task_loss = tf.reduce_mean(tf.square(
                tf.cast(y, tf.float32) - tf.squeeze(y_pred)
            ))

            # Proximal term: (mu/2) * ||w - w_global||²
            prox_term = tf.constant(0.0)
            trainable_weights = self.base_model.trainable_variables
            for w, w_global in zip(trainable_weights, self.global_weights_vars):
                prox_term += tf.reduce_sum(tf.square(w - w_global))
            prox_term = (self.mu / 2.0) * prox_term

            total_loss = task_loss + prox_term

        # Gradient update
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
    """
    Τρέχει σε background thread και καταγράφει το peak RAM (MB)
    της τρέχουσας διεργασίας κατά τη διάρκεια του model.fit().
    """
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
    sequence_length: int = 144,
    batch_size: int = 32,
):
    # -------- Dates --------
    start_dt = pd.to_datetime(training_date, errors="coerce")
    if pd.isna(start_dt):
        raise ValueError(f"Invalid training_date: {training_date}")

    training_day1 = start_dt.strftime("%Y-%m-%d")
    training_day2 = (start_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    evaluation_day1 = (start_dt + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    evaluation_day2 = (start_dt + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    logger.info(
        f"Training days {training_day1} → {training_day2}, "
        f"Evaluation days {evaluation_day1} → {evaluation_day2}"
    )

    training_data_path = EdgeResourcesPaths.TRAINING_DAYS_DATA_PATH.value
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

    before_metrics = compute_metrics(
        np.concatenate(y_true_before) if y_true_before else np.array([]),
        np.concatenate(y_pred_before) if y_pred_before else np.array([]),
    )
    logger.info(f"Metrics before training: {before_metrics}")

    # -------- Configure model (FedAvg ή FedProx) --------
    if FL_ALGORITHM == 'fedprox':
        logger.info(f"Using FedProx (mu={FEDPROX_MU}) — adding proximal regularization")
        # Αποθηκεύουμε τα global weights ΠΡΙΝ την εκπαίδευση
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
    # Για FedProx χρησιμοποιούμε το base_model για predictions
    inference_model = base_model if FL_ALGORITHM == 'fedprox' else model

    logger.info("Computing metrics after training...")
    y_true_after, y_pred_after = [], []
    for X_batch, y_batch in make_dataset(evaluation_data_path).take(eval_steps):
        preds = inference_model.predict(X_batch, verbose=0)
        y_true_after.append(y_batch.numpy())
        y_pred_after.append(preds.flatten())

    after_metrics = compute_metrics(
        np.concatenate(y_true_after) if y_true_after else np.array([]),
        np.concatenate(y_pred_after) if y_pred_after else np.array([]),
    )
    logger.info(f"Metrics after training ({FL_ALGORITHM.upper()}): {after_metrics}")

    # -------- Save model --------
    # Αποθηκεύουμε πάντα το base_model (τα βάρη έχουν ενημερωθεί)
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
        "training_time_secs": training_time_secs,
        "peak_ram_mb": peak_ram_mb,
    }
