import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from edge.model.data_preprocessing import preprocess_data
from edge.model.data_selection import filter_data_by_interval_date
from edge.communication.edge_resources_paths import EdgeResourcesPaths
from shared.logging_config import logger
from shared.utils import required_columns


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
# Streaming CSV generator
# =========================
def data_generator(file_path, feature_columns, target_column, sequence_length):
    """
    Reads the CSV in one shot (it's already filtered/small) and yields batches.
    Chunked reading caused edge cases with small datasets.
    """
    df = pd.read_csv(file_path)

    # Validate columns
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
        "datetime",
        training_day1,
        training_day2,
        training_data_path,
    )

    preprocess_data(training_data_path, "datetime", "apparent power (kWh)")
    train_df = pd.read_csv(training_data_path)
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Training data columns: {train_df.columns.tolist()}")

    # -------- Evaluation data --------
    filter_data_by_interval_date(
        EdgeResourcesPaths.INPUT_DATA_PATH.value,
        "datetime",
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
        preprocess_data(evaluation_data_path, "datetime", "apparent power (kWh)")

    eval_df = pd.read_csv(evaluation_data_path)
    logger.info(f"Evaluation data shape: {eval_df.shape}")

    # -------- Features --------
    # required_columns has 'value' as target — remove it for features
    feature_columns = [c for c in required_columns if c != "value"]
    num_features = len(feature_columns)
    logger.info(f"Using {num_features} input features: {feature_columns}")

    # -------- Validate columns exist --------
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
    eval_steps = calc_steps(len(eval_df))

    logger.info(f"Train steps: {train_steps}, Eval steps: {eval_steps}")

    # -------- Load model --------
    model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model not found: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.Huber(),
    )

    # -------- Metrics BEFORE training --------
    # Φτιάχνουμε fresh dataset — δεν κάνουμε reuse
    logger.info("Computing metrics before training...")
    y_true_before, y_pred_before = [], []
    for X_batch, y_batch in make_dataset(evaluation_data_path).take(eval_steps):
        preds = model.predict(X_batch, verbose=0)
        y_true_before.append(y_batch.numpy())
        y_pred_before.append(preds.flatten())

    before_metrics = compute_metrics(
        np.concatenate(y_true_before) if y_true_before else np.array([]),
        np.concatenate(y_pred_before) if y_pred_before else np.array([]),
    )
    logger.info(f"Metrics before training: {before_metrics}")

    # -------- Training --------
    logger.info("Starting local edge training...")
    model.fit(
        make_dataset(training_data_path).repeat(),
        epochs=40,
        steps_per_epoch=train_steps,
        # Fresh dataset για validation — ΔΕΝ reuse exhausted dataset
        validation_data=make_dataset(evaluation_data_path).repeat(),
        validation_steps=eval_steps,
        verbose=1,
    )

    # -------- Metrics AFTER training --------
    # Και εδώ fresh dataset
    logger.info("Computing metrics after training...")
    y_true_after, y_pred_after = [], []
    for X_batch, y_batch in make_dataset(evaluation_data_path).take(eval_steps):
        preds = model.predict(X_batch, verbose=0)
        y_true_after.append(y_batch.numpy())
        y_pred_after.append(preds.flatten())

    after_metrics = compute_metrics(
        np.concatenate(y_true_after) if y_true_after else np.array([]),
        np.concatenate(y_pred_after) if y_pred_after else np.array([]),
    )
    logger.info(f"Metrics after training: {after_metrics}")

    # -------- Save model --------
    Path(EdgeResourcesPaths.MODELS_FOLDER_PATH.value).mkdir(parents=True, exist_ok=True)
    model.save(
        EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value,
        include_optimizer=False,
    )

    logger.info("Local edge training completed successfully.")

    return {
        "before_training": before_metrics,
        "after_training": after_metrics,
    }
