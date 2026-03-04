"""
centralized_training_v2.py
==========================
Διορθωμένη κεντρικοποιημένη εκπαίδευση — χωρίς data leakage.

Διορθώσεις από v1:
  1. Split ΠΡΩΤΑ, preprocess ΜΕΤΑ ξεχωριστά (αποφυγή data leakage από rolling features)
  2. Ξεχωριστό test set (τελευταίες 20 μέρες) που δεν αγγίζει το training
  3. Feedback κάθε epoch
  4. Αποθήκευση μετά από κάθε epoch

Splits (από 90 μέρες):
  - Train:      ημέρες 1-60  (60 μέρες)
  - Validation: ημέρες 61-70 (10 μέρες) — για monitoring κατά το training
  - Test:       ημέρες 71-90 (20 μέρες) — για τελική αξιολόγηση
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

# ── Configuration ─────────────────────────────────────────────
HOUSES          = 5
SEQUENCE_LENGTH = 144
BATCH_SIZE      = 32
EPOCHS          = 40
DATA_DIR        = "data"
OUTPUT_FILE     = "results_centralized_v2.json"

POINTS_PER_DAY  = 144
TRAIN_END       = 60 * POINTS_PER_DAY   # ημέρες 1-60
VAL_END         = 70 * POINTS_PER_DAY   # ημέρες 61-70
TEST_END        = 90 * POINTS_PER_DAY   # ημέρες 71-90

WINDOWS = [3, 6, 12, 24]

FEATURE_COLS = [
    "value_diff",
    "value_rolling_mean_3",  "value_volatility_3",  "value_ewm_3",
    "value_rolling_mean_6",  "value_volatility_6",  "value_ewm_6",
    "value_rolling_mean_12", "value_volatility_12", "value_ewm_12",
    "value_rolling_mean_24", "value_volatility_24", "value_ewm_24",
    "drift_flag", "time_since_last_spike",
]


def preprocess_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing σε ένα split — rolling features υπολογίζονται
    ΜΟΝΟ μέσα στο split, όχι σε ολόκληρο το dataframe.
    """
    df = df.copy().reset_index(drop=True)
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

    return df.dropna(subset=FEATURE_COLS + ["value"])


def make_dataset(df, shuffle=False):
    if len(df) <= SEQUENCE_LENGTH:
        return None, 0
    X = df[FEATURE_COLS].astype("float32").values
    y = df["value"].astype("float32").values
    ds = timeseries_dataset_from_array(
        data=X, targets=y,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )
    steps = max(1, int(np.ceil((len(df) - SEQUENCE_LENGTH) / BATCH_SIZE)))
    return ds, steps


def evaluate_model(model, ds, steps):
    y_true, y_pred = [], []
    for X_batch, y_batch in ds.take(steps):
        preds = model.predict(X_batch, verbose=0)
        y_true.append(y_batch.numpy())
        y_pred.append(preds.flatten())
    if not y_true:
        return {}
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    return {
        "mse": float(mean_squared_error(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2":  float(r2_score(yt, yp)),
    }


# ── Load & split ΠΡΩΤΑ, preprocess ΜΕΤΑ ──────────────────────
print("=" * 60)
print(" Centralized Training v2 — No Data Leakage")
print(f" Houses: {HOUSES} | Epochs: {EPOCHS}")
print(" Split: Train=60d | Val=10d | Test=20d")
print("=" * 60)

print("\n[1/4] Loading and splitting data (preprocess per split)...")
train_dfs, val_dfs, test_dfs = [], [], []

for house_id in range(1, HOUSES + 1):
    path = os.path.join(DATA_DIR, f"house_{house_id}", "input_data.csv")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        sys.exit(1)

    raw = pd.read_csv(path)

    # ── Split ΠΡΩΤΑ ──
    raw_train = raw.iloc[:TRAIN_END]
    raw_val   = raw.iloc[TRAIN_END:VAL_END]
    raw_test  = raw.iloc[VAL_END:TEST_END]

    # ── Preprocess ΜΕΤΑ (ξεχωριστά για κάθε split) ──
    train_dfs.append(preprocess_split(raw_train))
    val_dfs.append(preprocess_split(raw_val))
    test_dfs.append(preprocess_split(raw_test))

    print(f"  House {house_id}: train={len(train_dfs[-1]):,} | "
          f"val={len(val_dfs[-1]):,} | test={len(test_dfs[-1]):,}")

train_df = pd.concat(train_dfs, ignore_index=True)
val_df   = pd.concat(val_dfs,   ignore_index=True)
test_df  = pd.concat(test_dfs,  ignore_index=True)

print(f"\n  Combined train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")

# ── Build datasets ────────────────────────────────────────────
print("\n[2/4] Building TF datasets...")
train_ds, train_steps = make_dataset(train_df, shuffle=True)
val_ds,   val_steps   = make_dataset(val_df,   shuffle=False)
test_ds,  test_steps  = make_dataset(test_df,  shuffle=False)
print(f"  Train steps: {train_steps} | Val steps: {val_steps} | Test steps: {test_steps}")

# ── Build model ───────────────────────────────────────────────
print("\n[3/4] Building LSTM model...")
inputs  = tf.keras.Input(shape=(SEQUENCE_LENGTH, len(FEATURE_COLS)))
x       = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x       = tf.keras.layers.LSTM(32)(x)
x       = tf.keras.layers.Dense(16, activation="relu")(x)
outputs = tf.keras.layers.Dense(1)(x)
model   = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.Huber(),
)
model.summary()

# ── Training — feedback κάθε epoch ───────────────────────────
print("\n[4/4] Training...")
epoch_metrics = []
results = {
    "experiment": "centralized_v2",
    "note": "No data leakage — split before preprocess",
    "houses": HOUSES,
    "epochs": EPOCHS,
    "splits": {"train_days": 60, "val_days": 10, "test_days": 20},
    "epoch_metrics": epoch_metrics,
}

# Before training
print("\n--- Before training (on test set) ---")
m = evaluate_model(model, test_ds, test_steps)
print(f"  MSE={m.get('mse',0):.4f} | MAE={m.get('mae',0):.4f} | R²={m.get('r2',0):.4f}")
epoch_metrics.append({"epoch": 0, "split": "test", **m})

for epoch in range(1, EPOCHS + 1):
    # Train 1 epoch
    history = model.fit(
        train_ds.repeat(),
        epochs=1,
        steps_per_epoch=train_steps,
        verbose=0,
    )
    train_loss = history.history["loss"][0]

    # Evaluate on validation set
    val_m = evaluate_model(model, val_ds, val_steps)

    print(f"  Epoch {epoch:3d}/{EPOCHS} | "
          f"train_loss={train_loss:.4f} | "
          f"val_MSE={val_m.get('mse',0):.4f} | "
          f"val_R²={val_m.get('r2',0):.4f}")

    epoch_metrics.append({
        "epoch": epoch,
        "train_loss": float(train_loss),
        "split": "val",
        **val_m,
    })

    # Αποθήκευση μετά από κάθε epoch
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

# ── Final evaluation on TEST set ─────────────────────────────
print("\n--- Final evaluation on TEST set (unseen data) ---")
test_m = evaluate_model(model, test_ds, test_steps)
print(f"  MSE={test_m.get('mse',0):.4f} | MAE={test_m.get('mae',0):.4f} | R²={test_m.get('r2',0):.4f}")

results["final_test_metrics"] = test_m
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print(" Centralized Training v2 Complete!")
print(f" Results saved to {OUTPUT_FILE}")
print("=" * 60)