"""
model_quantization_pruning.py
==============================
Πείραμα 6 — Post-Training Quantization (PTQ) & Magnitude-based Pruning

Παράγει 5 TFLite μοντέλα από το εκπαιδευμένο Conv1D-LSTM:
  1. conv1d_lstm_float32.tflite       — Baseline TFLite (χωρίς βελτιστοποίηση)
  2. conv1d_lstm_int8.tflite          — Μόνο PTQ INT8
  3. conv1d_lstm_pruned50_int8.tflite — Pruning 50% sparsity + PTQ INT8
  4. conv1d_lstm_pruned75_int8.tflite — Pruning 75% sparsity + PTQ INT8
  5. conv1d_lstm_pruned90_int8.tflite — Pruning 90% sparsity + PTQ INT8

Μέθοδος pruning:
  Magnitude-based unstructured sparsity μέσω tensorflow_model_optimization (tfmot).
  Μηδενίζει τα weights με τη μικρότερη απόλυτη τιμή (ConstantSparsity schedule),
  ακολουθούμενο από ελαφρύ fine-tuning (5 εποχές) για σταθεροποίηση.
  Η μέθοδος αυτή είναι η πιο τεκμηριωμένη για LSTM/Conv1D σε embedded συσκευές
  και συνδυάζεται απευθείας με PTQ για μέγιστη συμπίεση.

Απαιτήσεις:
    pip install tensorflow-model-optimization

Εκτέλεση (μία φορά, στον PC — πριν τρέξεις experiment6_edge.yml):
    python -m edge.model.model_quantization_pruning
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

# ── Έλεγχος tensorflow-model-optimization ────────────────────────────────────
try:
    import tensorflow_model_optimization as tfmot
except ImportError:
    print("[ERROR] Λείπει το tensorflow-model-optimization.")
    print("        Εγκατάστησέ το με: pip install tensorflow-model-optimization")
    sys.exit(1)

# ── Root path (ώστε να δουλεύει και ως module και ως script) ─────────────────
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from edge.communication.edge_resources_paths import EdgeResourcesPaths
from shared.logging_config import logger
from shared.utils import required_columns

# ── Paths ─────────────────────────────────────────────────────────────────────
# Χρησιμοποιεί το trained model του node_1 ως αντιπροσωπευτικό εκπαιδευμένο μοντέλο
MODEL_PATH = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
DATA_PATH  = EdgeResourcesPaths.INPUT_DATA_PATH.value

# Τα TFLite μοντέλα αποθηκεύονται στον ίδιο φάκελο models/ του edge
TFLITE_DIR = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH.value, "tflite")
os.makedirs(TFLITE_DIR, exist_ok=True)

# ── Hyperparameters (ίδια με τα υπόλοιπα πειράματα) ──────────────────────────
SEQUENCE_LENGTH = 144
BATCH_SIZE      = 32
PRUNING_EPOCHS  = 5    # Fine-tuning εποχές μετά το pruning
SPARSITY_LEVELS = [0.50, 0.75, 0.90]

# ── Feature columns (από shared/utils.py — εξαιρείται το 'value') ─────────────
FEATURE_COLUMNS = [c for c in required_columns if c != 'value']


# =============================================================================
# Preprocessing — αντίστοιχο του data_preprocessing.py
# =============================================================================
def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.rename(columns={'consumption_kwh': 'value'}, inplace=True)

    df['value_diff'] = df['value'].diff().fillna(0)
    for w in [3, 6, 12, 24]:
        df[f'value_rolling_mean_{w}'] = df['value'].rolling(w, min_periods=1).mean()
        df[f'value_volatility_{w}']   = df['value'].rolling(w, min_periods=1).std().fillna(0)
        df[f'value_ewm_{w}']          = df['value'].ewm(span=w, adjust=False).mean()

    rolling_std = df['value'].rolling(12, min_periods=1).std().fillna(0)
    df['drift_flag'] = (df['value_diff'].abs() > 2 * rolling_std).astype(float)

    time_since = np.zeros(len(df))
    last_spike = -1
    for i in range(len(df)):
        if df['drift_flag'].iloc[i] == 1:
            last_spike = i
        time_since[i] = i - last_spike if last_spike >= 0 else i
    df['time_since_last_spike'] = time_since

    return df.dropna().reset_index(drop=True)


def build_sequences(df: pd.DataFrame):
    """Κατασκευάζει (X, y) sequences — ίδια λογική με model_training_service.py."""
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_all = df[feature_cols].astype('float32').values
    y_all = df['value'].astype('float32').values

    X_seqs, y_seqs = [], []
    for i in range(SEQUENCE_LENGTH, len(X_all)):
        X_seqs.append(X_all[i - SEQUENCE_LENGTH:i])
        y_seqs.append(y_all[i])

    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


# =============================================================================
# Representative dataset για INT8 calibration
# =============================================================================
def make_representative_dataset(X_calib: np.ndarray):
    """
    Παράγει representative dataset για τον INT8 calibration του TFLite converter.
    Απαιτείται για full integer quantization των activations.
    """
    def representative_dataset():
        n_samples = min(200, len(X_calib))
        indices = np.random.choice(len(X_calib), n_samples, replace=False)
        for idx in indices:
            yield [X_calib[idx:idx+1]]
    return representative_dataset


# =============================================================================
# TFLite μετατροπές
# =============================================================================
def to_tflite_float32(model: tf.keras.Model, output_path: str) -> float:
    """Baseline TFLite χωρίς quantization — για σύγκριση μεγέθους και latency."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(output_path, 'wb') as f:
        f.write(converter.convert())
    size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"  ✓ {os.path.basename(output_path):<45}  {size_kb:8.1f} KB")
    return size_kb


def to_tflite_int8(model: tf.keras.Model, X_calib: np.ndarray, output_path: str) -> float:
    """
    Full INT8 PTQ: weights ΚΑΙ activations σε 8-bit integers.
    Είσοδος/έξοδος παραμένουν float32 για συμβατότητα με τον υπόλοιπο κώδικα.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(X_calib)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32
    with open(output_path, 'wb') as f:
        f.write(converter.convert())
    size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"  ✓ {os.path.basename(output_path):<45}  {size_kb:8.1f} KB")
    return size_kb


# =============================================================================
# Pruning με tfmot
# =============================================================================
def apply_pruning_and_strip(
    model_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sparsity: float,
) -> tf.keras.Model:
    """
    Εφαρμόζει magnitude-based unstructured pruning με tfmot.

    Διαδικασία:
      1. Φόρτωση φρέσκου αντιγράφου του μοντέλου (αποφυγή in-place αλλαγών)
      2. prune_low_magnitude(): wraps το μοντέλο με pruning masks
      3. Fine-tuning με UpdatePruningStep callback (ενημερώνει τα masks ανά batch)
      4. strip_pruning(): αφαιρεί τα wrappers, τα μηδενισμένα weights παραμένουν

    Γιατί ConstantSparsity:
      Εφαρμόζει αμέσως το target sparsity από το βήμα 0, χωρίς ramp-up.
      Κατάλληλο για μικρό αριθμό fine-tuning εποχών (≤10).

    Args:
        model_path: path του .keras αρχείου (φορτώνεται φρέσκο κάθε φορά)
        X_train:    training sequences για fine-tuning
        y_train:    training targets
        sparsity:   ποσοστό weights που μηδενίζονται (0.50, 0.75, 0.90)

    Returns:
        stripped_model: Keras μοντέλο έτοιμο για TFLite μετατροπή
    """
    logger.info(f"  [Pruning] Φόρτωση μοντέλου για sparsity={sparsity*100:.0f}%...")
    base_model = tf.keras.models.load_model(model_path, compile=False)

    n_steps = max(1, (len(X_train) // BATCH_SIZE) * PRUNING_EPOCHS)

    pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(
        target_sparsity=sparsity,
        begin_step=0,
        end_step=n_steps,
        frequency=100,
    )

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        base_model,
        pruning_schedule=pruning_schedule,
    )
    model_for_pruning.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
    )

    logger.info(f"  [Pruning] Fine-tuning {PRUNING_EPOCHS} εποχές ({n_steps} steps)...")
    model_for_pruning.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=PRUNING_EPOCHS,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        verbose=0,
    )

    stripped_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    # Επαλήθευση πραγματικής sparsity
    total = sum(w.size for layer in stripped_model.layers for w in layer.get_weights())
    zeros = sum(np.sum(w == 0) for layer in stripped_model.layers for w in layer.get_weights())
    actual = zeros / total if total > 0 else 0
    logger.info(f"  [Pruning] Πραγματική sparsity: {actual*100:.1f}% (target: {sparsity*100:.0f}%)")

    tf.keras.backend.clear_session()
    return stripped_model


# =============================================================================
# Main
# =============================================================================
def run_quantization_and_pruning():
    logger.info("=" * 65)
    logger.info("Πείραμα 6 — PTQ & Pruning: Δημιουργία TFLite μοντέλων")
    logger.info("=" * 65)

    # ── Έλεγχος μοντέλου ──────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Μοντέλο δεν βρέθηκε: {MODEL_PATH}")
        logger.error("Εκτέλεσε πρώτα ένα FL cycle για να παραχθεί το trained_local_edge_model.keras")
        sys.exit(1)

    logger.info(f"\n[1/4] Φόρτωση μοντέλου: {MODEL_PATH}")
    base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    original_size_kb = os.path.getsize(MODEL_PATH) / 1024
    logger.info(f"  Μέγεθος .keras: {original_size_kb:.1f} KB | Παράμετροι: {base_model.count_params():,}")

    # ── Φόρτωση δεδομένων ─────────────────────────────────────────────────
    logger.info(f"\n[2/4] Φόρτωση δεδομένων: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        logger.error(f"Δεδομένα δεν βρέθηκαν: {DATA_PATH}")
        sys.exit(1)

    df = preprocess_raw(pd.read_csv(DATA_PATH))
    X, y = build_sequences(df)
    split    = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_calib          = X[split:]
    logger.info(f"  Train: {len(X_train)} sequences | Calibration: {len(X_calib)} sequences")

    # ── Δημιουργία TFLite μοντέλων ────────────────────────────────────────
    logger.info(f"\n[3/4] Δημιουργία TFLite μοντέλων...")
    logger.info(f"\n  {'Αρχείο':<45}  {'Μέγεθος':>8}")
    logger.info(f"  {'-'*55}")

    sizes = {}

    # 1. Float32 baseline
    sizes["float32"] = to_tflite_float32(
        base_model,
        os.path.join(TFLITE_DIR, "conv1d_lstm_float32.tflite")
    )

    # 2. Μόνο PTQ INT8
    sizes["int8"] = to_tflite_int8(
        base_model,
        X_calib,
        os.path.join(TFLITE_DIR, "conv1d_lstm_int8.tflite")
    )

    # 3–5. Pruning (50%, 75%, 90%) + PTQ INT8
    for sparsity in SPARSITY_LEVELS:
        pct = int(sparsity * 100)
        logger.info(f"\n  ── Pruning {pct}% {'─'*38}")
        stripped = apply_pruning_and_strip(MODEL_PATH, X_train, y_train, sparsity)
        sizes[f"pruned{pct}_int8"] = to_tflite_int8(
            stripped,
            X_calib,
            os.path.join(TFLITE_DIR, f"conv1d_lstm_pruned{pct}_int8.tflite")
        )

    # ── Σύνοψη ────────────────────────────────────────────────────────────
    logger.info(f"\n[4/4] Σύνοψη")
    logger.info("=" * 65)
    logger.info(f"  {'Μοντέλο':<35} {'KB':>8} {'Μείωση vs Float32':>18}")
    logger.info(f"  {'-'*63}")

    f32 = sizes.get("float32", 1)
    labels = {
        "float32":       "Float32 TFLite (baseline)",
        "int8":          "INT8 TFLite (μόνο PTQ)",
        "pruned50_int8": "Pruning 50% + PTQ INT8",
        "pruned75_int8": "Pruning 75% + PTQ INT8",
        "pruned90_int8": "Pruning 90% + PTQ INT8",
    }
    for key, label in labels.items():
        if key in sizes:
            reduction = (1 - sizes[key] / f32) * 100 if key != "float32" else 0
            suffix = f"{reduction:+.1f}%" if key != "float32" else "—"
            logger.info(f"  {label:<35} {sizes[key]:>8.1f} {suffix:>18}")

    logger.info(f"\n  Αρχεία: {TFLITE_DIR}")
    logger.info("=" * 65)
    logger.info("Επόμενο βήμα: docker-compose -f experiment6_edge.yml up (στα RPi)")


if __name__ == "__main__":
    run_quantization_and_pruning()