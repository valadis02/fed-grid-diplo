import os

import tensorflow as tf
from shared.utils import required_columns
from shared.logging_config import logger
from shared import forecasting as fc

# ---------------------------------------------------------------
# MLP Baseline αρχιτεκτονική (μειωμένη έκδοση, ~48K params):
# Flatten -> Dense(64) -> Dropout(0.2) -> Dense(32) -> Dropout(0.2) -> Dense(1)
# ~48K παράμετροι — μειωμένη από την αρχική Dense(150)->Dense(75) (~120K)
# για ταχύτερα πειράματα σε RPi hardware. Ενημέρωσε το paper's Section V
# (Model paragraph) ώστε να αναφέρει ~48K αντί για ~119K παραμέτρους.
#
# ΚΟΙΝΗ ΑΡΧΙΚΟΠΟΙΗΣΗ (MODEL_INIT_SEED):
# Όλα τα edges χτίζουν το αρχικό μοντέλο με τους ΙΔΙΟΥΣ seeded initializers,
# ώστε στον γύρο 1 να ξεκινούν από το ίδιο w0 (προϋπόθεση του FedAvg).
# Χωρίς αυτό, ο μέσος όρος ανεξάρτητα αρχικοποιημένων MLP ακυρώνει τα
# hidden features και το δίκτυο καταρρέει σε σταθερή πρόβλεψη (dead ReLUs).
# ---------------------------------------------------------------

MODEL_INIT_SEED = int(os.getenv("MODEL_INIT_SEED", "42"))

_KNOWN_LABELS = {
    'conv1d_lstm_small',
    'simple_lstm_two_gates',
    'base',
    'conv1d_lstm',
    'mlp_baseline',
}

def _num_features() -> int:
    return len(required_columns) - 1

def _compile(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

def _kernel_init(layer_idx: int):
    # Διαφορετικό (αλλά σταθερό) seed ανά στρώση, ίδιο σε όλα τα edges.
    return tf.keras.initializers.GlorotUniform(seed=MODEL_INIT_SEED + layer_idx)

def create_model(model_label_or_seq_len=48,
                 mask_value: int = -1) -> tf.keras.Model:
    """
    Επιστρέφει το compiled MLP baseline μοντέλο.
    Δέχεται είτε:
      - string label → αγνοείται, χτίζει πάντα MLP baseline
      - int sequence_length (π.χ. 48) → χρησιμοποιείται ως sequence length
    ~48K παράμετροι: Flatten + Dense(64) + Dense(32) + Dense(1).
    Βάρη αρχικοποιούνται ντετερμινιστικά από MODEL_INIT_SEED (κοινό w0).
    """
    if isinstance(model_label_or_seq_len, str):
        if model_label_or_seq_len not in _KNOWN_LABELS:
            logger.warning(
                f"[model_architectures] Unknown label '{model_label_or_seq_len}' "
                f"— using mlp_baseline with sequence_length=48"
            )
        sequence_length = 48
    else:
        sequence_length = int(model_label_or_seq_len)

    if fc.dayahead():
        return _create_dayahead_model(sequence_length)

    inputs = tf.keras.layers.Input(
        shape=(sequence_length, _num_features()), dtype=tf.float32
    )
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=_kernel_init(0))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer=_kernel_init(1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=_kernel_init(2))(x)

    model = tf.keras.Model(inputs, outputs, name='mlp_baseline')

    # Fingerprint για επιβεβαίωση ότι όλα τα edges έχουν το ίδιο w0.
    w0_sum = float(sum(float(tf.reduce_sum(w)) for w in model.weights))
    logger.info(
        f"[mlp_baseline] δημιουργήθηκε | "
        f"params: {model.count_params():,} | "
        f"input: ({sequence_length}, {_num_features()}) | "
        f"init_seed={MODEL_INIT_SEED} | w0_checksum={w0_sum:.6f}"
    )
    return _compile(model)

def _create_dayahead_model(sequence_length: int) -> tf.keras.Model:
    """
    Day-ahead multi-output MLP (FORECAST_MODE=dayahead, βλ. shared/forecasting.py):
      [ιστορικό (L x 15) → Flatten] ⊕ [features των 48 μισάωρων-στόχων (336)]
      → Dense(128) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Dense(48)
    Έξοδος: κανονικοποιημένη απόκλιση από το προφίλ 7 ημερών για κάθε μισάωρο της επόμενης μέρας.
    """
    hist = tf.keras.layers.Input(shape=(sequence_length, fc.N_HIST), dtype=tf.float32, name='history')
    fut = tf.keras.layers.Input(shape=(fc.N_FUT,), dtype=tf.float32, name='target_day')
    x = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(hist), fut])
    x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=_kernel_init(0))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=_kernel_init(1))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(fc.HORIZON, kernel_initializer=_kernel_init(2))(x)
    model = tf.keras.Model([hist, fut], outputs, name='mlp_dayahead')

    w0_sum = float(sum(float(tf.reduce_sum(w)) for w in model.weights))
    logger.info(
        f"[mlp_dayahead] δημιουργήθηκε | params: {model.count_params():,} | "
        f"inputs: ({sequence_length}, {fc.N_HIST}) + ({fc.N_FUT},) → {fc.HORIZON} | "
        f"init_seed={MODEL_INIT_SEED} | w0_checksum={w0_sum:.6f}"
    )
    return _compile(model)


# Backwards compatibility
def simple_lstm_model(sequence_length: int = 48,
                      mask_value: int = -1) -> tf.keras.Model:
    """Alias για create_model() — διατηρείται για συμβατότητα."""
    return create_model(sequence_length, mask_value)

