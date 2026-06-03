import tensorflow as tf
from shared.utils import required_columns
from shared.logging_config import logger

# ---------------------------------------------------------------
# MLP Baseline αρχιτεκτονική:
# Flatten -> Dense(150) -> Dropout(0.2) -> Dense(75) -> Dropout(0.2) -> Dense(1)
# ~120K παράμετροι, κατάλληλο για scalability benchmarking.
# ---------------------------------------------------------------

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

def create_model(model_label_or_seq_len=48,
                 mask_value: int = -1) -> tf.keras.Model:
    """
    Επιστρέφει το compiled MLP baseline μοντέλο.
    Δέχεται είτε:
      - string label → αγνοείται, χτίζει πάντα MLP baseline
      - int sequence_length (π.χ. 48) → χρησιμοποιείται ως sequence length
    ~120K παράμετροι: Flatten + Dense(150) + Dense(75) + Dense(1).
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

    inputs = tf.keras.layers.Input(
        shape=(sequence_length, _num_features()), dtype=tf.float32
    )
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(150, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(75, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs, name='mlp_baseline')
    logger.info(
        f"[mlp_baseline] δημιουργήθηκε | "
        f"params: {model.count_params():,} | "
        f"input: ({sequence_length}, {_num_features()})"
    )
    return _compile(model)

# Backwards compatibility
def simple_lstm_model(sequence_length: int = 48,
                      mask_value: int = -1) -> tf.keras.Model:
    """Alias για create_model() — διατηρείται για συμβατότητα."""
    return create_model(sequence_length, mask_value)