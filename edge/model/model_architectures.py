import tensorflow as tf
from shared.utils import required_columns
from shared.logging_config import logger

# ---------------------------------------------------------------
# Νικήτρια αρχιτεκτονική από το Πείραμα 7 (scaling experiment):
# Conv1D(16,k=3) + LSTM(32,64) + Dense(32) — ~34K παράμετροι
# Βέλτιστο trade-off ακρίβειας / client drift / παραμέτρων
# για FL σε non-IID δεδομένα κατανάλωσης ενέργειας.
# ---------------------------------------------------------------

# String labels που αντιστοιχούν σε αυτό το μοντέλο
_KNOWN_LABELS = {
    'conv1d_lstm_small',
    'simple_lstm_two_gates',  # backwards compatibility
    'base',                   # default fallback
    'conv1d_lstm',
}


def _num_features() -> int:
    return len(required_columns) - 1


def _compile(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model


def create_model(model_label_or_seq_len=144,
                 mask_value: int = -1) -> tf.keras.Model:
    """
    Επιστρέφει το compiled Conv1D-LSTM-Small μοντέλο.

    Δέχεται είτε:
      - string label (π.χ. 'base', 'conv1d_lstm_small') → αγνοείται, χτίζει πάντα το ίδιο μοντέλο
      - int sequence_length (π.χ. 144) → χρησιμοποιείται ως sequence length

    ~34K παράμετροι: Conv1D(16,k=3) + LSTM(32,64) + Dense(32).
    """
    # Αν περαστεί string label, χρησιμοποίησε default sequence_length=144
    if isinstance(model_label_or_seq_len, str):
        if model_label_or_seq_len not in _KNOWN_LABELS:
            logger.warning(
                f"[model_architectures] Unknown label '{model_label_or_seq_len}' "
                f"— using default conv1d_lstm_small with sequence_length=144"
            )
        sequence_length = 144
    else:
        sequence_length = int(model_label_or_seq_len)

    inputs = tf.keras.layers.Input(
        shape=(sequence_length, _num_features()), dtype=tf.float32
    )
    x = tf.keras.layers.Conv1D(
        16, kernel_size=3, activation='relu', padding='same'
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Masking(mask_value=mask_value)(x)
    x = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64, activation='tanh')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs, name='conv1d_lstm_small')
    logger.info(
        f"[conv1d_lstm_small] δημιουργήθηκε | "
        f"params: {model.count_params():,} | "
        f"input: ({sequence_length}, {_num_features()})"
    )
    return _compile(model)


# Backwards compatibility
def simple_lstm_model(sequence_length: int = 144,
                      mask_value: int = -1) -> tf.keras.Model:
    """Alias για create_model() — διατηρείται για συμβατότητα."""
    return create_model(sequence_length, mask_value)