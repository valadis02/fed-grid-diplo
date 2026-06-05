"""
model_compression_service.py
-----------------------------
Πείραμα 6 — Cloud-side Pruning (manual, χωρίς tfmot)

Ελέγχεται από το env var QUANTIZATION_MODE:
  - none      : κανένα compression (baseline)
  - pruned50  : Magnitude pruning 50%
  - pruned70  : Magnitude pruning 70%
  - pruned90  : Magnitude pruning 90%

Το pruned μοντέλο αποστέλλεται ως Keras (float32, sparse weights)
ώστε τα edges να συνεχίζουν το FL training κανονικά.
"""

import os
import io
import numpy as np
import tensorflow as tf
from shared.logging_config import logger

QUANTIZATION_MODE = os.getenv("QUANTIZATION_MODE", "none").lower()

SPARSITY_MAP = {
    "pruned50": 0.50,
    "pruned70": 0.70,
    "pruned90": 0.90,
}


def compress_model(model_path: str) -> tuple[bytes, str]:
    """
    Εφαρμόζει magnitude pruning στο global μοντέλο.
    Επιστρέφει πάντα Keras format.
    """
    if QUANTIZATION_MODE == "none":
        with open(model_path, "rb") as f:
            return f.read(), "keras"

    if QUANTIZATION_MODE not in SPARSITY_MAP:
        logger.warning(f"[Compression] Άγνωστο QUANTIZATION_MODE='{QUANTIZATION_MODE}' — αποστολή χωρίς compression")
        with open(model_path, "rb") as f:
            return f.read(), "keras"

    sparsity = SPARSITY_MAP[QUANTIZATION_MODE]
    logger.info(f"[Compression] Εφαρμογή magnitude pruning: {QUANTIZATION_MODE} (sparsity={sparsity*100:.0f}%)")

    model = tf.keras.models.load_model(model_path, compile=False)
    _apply_magnitude_pruning(model, sparsity)

    # Αποθήκευση pruned Keras μοντέλου
    pruned_path = model_path.replace(".keras", f"_{QUANTIZATION_MODE}.keras")
    model.save(pruned_path, include_optimizer=False)

    size_kb = os.path.getsize(pruned_path) / 1024
    orig_kb = os.path.getsize(model_path) / 1024
    logger.info(f"[Compression] Pruned Keras: {orig_kb:.1f} KB → {size_kb:.1f} KB ({QUANTIZATION_MODE})")

    with open(pruned_path, "rb") as f:
        return f.read(), "keras"


def _apply_magnitude_pruning(model: tf.keras.Model, sparsity: float):
    """
    Magnitude-based pruning: μηδενίζει τα (sparsity*100)% μικρότερα weights.
    In-place — τροποποιεί το μοντέλο απευθείας.
    """
    total_params = 0
    zeroed_params = 0

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        new_weights = []
        for w in weights:
            if w.ndim < 2:
                # Biases — δεν κάνουμε pruning
                new_weights.append(w)
                continue

            # Υπολογισμός threshold βάσει magnitude
            flat = np.abs(w.flatten())
            threshold = np.percentile(flat, sparsity * 100)
            mask = np.abs(w) >= threshold
            pruned_w = w * mask

            total_params += w.size
            zeroed_params += np.sum(mask == 0)
            new_weights.append(pruned_w)

        layer.set_weights(new_weights)

    actual_sparsity = zeroed_params / total_params if total_params > 0 else 0
    logger.info(f"[Compression] Πραγματική sparsity: {actual_sparsity*100:.1f}% "
                f"({zeroed_params:,}/{total_params:,} params μηδενίστηκαν)")