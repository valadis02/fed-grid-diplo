"""
rpi5_ckks_benchmark.py
----------------------
Standalone benchmark για RPi5:
- Μετράει training time
- Μετράει CKKS encryption time (ctx_load + encrypt)
- Δεν απαιτεί σύνδεση με fog/cloud

Αποθηκεύει αποτελέσματα στο /app/edge/models/rpi5_benchmark_results.json
"""
import os
import sys
import time
import json
import numpy as np
from datetime import datetime

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from edge.model.model_architectures import create_model
from edge.model.model_training_service import train_local_edge_model
from edge.communication.edge_resources_paths import EdgeResourcesPaths
from shared.logging_config import logger

MAX_ROUNDS   = int(os.getenv('MAX_ROUNDS', 10))
TEST_DATE    = os.getenv('TRAINING_DATE', '2024-01-01')
DEVICE_LABEL = os.getenv('DEVICE_LABEL', 'RPi5')
RESULTS_PATH = os.path.join(
    EdgeResourcesPaths.MODELS_FOLDER_PATH.value,
    "rpi5_benchmark_results.json"
)

def create_base_model():
    model_path  = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    trained_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for old in [model_path, trained_path]:
        if os.path.exists(old):
            os.remove(old)
    model = create_model(os.getenv('MODEL_ARCHITECTURE', 'base'))
    model.save(model_path)
    logger.info(f"Base model saved at {model_path}")


def benchmark_ckks_encrypt(model_path: str) -> dict:
    """Μετράει CKKS encryption time χωρίς αποστολή."""
    import subprocess
    import gc
    import tensorflow as tf

    script_path  = "/app/shared/ckks_encrypt_subprocess.py"
    weights_path = model_path + ".weights.npz"
    tmp_output   = model_path + ".ckks.bin"

    # Εξαγωγή weights
    model = tf.keras.models.load_model(model_path)
    weights = model.get_weights()
    np.savez(weights_path, *weights)
    del model, weights
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()

    # Subprocess CKKS encryption
    t_start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, script_path, weights_path, tmp_output],
        capture_output=True,
        text=True,
        timeout=300,
    )
    total_time = round(time.perf_counter() - t_start, 4)

    # Καθάρισε temp files
    for f in [weights_path, tmp_output]:
        try:
            os.remove(f)
        except Exception:
            pass

    if result.returncode != 0:
        logger.error(f"CKKS subprocess failed: {result.stderr}")
        return {"error": result.stderr}

    stdout_lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    enc_metrics = json.loads(stdout_lines[-1])
    enc_metrics["total_subprocess_time_s"] = total_time
    return enc_metrics


def save_results(results: list):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            "device":       DEVICE_LABEL,
            "timestamp":    datetime.now().isoformat(),
            "max_rounds":   MAX_ROUNDS,
            "rounds":       results,
        }, f, indent=2)
    logger.info(f"Results saved to {RESULTS_PATH}")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info(f" RPi5 CKKS BENCHMARK")
    logger.info(f" Device: {DEVICE_LABEL} | Rounds: {MAX_ROUNDS}")
    logger.info("=" * 60)

    create_base_model()
    all_results = []

    for round_num in range(1, MAX_ROUNDS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f" ROUND {round_num}/{MAX_ROUNDS}")
        logger.info(f"{'='*60}")

        round_result = {"round": round_num}
        t_round_start = time.perf_counter()

        # 1. Training
        logger.info(f"[Round {round_num}] Training...")
        try:
            metrics = train_local_edge_model(training_date=TEST_DATE)
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            break

        round_result["training_time_secs"] = metrics.get("training_time_secs")
        round_result["peak_ram_mb"]        = metrics.get("peak_ram_mb")
        round_result["after_metrics"]      = metrics.get("after_training", {})

        logger.info(
            f"[Round {round_num}] Training done | "
            f"time={metrics.get('training_time_secs')}s | "
            f"RAM={metrics.get('peak_ram_mb')}MB | "
            f"R²={metrics.get('after_training', {}).get('r2', 0):.4f}"
        )

        # 2. CKKS Encryption benchmark
        model_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
        logger.info(f"[Round {round_num}] CKKS encryption benchmark...")
        enc_metrics = benchmark_ckks_encrypt(model_path)

        round_result["ckks_ctx_load_time_s"]  = enc_metrics.get("ctx_load_time_s")
        round_result["ckks_encrypt_time_s"]   = enc_metrics.get("encrypt_time_s")
        round_result["ckks_total_time_s"]     = enc_metrics.get("total_time_s")
        round_result["ckks_payload_size_kb"]  = enc_metrics.get("payload_size_kb")
        round_result["ckks_overhead_x"]       = enc_metrics.get("overhead_x")

        logger.info(
            f"[Round {round_num}] CKKS done | "
            f"ctx_load={enc_metrics.get('ctx_load_time_s')}s | "
            f"encrypt={enc_metrics.get('encrypt_time_s')}s | "
            f"total={enc_metrics.get('total_time_s')}s | "
            f"payload={enc_metrics.get('payload_size_kb')}KB | "
            f"overhead={enc_metrics.get('overhead_x')}x"
        )

        round_result["total_round_time_secs"] = round(
            time.perf_counter() - t_round_start, 2
        )

        logger.info(
            f"[Round {round_num}] SUMMARY | "
            f"train={round_result['training_time_secs']}s | "
            f"ckks={round_result['ckks_total_time_s']}s | "
            f"total={round_result['total_round_time_secs']}s"
        )

        all_results.append(round_result)

    if all_results:
        save_results(all_results)

    logger.info("\n" + "=" * 60)
    logger.info(" BENCHMARK COMPLETE")
    logger.info("=" * 60)