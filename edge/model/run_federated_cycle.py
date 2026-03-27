"""
run_federated_cycle.py
----------------------
Εκτελεί έναν πλήρη federated learning κύκλο:
  1. Αρχικό training → αποστολή στο Fog
  2. Αναμονή για νέο μοντέλο από Fog (command '2')
  3. Αποθήκευση νέου μοντέλου → re-train → αποστολή στο Fog
  4. Επανάληψη για MAX_ROUNDS γύρους

  [Πείραμα 3/4] Καταγράφει χρόνους και RAM ανά γύρο.
  [Πείραμα 4]   AES-256-GCM κρυπτογράφηση.
  [Πείραμα 8]   CKKS Homomorphic Encryption (poly_mod=16384).
"""
import os
import sys
import base64
import json
import time
import pika
import pandas as pd
import numpy as np
from datetime import datetime

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from edge.communication.edge_resources_paths import EdgeResourcesPaths
from edge.model.model_architectures import create_model
from edge.model.model_training_service import train_local_edge_model
from shared.logging_config import logger

# ── Encryption mode ───────────────────────────────────────────────────────────
ENCRYPTION_MODE = os.getenv('ENCRYPTION_MODE', 'none').lower()

AES_ENABLED  = False
CKKS_ENABLED = False

if ENCRYPTION_MODE == 'ckks':
    try:
        import tensorflow as tf
        CKKS_ENABLED = True
        logger.info("Edge: CKKS Homomorphic Encryption ENABLED (poly_mod=16384).")
    except ImportError as e:
        logger.warning(f"Edge: CKKS not available — {e}. Falling back to no encryption.")
elif ENCRYPTION_MODE == 'aes' or os.getenv('AES_ENCRYPTION_KEY') is not None:
    try:
        from shared.crypto import encrypt_to_b64
        AES_ENABLED = True
        logger.info("Edge: AES-256-GCM encryption ENABLED.")
    except ImportError:
        logger.warning("Edge: shared.crypto not found — encryption DISABLED.")
else:
    logger.info("Edge: encryption DISABLED (ENCRYPTION_MODE=none).")
# ─────────────────────────────────────────────────────────────────────────────

# ── ρυθμίσεις ────────────────────────────────────────────────────────────────
FOG_HOST     = os.getenv('FOG_RABBITMQ_HOST', 'localhost')
FOG_PORT     = int(os.getenv('FOG_RABBITMQ_PORT', 5672))
EDGE_NAME    = os.getenv('EDGE_NAME', 'edge_node_1')
EDGE_MAC     = os.getenv('EDGE_MAC',  '00:00:00:00:00:00')
TEST_DATE    = os.getenv('TRAINING_DATE', '2024-01-01')
MAX_ROUNDS   = int(os.getenv('MAX_ROUNDS', 3))
WAIT_TIMEOUT = int(os.getenv('WAIT_TIMEOUT_SECS', 120))

SEND_QUEUE   = 'edge_to_fog_models'
RECV_QUEUE   = f'edge_{EDGE_NAME}_messages_queue'

RESULTS_PATH = os.path.join(
    EdgeResourcesPaths.MODELS_FOLDER_PATH.value,
    "experiment_results.json"
)
# ─────────────────────────────────────────────────────────────────────────────


def validate_input_data():
    data_path = EdgeResourcesPaths.INPUT_DATA_PATH.value
    if not os.path.exists(data_path):
        raise FileNotFoundError(...)
    df = pd.read_csv(data_path, nrows=5)
    if 'timestamp' not in df.columns or 'consumption_kwh' not in df.columns:
        raise ValueError(...)
    os.makedirs(os.path.join(os.path.dirname(data_path), "filtered_data"), exist_ok=True)
    logger.info(f"Input data found at {data_path} ...")


def create_base_model():
    logger.info("Creating base (untrained) model...")
    model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    trained_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for old_model in [model_path, trained_path]:
        if os.path.exists(old_model):
            os.remove(old_model)
            logger.info(f"Removed old model: {old_model}")

    model = create_model(os.getenv('MODEL_ARCHITECTURE', 'base'))
    model.save(model_path)
    logger.info(f"Base model saved at {model_path}")


def _pika_connect(retries=10, delay=5):
    for attempt in range(1, retries + 1):
        try:
            conn = pika.BlockingConnection(pika.ConnectionParameters(
                host=FOG_HOST, port=FOG_PORT, heartbeat=60))
            return conn
        except Exception as e:
            logger.warning(f"RabbitMQ connect failed ({attempt}/{retries}): {e}")
            if attempt == retries:
                raise
            time.sleep(delay)


def send_to_fog(metrics: dict) -> dict:
    """
    Στέλνει το εκπαιδευμένο μοντέλο στο Fog.
    Υποστηρίζει CKKS (poly_mod=16384), AES-256-GCM, ή χωρίς κρυπτογράφηση.
    """
    model_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        return {}

    encrypt_time_secs  = 0.0
    ctx_load_time_secs = 0.0
    encrypted          = False
    encryption_mode    = 'none'

    # ── CKKS Homomorphic Encryption (Πείραμα 8, poly_mod=16384) ──────────────
    if CKKS_ENABLED:
        try:
            import subprocess
            import gc

            script_path  = "/app/shared/ckks_encrypt_subprocess.py"
            weights_path = model_path + ".weights.npz"
            tmp_output   = model_path + ".ckks.bin"

            # Εξαγωγή weights σε .npz ΠΡΙΝ καθαρίσουμε το TF session
            model = tf.keras.models.load_model(model_path)
            weights = model.get_weights()
            np.savez(weights_path, *weights)
            del model, weights
            gc.collect()

            # Καθάρισε TF μνήμη
            tf.keras.backend.clear_session()
            gc.collect()

            # Τρέξε subprocess ΧΩΡΙΣ TF — μόνο numpy + tenseal
            # Το subprocess διαβάζει το context από hardcoded path
            t_enc_start = time.perf_counter()
            result = subprocess.run(
                [sys.executable, script_path, weights_path, tmp_output],
                capture_output=True,
                text=True,
                timeout=300,
            )
            encrypt_time_secs = round(time.perf_counter() - t_enc_start, 4)

            # Καθάρισε weights file
            try:
                os.remove(weights_path)
            except Exception:
                pass

            if result.returncode != 0:
                raise RuntimeError(
                    f"CKKS subprocess failed (rc={result.returncode}): {result.stderr}"
                )

            # Διάβασε metrics από stdout — πάρε μόνο την τελευταία γραμμή (JSON)
            stdout_lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
            enc_metrics = json.loads(stdout_lines[-1])
            ctx_load_time_secs = enc_metrics.get("ctx_load_time_s", 0.0)

            # Διάβασε payload από temp file
            with open(tmp_output, "rb") as f:
                payload_bytes_raw = f.read()
            model_b64 = base64.b64encode(payload_bytes_raw).decode("utf-8")

            try:
                os.remove(tmp_output)
            except Exception:
                pass

            encrypted = True
            encryption_mode = 'ckks'
            logger.info(
                f"Edge: CKKS encrypted (subprocess) | "
                f"ctx_load={ctx_load_time_secs}s | "
                f"encrypt={enc_metrics.get('encrypt_time_s')}s | "
                f"total={enc_metrics.get('total_time_s')}s | "
                f"payload={enc_metrics.get('payload_size_kb'):.1f}KB | "
                f"overhead={enc_metrics.get('overhead_x'):.0f}x"
            )

        except Exception as e:
            logger.error(f"Edge: CKKS encryption failed — sending unencrypted: {e}")
            try:
                os.remove(weights_path)
            except Exception:
                pass
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            encrypted = False
            encryption_mode = 'none'

    # ── AES-256-GCM (Πείραμα 4) ───────────────────────────────────────────────
    elif AES_ENABLED:
        with open(model_path, 'rb') as f:
            model_bytes = f.read()

        try:
            t_enc_start = time.perf_counter()
            model_b64, enc_metrics = encrypt_to_b64(model_bytes)
            encrypt_time_secs = round(time.perf_counter() - t_enc_start, 4)
            encrypted = True
            encryption_mode = 'aes'
            logger.info(
                f"Edge: AES encrypted | encrypt={encrypt_time_secs}s | "
                f"plaintext={enc_metrics['plaintext_size_b']/1024:.1f}KB | "
                f"ciphertext={enc_metrics['ciphertext_size_b']/1024:.1f}KB"
            )
        except Exception as e:
            logger.error(f"Edge: AES encryption failed — sending unencrypted: {e}")
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            encrypted = False
            encryption_mode = 'none'

    # ── Χωρίς κρυπτογράφηση ──────────────────────────────────────────────────
    else:
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')

    # ── Κατασκευή payload ─────────────────────────────────────────────────────
    payload = {
        'edge_mac':        EDGE_MAC,
        'edge_name':       EDGE_NAME,
        'model':           model_b64,
        'metrics':         metrics,
        'encrypted':       encrypted,
        'encryption_mode': encryption_mode,
    }
    payload_bytes = json.dumps(payload).encode('utf-8')

    # ── Αποστολή ─────────────────────────────────────────────────────────────
    t_send_start = time.perf_counter()
    conn = _pika_connect()
    ch   = conn.channel()
    ch.queue_declare(queue=SEND_QUEUE, durable=True)
    ch.basic_publish(
        exchange='',
        routing_key=SEND_QUEUE,
        body=payload_bytes,
        properties=pika.BasicProperties(delivery_mode=2, content_type='application/json'),
    )
    conn.close()
    send_time_secs = round(time.perf_counter() - t_send_start, 4)

    total = round(encrypt_time_secs + ctx_load_time_secs + send_time_secs, 4)

    timing = {
        "encryption_mode":      encryption_mode,
        "ctx_load_time_secs":   ctx_load_time_secs,
        "encrypt_time_secs":    encrypt_time_secs,
        "send_time_secs":       send_time_secs,
        "total_send_time_secs": total,
        "payload_size_bytes":   len(payload_bytes),
        "encrypted":            encrypted,
    }
    logger.info(
        f"Model sent to Fog | mode={encryption_mode} | "
        f"ctx_load={ctx_load_time_secs}s | enc={encrypt_time_secs}s | "
        f"send={send_time_secs}s | payload={len(payload_bytes)/1024:.1f} KB | "
        f"encrypted={encrypted}"
    )
    return timing


def wait_for_fog_model(timeout_secs: int) -> dict | None:
    logger.info(f"Waiting for new model from Fog (queue: {RECV_QUEUE}, timeout={timeout_secs}s)...")
    conn = _pika_connect()
    ch   = conn.channel()
    ch.queue_declare(queue=RECV_QUEUE, durable=True, auto_delete=False)

    received = [None]
    deadline = time.time() + timeout_secs

    while time.time() < deadline:
        method, props, body = ch.basic_get(queue=RECV_QUEUE, auto_ack=False)
        if method is None:
            time.sleep(2)
            continue

        try:
            msg = json.loads(body.decode('utf-8'))
            cmd = str(msg.get('command', ''))
            if cmd == '2':
                logger.info("Received command '2' from Fog — new global model available.")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                received[0] = msg
                break
            else:
                logger.debug(f"Ignoring message with command={cmd!r}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to parse Fog message: {e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)

    conn.close()
    if received[0] is None:
        logger.warning(f"Timeout: no model received from Fog after {timeout_secs}s.")
    return received[0]


def apply_fog_model(msg: dict):
    model_b64 = msg.get('model')
    if not model_b64:
        logger.warning("Fog message has no 'model' field — skipping model update.")
        return False

    model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        f.write(base64.b64decode(model_b64))
    logger.info(f"New global model saved at {model_path}")
    return True


def save_results(results: list):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            "edge_name":       EDGE_NAME,
            "timestamp":       datetime.now().isoformat(),
            "max_rounds":      MAX_ROUNDS,
            "encryption_mode": ENCRYPTION_MODE,
            "aes_enabled":     AES_ENABLED,
            "ckks_enabled":    CKKS_ENABLED,
            "rounds":          results,
        }, f, indent=2)
    logger.info(f"Experiment results saved to {RESULTS_PATH}")


# ── Κύριος κύκλος ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info(" FEDERATED LEARNING CYCLE STARTING")
    logger.info(f" Edge: {EDGE_NAME} | Rounds: {MAX_ROUNDS} | Date: {TEST_DATE}")
    logger.info(f" Encryption mode: {ENCRYPTION_MODE.upper()}")
    logger.info("=" * 60)

    try:
        validate_input_data()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Data validation failed: {e}")
        sys.exit(1)

    create_base_model()

    all_round_results = []

    for round_num in range(1, MAX_ROUNDS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f" ROUND {round_num}/{MAX_ROUNDS}")
        logger.info(f"{'='*60}")

        round_result = {"round": round_num}
        t_round_start = time.perf_counter()

        logger.info(f"[Round {round_num}] Training local model...")
        try:
            metrics = train_local_edge_model(training_date=TEST_DATE)
        except Exception as e:
            logger.exception(f"[Round {round_num}] Training failed: {e}")
            break

        if not metrics:
            logger.error(f"[Round {round_num}] Training returned no metrics. Stopping.")
            break

        logger.info(f"[Round {round_num}] Training complete. Metrics: {metrics.get('after_training', {})}")

        round_result["training_time_secs"] = metrics.get("training_time_secs")
        round_result["peak_ram_mb"]        = metrics.get("peak_ram_mb")
        round_result["after_metrics"]      = metrics.get("after_training", {})

        try:
            send_timing = send_to_fog(metrics)
            round_result.update(send_timing)
        except Exception as e:
            logger.exception(f"[Round {round_num}] Failed to send model to Fog: {e}")
            break

        if round_num < MAX_ROUNDS:
            fog_msg = wait_for_fog_model(WAIT_TIMEOUT)
            if fog_msg is None:
                logger.error(f"[Round {round_num}] No response from Fog. Stopping cycle.")
                break

            if not apply_fog_model(fog_msg):
                logger.warning(f"[Round {round_num}] Could not apply fog model. Stopping.")
                break
        else:
            logger.info(f"[Round {round_num}] Final round complete — cycle finished.")

        round_result["total_round_time_secs"] = round(
            time.perf_counter() - t_round_start, 2
        )

        logger.info(
            f"[Round {round_num}] SUMMARY | "
            f"train={round_result.get('training_time_secs')}s | "
            f"RAM={round_result.get('peak_ram_mb')}MB | "
            f"mode={round_result.get('encryption_mode', 'none')} | "
            f"ctx_load={round_result.get('ctx_load_time_secs', 0.0)}s | "
            f"enc={round_result.get('encrypt_time_secs', 0.0)}s | "
            f"send={round_result.get('send_time_secs', 0.0)}s | "
            f"total={round_result.get('total_round_time_secs')}s"
        )

        all_round_results.append(round_result)

    if all_round_results:
        save_results(all_round_results)

    logger.info("\n" + "=" * 60)
    logger.info(" FEDERATED CYCLE COMPLETE")
    logger.info("=" * 60)