"""
run_federated_cycle.py
----------------------
Εκτελεί έναν πλήρη federated learning κύκλο:
  1. Αρχικό training → αποστολή στο Fog
  2. Αναμονή για νέο μοντέλο από Fog (command '2')
  3. Αποθήκευση νέου μοντέλου → re-train → αποστολή στο Fog
  4. Επανάληψη για MAX_ROUNDS γύρους

  [Πείραμα 3/4] Καταγράφει χρόνους και RAM ανά γύρο και αποθηκεύει
  αποτελέσματα στο /app/edge/models/experiment_results.json

  [Πείραμα 4] Προσθήκη AES-256-GCM κρυπτογράφησης στην αποστολή
  μοντέλου προς Fog. Μετρά encrypt_time_secs και καταγράφει
  στα αποτελέσματα.
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

# ── AES-256-GCM (Πείραμα 4) ──────────────────────────────────────────────────
AES_ENABLED = os.getenv('AES_ENCRYPTION_KEY') is not None
if AES_ENABLED:
    try:
        from shared.crypto import encrypt_to_b64
        logger.info("Edge: AES-256-GCM encryption ENABLED.")
    except ImportError:
        logger.warning("Edge: shared.crypto not found — encryption DISABLED.")
        AES_ENABLED = False
else:
    logger.info("Edge: AES_ENCRYPTION_KEY not set — encryption DISABLED.")
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

# Αρχείο αποθήκευσης αποτελεσμάτων (Πείραμα 3 & 4)
RESULTS_PATH = os.path.join(
    EdgeResourcesPaths.MODELS_FOLDER_PATH.value,
    "experiment_results.json"
)
# ─────────────────────────────────────────────────────────────────────────────


def validate_input_data():
    """Ελέγχει ότι υπάρχουν τα synthetic δεδομένα."""
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

    # Καθαρισμός παλιών μοντέλων ώστε να ξεκινάμε πάντα από τυχαία βάρη
    for old_model in [model_path, trained_path]:
        if os.path.exists(old_model):
            os.remove(old_model)
            logger.info(f"Removed old model: {old_model}")

    model = create_model('simple_lstm_two_gates')
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
    [Πείραμα 4] Κρυπτογραφεί με AES-256-GCM αν AES_ENCRYPTION_KEY είναι set.

    Επιστρέφει dict με:
      - serialization_time_secs : χρόνος κωδικοποίησης base64 + κατασκευής payload
      - encrypt_time_secs       : χρόνος AES κρυπτογράφησης (0.0 αν disabled)
      - send_time_secs          : χρόνος σύνδεσης + publish
      - total_send_time_secs    : άθροισμα των τριών
      - payload_size_bytes      : μέγεθος τελικού payload σε bytes
      - encrypted               : True/False
    """
    model_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        return {}

    # ── Σειριοποίηση ─────────────────────────────────────────────────────────
    t_ser_start = time.perf_counter()
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    model_b64 = base64.b64encode(model_bytes).decode('utf-8')
    serialization_time_secs = round(time.perf_counter() - t_ser_start, 4)

    # ── AES-256-GCM κρυπτογράφηση (Πείραμα 4) ────────────────────────────────
    encrypt_time_secs = 0.0
    encrypted = False
    if AES_ENABLED:
        try:
            t_enc_start = time.perf_counter()
            model_b64_enc, enc_metrics = encrypt_to_b64(model_bytes)
            encrypt_time_secs = round(time.perf_counter() - t_enc_start, 4)
            model_b64 = model_b64_enc
            encrypted = True
            logger.info(
                f"Edge: model encrypted | "
                f"encrypt={encrypt_time_secs}s | "
                f"plaintext={enc_metrics['plaintext_size_b']/1024:.1f}KB | "
                f"ciphertext={enc_metrics['ciphertext_size_b']/1024:.1f}KB"
            )
        except Exception as e:
            logger.error(f"Edge: encryption failed — sending unencrypted: {e}")
    # ─────────────────────────────────────────────────────────────────────────

    payload = {
        'edge_mac':  EDGE_MAC,
        'edge_name': EDGE_NAME,
        'model':     model_b64,
        'metrics':   metrics,
        'encrypted': encrypted,
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

    total = round(serialization_time_secs + encrypt_time_secs + send_time_secs, 4)

    timing = {
        "serialization_time_secs": serialization_time_secs,
        "encrypt_time_secs":       encrypt_time_secs,
        "send_time_secs":          send_time_secs,
        "total_send_time_secs":    total,
        "payload_size_bytes":      len(payload_bytes),
        "encrypted":               encrypted,
    }
    logger.info(
        f"Model sent to Fog | "
        f"ser={serialization_time_secs}s | "
        f"enc={encrypt_time_secs}s | "
        f"send={send_time_secs}s | "
        f"payload={len(payload_bytes)/1024:.1f} KB | "
        f"encrypted={encrypted}"
    )
    return timing


def wait_for_fog_model(timeout_secs: int) -> dict | None:
    """
    Περιμένει μήνυμα command='2' από το Fog στην ουρά RECV_QUEUE.
    Επιστρέφει το payload ή None αν λήξει το timeout.
    """
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
    """Αποθηκεύει το νέο μοντέλο από το Fog ως base model για το επόμενο training."""
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
    """Αποθηκεύει τα αποτελέσματα όλων των γύρων σε JSON."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            "edge_name":   EDGE_NAME,
            "timestamp":   datetime.now().isoformat(),
            "max_rounds":  MAX_ROUNDS,
            "aes_enabled": AES_ENABLED,
            "rounds":      results,
        }, f, indent=2)
    logger.info(f"Experiment results saved to {RESULTS_PATH}")


# ── Κύριος κύκλος ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info(" FEDERATED LEARNING CYCLE STARTING")
    logger.info(f" Edge: {EDGE_NAME} | Rounds: {MAX_ROUNDS} | Date: {TEST_DATE}")
    logger.info(f" AES Encryption: {'ENABLED' if AES_ENABLED else 'DISABLED'}")
    logger.info("=" * 60)

    try:
        validate_input_data()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Data validation failed: {e}")
        sys.exit(1)

    create_base_model()

    # Συλλογή αποτελεσμάτων ανά γύρο (Πείραμα 3 & 4)
    all_round_results = []

    for round_num in range(1, MAX_ROUNDS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f" ROUND {round_num}/{MAX_ROUNDS}")
        logger.info(f"{'='*60}")

        round_result = {"round": round_num}

        # ── Συνολικός χρόνος γύρου (ξεκινά εδώ) ─────────────────────────────
        t_round_start = time.perf_counter()

        # 1. Train
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

        # Αποθήκευση μετρικών εκπαίδευσης
        round_result["training_time_secs"] = metrics.get("training_time_secs")
        round_result["peak_ram_mb"]        = metrics.get("peak_ram_mb")
        round_result["after_metrics"]      = metrics.get("after_training", {})

        # 2. Send to Fog (με κρυπτογράφηση αν AES_ENABLED)
        try:
            send_timing = send_to_fog(metrics)
            round_result.update(send_timing)
        except Exception as e:
            logger.exception(f"[Round {round_num}] Failed to send model to Fog: {e}")
            break

        # 3. Αναμονή νέου μοντέλου (παρακάμπτεται στον τελευταίο γύρο)
        if round_num < MAX_ROUNDS:
            fog_msg = wait_for_fog_model(WAIT_TIMEOUT)
            if fog_msg is None:
                logger.error(f"[Round {round_num}] No response from Fog. Stopping cycle.")
                break

            # 4. Εφαρμογή νέου global μοντέλου
            if not apply_fog_model(fog_msg):
                logger.warning(f"[Round {round_num}] Could not apply fog model. Stopping.")
                break
        else:
            logger.info(f"[Round {round_num}] Final round complete — cycle finished.")

        # ── Συνολικός χρόνος γύρου (τελειώνει εδώ) ──────────────────────────
        round_result["total_round_time_secs"] = round(
            time.perf_counter() - t_round_start, 2
        )

        logger.info(
            f"[Round {round_num}] SUMMARY | "
            f"train={round_result.get('training_time_secs')}s | "
            f"RAM={round_result.get('peak_ram_mb')}MB | "
            f"enc={round_result.get('encrypt_time_secs', 0.0)}s | "
            f"send={round_result.get('total_send_time_secs')}s | "
            f"total={round_result.get('total_round_time_secs')}s"
        )

        all_round_results.append(round_result)

    # Αποθήκευση αποτελεσμάτων
    if all_round_results:
        save_results(all_round_results)

    logger.info("\n" + "=" * 60)
    logger.info(" FEDERATED CYCLE COMPLETE")
    logger.info("=" * 60)