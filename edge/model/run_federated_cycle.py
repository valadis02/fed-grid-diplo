"""
run_federated_cycle.py
----------------------
Εκτελεί έναν πλήρη federated learning κύκλο:
  1. Αρχικό training → αποστολή στο Fog
  2. Αναμονή για νέο μοντέλο από Fog (command '2')
  3. Αποθήκευση νέου μοντέλου → re-train → αποστολή στο Fog
  4. Επανάληψη για MAX_ROUNDS γύρους
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

# ── Ρυθμίσεις ──────────────────────────────────────────────────────────────
FOG_HOST      = os.getenv('FOG_RABBITMQ_HOST', 'localhost')
FOG_PORT      = int(os.getenv('FOG_RABBITMQ_PORT', 5672))
EDGE_NAME     = os.getenv('EDGE_NAME', 'edge_node_1')
EDGE_MAC      = os.getenv('EDGE_MAC',  '00:00:00:00:00:00')
TEST_DATE     = os.getenv('TRAINING_DATE', '2024-01-01')
MAX_ROUNDS    = int(os.getenv('MAX_ROUNDS', 3))
WAIT_TIMEOUT  = int(os.getenv('WAIT_TIMEOUT_SECS', 120))  # max αναμονή για νέο μοντέλο

SEND_QUEUE    = 'edge_to_fog_models'
RECV_QUEUE    = f'edge_{EDGE_NAME}_messages_queue'
DATA_PATH     = os.path.join(root_path, 'edge', 'data', 'input_data.csv')
# ───────────────────────────────────────────────────────────────────────────


def generate_dummy_data():
    logger.info("Generating dummy data...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    periods    = 10 * 24 * 4
    date_range = pd.date_range(start=TEST_DATE, periods=periods, freq='15min')
    values     = np.sin(np.linspace(0, 10 * np.pi, periods)) * 10 + 20
    values    += np.random.normal(0, 2, periods)
    pd.DataFrame({
        'datetime':              date_range,
        'value':                 values,
        'apparent power (kWh)':  values,
    }).to_csv(DATA_PATH, index=False)
    logger.info(f"Dummy data saved at {DATA_PATH}")


def create_base_model():
    logger.info("Creating base (untrained) model...")
    model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
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


def send_to_fog(metrics: dict):
    """Στέλνει το εκπαιδευμένο μοντέλο στο Fog."""
    model_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        return

    with open(model_path, 'rb') as f:
        model_b64 = base64.b64encode(f.read()).decode('utf-8')

    payload = {
        'edge_mac':  EDGE_MAC,
        'edge_name': EDGE_NAME,
        'model':     model_b64,
        'metrics':   metrics,
    }

    conn = _pika_connect()
    ch   = conn.channel()
    ch.queue_declare(queue=SEND_QUEUE, durable=True)
    ch.basic_publish(
        exchange='',
        routing_key=SEND_QUEUE,
        body=json.dumps(payload).encode('utf-8'),
        properties=pika.BasicProperties(delivery_mode=2, content_type='application/json'),
    )
    conn.close()
    logger.info(f"✅ Round model sent to Fog (queue: {SEND_QUEUE})")


def wait_for_fog_model(timeout_secs: int) -> dict | None:
    """
    Περιμένει μήνυμα command='2' από το Fog στην ουρά RECV_QUEUE.
    Επιστρέφει το payload ή None αν λήξει το timeout.
    """
    logger.info(f"⏳ Waiting for new model from Fog (queue: {RECV_QUEUE}, timeout={timeout_secs}s)...")
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
                logger.info("✅ Received command '2' from Fog — new global model available.")
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
        logger.warning(f"⌛ Timeout: no model received from Fog after {timeout_secs}s.")
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
    logger.info(f"✅ New global model saved at {model_path}")
    return True


# ── Κύριος κύκλος ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info(" FEDERATED LEARNING CYCLE STARTING")
    logger.info(f" Edge: {EDGE_NAME} | Rounds: {MAX_ROUNDS} | Date: {TEST_DATE}")
    logger.info("=" * 60)

    generate_dummy_data()
    create_base_model()

    for round_num in range(1, MAX_ROUNDS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f" ROUND {round_num}/{MAX_ROUNDS}")
        logger.info(f"{'='*60}")

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

        # 2. Send to Fog
        try:
            send_to_fog(metrics)
        except Exception as e:
            logger.exception(f"[Round {round_num}] Failed to send model to Fog: {e}")
            break

        # 3. Wait for global model (skip wait on last round)
        if round_num < MAX_ROUNDS:
            fog_msg = wait_for_fog_model(WAIT_TIMEOUT)
            if fog_msg is None:
                logger.error(f"[Round {round_num}] No response from Fog. Stopping cycle.")
                break

            # 4. Apply new global model
            if not apply_fog_model(fog_msg):
                logger.warning(f"[Round {round_num}] Could not apply fog model. Stopping.")
                break
        else:
            logger.info(f"[Round {round_num}] Final round complete — cycle finished.")

    logger.info("\n" + "=" * 60)
    logger.info(" FEDERATED CYCLE COMPLETE")
    logger.info("=" * 60)