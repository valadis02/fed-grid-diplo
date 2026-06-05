"""
run_federated_cycle_exp6.py
---------------------------
Πείραμα 6 — Edge FL cycle με TinyML Inference Benchmark (house_5 / RPi5)

Διαφορές από run_federated_cycle.py:
  1. Μετά από κάθε γύρο, αν το μοντέλο που ελήφθη είναι TFLite (.tflite),
     εκτελεί inference benchmark και καταγράφει latency/RAM/R²
  2. Αν το μοντέλο είναι κανονικό .keras (QUANTIZATION_MODE=none),
     δεν κάνει benchmark (δεν έχει νόημα η σύγκριση)
  3. Αποθηκεύει αποτελέσματα στο experiment_results.json με format:
     { experiment, device, quantization_mode, run_id, rounds: [...] }

Env vars:
  QUANTIZATION_MODE : none/ptq/pruned50/pruned75/pruned90
  RUN_ID            : αριθμός τρέχουσας εκτέλεσης (1-5)
  DEVICE_LABEL      : RPi5
  NUM_INFERENCE_RUNS: επαναλήψεις inference benchmark ανά γύρο (default: 3)
"""

import os
import sys
import base64
import json
import time
import threading
import numpy as np
import pandas as pd
import psutil
import pika
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from edge.communication.edge_resources_paths import EdgeResourcesPaths
from edge.model.model_architectures import create_model
from edge.model.model_training_service import train_local_edge_model
from shared.logging_config import logger

# ── AES ───────────────────────────────────────────────────────────────────────
AES_ENABLED = os.getenv('AES_ENCRYPTION_KEY') is not None
if AES_ENABLED:
    try:
        from shared.crypto import encrypt_to_b64
    except ImportError:
        AES_ENABLED = False

# ── Ρυθμίσεις ─────────────────────────────────────────────────────────────────
FOG_HOST           = os.getenv('FOG_RABBITMQ_HOST', 'localhost')
FOG_PORT           = int(os.getenv('FOG_RABBITMQ_PORT', 5672))
EDGE_NAME          = os.getenv('EDGE_NAME', 'edge_node_1')
EDGE_MAC           = os.getenv('EDGE_MAC', '00:00:00:00:00:00')
TEST_DATE          = os.getenv('TRAINING_DATE', '2024-01-01')
MAX_ROUNDS         = int(os.getenv('MAX_ROUNDS', 10))
WAIT_TIMEOUT       = int(os.getenv('WAIT_TIMEOUT_SECS', 600))
DEVICE_LABEL       = os.getenv('DEVICE_LABEL', 'unknown_device')
QUANTIZATION_MODE  = os.getenv('QUANTIZATION_MODE', 'none').lower()
RUN_ID             = int(os.getenv('RUN_ID', 1))
NUM_INFERENCE_RUNS = int(os.getenv('NUM_INFERENCE_RUNS', 3))
DO_BENCHMARK       = os.getenv('DO_BENCHMARK', 'false').lower() == 'true'

SEND_QUEUE  = 'edge_to_fog_models'
RECV_QUEUE  = f'edge_{EDGE_NAME}_messages_queue'

TFLITE_DIR   = os.path.join(EdgeResourcesPaths.MODELS_FOLDER_PATH.value, "tflite")
RESULTS_PATH = os.path.join(
    EdgeResourcesPaths.MODELS_FOLDER_PATH.value,
    f"exp6_{QUANTIZATION_MODE}_run{RUN_ID}.json"
)
os.makedirs(TFLITE_DIR, exist_ok=True)

SEQUENCE_LENGTH = 144
FEATURE_COLUMNS = [
    'value_diff',
    'value_rolling_mean_3', 'value_rolling_mean_6',
    'value_rolling_mean_12', 'value_rolling_mean_24',
    'value_volatility_3', 'value_volatility_6',
    'value_volatility_12', 'value_volatility_24',
    'value_ewm_3', 'value_ewm_6', 'value_ewm_12', 'value_ewm_24',
    'drift_flag', 'time_since_last_spike',
]


# =============================================================================
# Preprocessing
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
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_all = df[feature_cols].astype('float32').values
    y_all = df['value'].astype('float32').values
    X_seqs, y_seqs = [], []
    for i in range(SEQUENCE_LENGTH, len(X_all)):
        X_seqs.append(X_all[i - SEQUENCE_LENGTH:i])
        y_seqs.append(y_all[i])
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


# =============================================================================
# Peak RAM Monitor
# =============================================================================
class PeakRamMonitor:
    def __init__(self, interval=0.1):
        self._interval = interval
        self._peak_mb  = 0.0
        self._stop     = threading.Event()
        self._proc     = psutil.Process(os.getpid())
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                mb = self._proc.memory_info().rss / (1024 ** 2)
                if mb > self._peak_mb:
                    self._peak_mb = mb
            except psutil.NoSuchProcess:
                break
            time.sleep(self._interval)

    def start(self): self._thread.start()
    def stop(self):
        self._stop.set()
        self._thread.join()
        return round(self._peak_mb, 2)


# =============================================================================
# TFLite Inference Benchmark
# =============================================================================
def run_tflite_inference(tflite_path: str, X: np.ndarray, y_true: np.ndarray) -> dict:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    ram_monitor = PeakRamMonitor()
    ram_monitor.start()

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    y_pred  = []
    t_start = time.perf_counter()
    for i in range(len(X)):
        interpreter.set_tensor(inp[0]['index'], X[i:i+1])
        interpreter.invoke()
        y_pred.append(float(interpreter.get_tensor(out[0]['index']).flatten()[0]))

    total_s  = time.perf_counter() - t_start
    peak_ram = ram_monitor.stop()
    y_pred   = np.array(y_pred, dtype=np.float32)

    return {
        "peak_ram_mb":           round(peak_ram, 2),
        "latency_ms_per_sample": round((total_s / len(X)) * 1000, 4),
        "mse": round(float(mean_squared_error(y_true, y_pred)), 6),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "r2":  round(float(r2_score(y_true, y_pred)), 6),
    }


def benchmark_tflite(tflite_path: str, X_eval: np.ndarray, y_eval: np.ndarray) -> dict:
    """NUM_INFERENCE_RUNS επαναλήψεις, επιστρέφει μέσους όρους."""
    runs = []
    for i in range(1, NUM_INFERENCE_RUNS + 1):
        result = run_tflite_inference(tflite_path, X_eval, y_eval)
        runs.append(result)
        logger.info(
            f"[Exp6] Benchmark run {i}/{NUM_INFERENCE_RUNS}: "
            f"lat={result['latency_ms_per_sample']:.3f}ms | "
            f"RAM={result['peak_ram_mb']:.1f}MB | R²={result['r2']:.4f}"
        )

    def avg(k): return round(float(np.mean([r[k] for r in runs])), 6)
    def std(k): return round(float(np.std( [r[k] for r in runs])), 6)

    return {
        "n_runs":                      NUM_INFERENCE_RUNS,
        "latency_ms_per_sample_mean":  avg("latency_ms_per_sample"),
        "latency_ms_per_sample_std":   std("latency_ms_per_sample"),
        "peak_ram_mb_mean":            avg("peak_ram_mb"),
        "peak_ram_mb_std":             std("peak_ram_mb"),
        "r2_mean":  avg("r2"),   "r2_std":  std("r2"),
        "mae_mean": avg("mae"),
        "mse_mean": avg("mse"),
        "runs": runs,
    }


# =============================================================================
# FL Communication
# =============================================================================
def _pika_connect(retries=10, delay=5):
    for attempt in range(1, retries + 1):
        try:
            return pika.BlockingConnection(pika.ConnectionParameters(
                host=FOG_HOST, port=FOG_PORT, heartbeat=60))
        except Exception as e:
            logger.warning(f"RabbitMQ connect failed ({attempt}/{retries}): {e}")
            if attempt == retries:
                raise
            time.sleep(delay)


def send_to_fog(metrics: dict) -> dict:
    model_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        return {}

    t_ser = time.perf_counter()
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    model_b64 = base64.b64encode(model_bytes).decode('utf-8')
    ser_time  = round(time.perf_counter() - t_ser, 4)

    enc_time  = 0.0
    encrypted = False
    if AES_ENABLED:
        try:
            t_enc = time.perf_counter()
            model_b64, _ = encrypt_to_b64(model_bytes)
            enc_time  = round(time.perf_counter() - t_enc, 4)
            encrypted = True
        except Exception as e:
            logger.error(f"Encryption failed: {e}")

    payload = json.dumps({
        'edge_mac': EDGE_MAC, 'edge_name': EDGE_NAME,
        'model': model_b64, 'metrics': metrics, 'encrypted': encrypted,
    }).encode('utf-8')

    t_send = time.perf_counter()
    conn   = _pika_connect()
    ch     = conn.channel()
    ch.queue_declare(queue=SEND_QUEUE, durable=True)
    ch.basic_publish(
        exchange='', routing_key=SEND_QUEUE, body=payload,
        properties=pika.BasicProperties(delivery_mode=2, content_type='application/json'),
    )
    conn.close()
    send_time = round(time.perf_counter() - t_send, 4)

    return {
        "serialization_time_secs": ser_time,
        "encrypt_time_secs":       enc_time,
        "send_time_secs":          send_time,
        "total_send_time_secs":    round(ser_time + enc_time + send_time, 4),
        "payload_size_bytes":      len(payload),
        "encrypted":               encrypted,
    }


def wait_for_fog_model(timeout_secs: int) -> dict | None:
    logger.info(f"Waiting for model from Fog (timeout={timeout_secs}s)...")
    conn     = _pika_connect()
    ch       = conn.channel()
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
            if str(msg.get('command', '')) == '2':
                ch.basic_ack(delivery_tag=method.delivery_tag)
                received[0] = msg
                break
            else:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to parse Fog message: {e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)

    conn.close()
    return received[0]


def apply_fog_model(msg: dict) -> tuple[bool, str]:
    """
    Αποθηκεύει το μοντέλο από το Fog.
    Αν είναι TFLite, αποθηκεύεται στο TFLITE_DIR.
    Επιστρέφει (success, format) όπου format='keras' ή 'tflite'.
    """
    model_b64  = msg.get('model')
    model_fmt  = msg.get('model_format', 'keras')
    if not model_b64:
        return False, 'keras'

    model_bytes = base64.b64decode(model_b64)

    if model_fmt == 'tflite':
        tflite_path = os.path.join(TFLITE_DIR, f"global_{QUANTIZATION_MODE}.tflite")
        os.makedirs(TFLITE_DIR, exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(model_bytes)
        logger.info(f"TFLite model saved: {tflite_path} ({len(model_bytes)/1024:.1f} KB)")
        return True, 'tflite'
    else:
        model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(model_bytes)
        logger.info(f"Keras model saved: {model_path}")
        return True, 'keras'


def validate_input_data():
    data_path = EdgeResourcesPaths.INPUT_DATA_PATH.value
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data not found: {data_path}")
    df = pd.read_csv(data_path, nrows=5)
    if 'timestamp' not in df.columns or 'consumption_kwh' not in df.columns:
        raise ValueError("Missing required columns.")
    os.makedirs(os.path.join(os.path.dirname(data_path), "filtered_data"), exist_ok=True)


def create_base_model():
    model_path   = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    trained_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for p in [model_path, trained_path]:
        if os.path.exists(p):
            os.remove(p)
    model = create_model('simple_lstm_two_gates')
    model.save(model_path)
    logger.info(f"Base model created: {model_path}")


def save_results(results: list):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    output = {
        "experiment":         6,
        "edge_name":          EDGE_NAME,
        "device":             DEVICE_LABEL,
        "quantization_mode":  QUANTIZATION_MODE,
        "run_id":             RUN_ID,
        "timestamp":          datetime.now().isoformat(),
        "max_rounds":         MAX_ROUNDS,
        "rounds":             results,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    logger.info("=" * 65)
    logger.info(" EXPERIMENT 6 — FL CYCLE + TinyML BENCHMARK")
    logger.info(f" Edge: {EDGE_NAME} | Device: {DEVICE_LABEL}")
    logger.info(f" Mode: {QUANTIZATION_MODE} | Run: {RUN_ID}/{5}")
    logger.info(f" Benchmark: {'YES' if DO_BENCHMARK else 'NO'}")
    logger.info("=" * 65)

    try:
        validate_input_data()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Data validation failed: {e}")
        sys.exit(1)

    create_base_model()

    # Φόρτωση eval data μία φορά
    if DO_BENCHMARK:
        logger.info("[Exp6] Φόρτωση evaluation data...")
        df = preprocess_raw(pd.read_csv(EdgeResourcesPaths.INPUT_DATA_PATH.value))
        X_all, y_all = build_sequences(df)
        split = int(0.8 * len(X_all))
        X_eval, y_eval = X_all[split:], y_all[split:]
        logger.info(f"[Exp6] Evaluation sequences: {len(X_eval)}")

    all_round_results = []

    for round_num in range(1, MAX_ROUNDS + 1):
        logger.info(f"\n{'='*65}")
        logger.info(f" ROUND {round_num}/{MAX_ROUNDS} | Mode: {QUANTIZATION_MODE} | Run: {RUN_ID}")
        logger.info(f"{'='*65}")

        round_result  = {"round": round_num}
        t_round_start = time.perf_counter()

        # 1. Train
        try:
            metrics = train_local_edge_model(training_date=TEST_DATE)
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            break

        round_result["training_time_secs"] = metrics.get("training_time_secs")
        round_result["peak_ram_mb"]        = metrics.get("peak_ram_mb")
        round_result["after_metrics"]      = metrics.get("after_training", {})

        # 2. Send to Fog
        try:
            send_timing = send_to_fog(metrics)
            round_result.update(send_timing)
        except Exception as e:
            logger.exception(f"Send to Fog failed: {e}")
            break

        # 3. Αναμονή global μοντέλου
        fog_msg = wait_for_fog_model(WAIT_TIMEOUT)
        if fog_msg is None:
            logger.error(f"[Round {round_num}] No response from Fog.")
            break

        # 4. Αποθήκευση global μοντέλου
        success, model_fmt = apply_fog_model(fog_msg)
        if not success:
            logger.warning(f"[Round {round_num}] Could not apply fog model.")
            break

        round_result["model_format"] = model_fmt

        # 5. Inference Benchmark (μόνο αν DO_BENCHMARK=true και μοντέλο είναι TFLite)
        if DO_BENCHMARK and model_fmt == 'tflite':
            tflite_path = os.path.join(TFLITE_DIR, f"global_{QUANTIZATION_MODE}.tflite")
            if os.path.exists(tflite_path):
                file_size_kb = os.path.getsize(tflite_path) / 1024
                logger.info(f"[Exp6] Inference benchmark ({file_size_kb:.1f} KB)...")
                try:
                    benchmark = benchmark_tflite(tflite_path, X_eval, y_eval)
                    benchmark["file_size_kb"] = round(file_size_kb, 1)
                    round_result["inference_benchmark"] = benchmark
                except Exception as e:
                    logger.exception(f"Benchmark failed: {e}")
                    round_result["inference_benchmark"] = {"error": str(e)}

        round_result["total_round_time_secs"] = round(
            time.perf_counter() - t_round_start, 2
        )

        logger.info(
            f"[Round {round_num}] DONE | "
            f"train={round_result.get('training_time_secs')}s | "
            f"RAM={round_result.get('peak_ram_mb')}MB | "
            f"fmt={model_fmt} | "
            f"total={round_result['total_round_time_secs']}s"
        )

        all_round_results.append(round_result)

    if all_round_results:
        save_results(all_round_results)

    logger.info("\n" + "=" * 65)
    logger.info(f" EXPERIMENT 6 COMPLETE | Mode: {QUANTIZATION_MODE} | Run: {RUN_ID}")
    logger.info("=" * 65)