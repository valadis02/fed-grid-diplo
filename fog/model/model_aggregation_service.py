import os
import time
import threading
import numpy as np
import tensorflow as tf
import psutil
from scipy.optimize import curve_fit
from fog.communication.fog_resources_paths import FogResourcesPaths
from fog.model.fltrust2_server_training import train_server_model as fltrust2_train_server
from shared.utils import delete_files_containing
from shared.logging_config import logger

# β”€β”€ Aggregation strategy β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€
AGGREGATION_STRATEGY  = os.getenv('AGGREGATION_STRATEGY', 'fedavg').lower()
NUM_BYZANTINE         = int(os.getenv('KRUM_NUM_BYZANTINE', '0'))
MULTI_KRUM_M          = int(os.getenv('MULTI_KRUM_M', '0'))
TMEAN_TRIM_FRACTION   = float(os.getenv('TMEAN_TRIM_FRACTION', '0.0'))

# β”€β”€ FLTrust Parameters β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€
FLTRUST_ROOT_DATA_PATH    = os.getenv('FLTRUST_ROOT_DATA_PATH', '')
FLTRUST_LR                = float(os.getenv('FLTRUST_LR', '0.001'))
FLTRUST_EPOCHS            = int(os.getenv('FLTRUST_EPOCHS', '1'))
FLTRUST_TS_THRESHOLD      = float(os.getenv('FLTRUST_TS_THRESHOLD', '0.05'))
FLTRUST_SERVER_NODE       = os.getenv('FLTRUST_SERVER_NODE', 'edge_node_1')
FLTRUST_ROUND_FLAG_PATH   = os.getenv(
    'FLTRUST_ROUND_FLAG_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fltrust_initialized.flag')
)
FLTRUST_PREV_WEIGHTS_PATH = os.getenv(
    'FLTRUST_PREV_WEIGHTS_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fltrust_prev_weights.npz')
)

# β”€β”€ FedEMA Parameters β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€
FEDEMA_MIN_ROUNDS     = int(os.getenv('FEDEMA_MIN_ROUNDS', '3'))
FEDEMA_RANSAC_ITERS   = int(os.getenv('FEDEMA_RANSAC_ITERS', '30'))
FEDEMA_SAMPLE_SIZE    = int(os.getenv('FEDEMA_SAMPLE_SIZE', '3'))
FEDEMA_THRESHOLD_K    = float(os.getenv('FEDEMA_THRESHOLD_K', '0.10'))
FEDEMA_REFIT_STEPS    = int(os.getenv('FEDEMA_REFIT_STEPS', '200'))
FEDEMA_COS_THRESHOLD  = float(os.getenv('FEDEMA_COS_THRESHOLD', '0.5'))

# β”€β”€ FedConsensus Parameters β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€
FEDCONSENSUS_MAD_K = float(os.getenv('FEDCONSENSUS_MAD_K', '1.4826'))

# β”€β”€ FedConsensus-P Parameters β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€
FCP_EMA_BETA           = float(os.getenv('FCP_EMA_BETA', '0.5'))
FCP_MODELS_DIR         = os.getenv(
    'FCP_MODELS_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fcp_models')
)

logger.info(
    f"Aggregation strategy: {AGGREGATION_STRATEGY.upper()} | "
    f"num_byzantine={NUM_BYZANTINE}"
)

# β”€β”€ FedEMA global state β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€
_fedema_history: dict = {}
_fedema_round: int    = 0

# β”€β”€ FedConsensus-P global state β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€
_fcp_trust: dict = {"T": None, "ids": None}

# β”€β”€ Hardware monitoring β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€

class PeakRamMonitor:
    def __init__(self, interval: float = 0.5):
        self._interval  = interval
        self._peak_mb   = 0.0
        self._stop      = threading.Event()
        self._process   = psutil.Process(os.getpid())
        self._thread    = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                mb = self._process.memory_info().rss / (1024 ** 2)
                if mb > self._peak_mb:
                    self._peak_mb = mb
            except psutil.NoSuchProcess:
                break
            time.sleep(self._interval)

    def start(self):
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return round(self._peak_mb, 2)


def _read_cpu_temp() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        return -1.0


def _estimate_energy_wh(cpu_usage_pct: float, duration_secs: float,
                         tdp_watts: float = 5.0) -> float:
    tdp = float(os.getenv('FOG_TDP_WATTS', str(tdp_watts)))
    return round((cpu_usage_pct / 100.0) * tdp * (duration_secs / 3600.0), 6)


def _wait_for_file(path: str, timeout_s: float = 5.0, poll_s: float = 0.1) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def _safe_load_model(path: str, retries: int = 3, delay_s: float = 0.25):
    last_err = None
    for _ in range(retries):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            last_err = e
            time.sleep(delay_s)
    raise last_err


def initialize_fltrust_reference_model():
    pass


# β”€β”€ Aggregation algorithms β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€

def _fedavg(weights_list: list, scores: list) -> list:
    total = sum(scores)
    if total == 0:
        total = 1.0
    result = [np.zeros_like(w) for w in weights_list[0]]
    for w_list, score in zip(weights_list, scores):
        for i, w in enumerate(w_list):
            result[i] += (score / total) * w
    return result


def _krum(weights_list: list, f: int) -> list:
    n = len(weights_list)
    if n <= 2 * f + 2:
        logger.warning(f"Krum: n={n} <= 2f+2={2*f+2} β€” falling back to FedAvg.")
        return _fedavg(weights_list, [1.0] * n)

    flat = [np.concatenate([w.flatten() for w in wl]) for wl in weights_list]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((flat[i] - flat[j]) ** 2)
            distances[i, j] = d
            distances[j, i] = d

    k = n - f - 2
    scores = []
    for i in range(n):
        dists_i = sorted([distances[i, j] for j in range(n) if j != i])
        scores.append(sum(dists_i[:k]))

    selected = int(np.argmin(scores))
    logger.info(f"Krum: selected client {selected} | score={scores[selected]:.4f} | n={n} | f={f} | k={k}")
    return weights_list[selected]


def _multi_krum(weights_list: list, f: int, m: int = 0) -> list:
    n = len(weights_list)
    if m <= 0:
        m = max(1, n - f)
    if n <= 2 * f + 2:
        logger.warning(f"Multi-Krum: n={n} <= 2f+2={2*f+2} β€” falling back to FedAvg.")
        return _fedavg(weights_list, [1.0] * n)

    flat = [np.concatenate([w.flatten() for w in wl]) for wl in weights_list]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((flat[i] - flat[j]) ** 2)
            distances[i, j] = d
            distances[j, i] = d

    k = n - f - 2
    scores = []
    for i in range(n):
        dists_i = sorted([distances[i, j] for j in range(n) if j != i])
        scores.append(sum(dists_i[:k]))

    selected_indices = sorted(range(n), key=lambda i: scores[i])[:m]
    logger.info(f"Multi-Krum: selected {m}/{n} clients: {selected_indices} | f={f}")
    return _fedavg([weights_list[i] for i in selected_indices], [1.0] * m)


def _trimmed_mean(weights_list: list, f: int, trim_fraction: float = 0.0) -> list:
    n = len(weights_list)
    f_actual = max(0, int(trim_fraction * n)) if trim_fraction > 0.0 else f
    if 2 * f_actual >= n:
        logger.warning(f"TrimmedMean: 2f={2*f_actual} >= n={n} β€” falling back to FedAvg.")
        return _fedavg(weights_list, [1.0] * n)

    result = []
    for layer_idx in range(len(weights_list[0])):
        stacked = np.stack([weights_list[i][layer_idx] for i in range(n)], axis=0)
        trimmed = np.sort(stacked, axis=0)[f_actual: n - f_actual]
        result.append(np.mean(trimmed, axis=0))

    logger.info(f"TrimmedMean: f={f_actual} | kept {n - 2*f_actual}/{n} clients")
    return result


# β”€β”€ FLTrust core β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€

def _load_fltrust_prev_weights() -> list | None:
    try:
        if os.path.exists(FLTRUST_PREV_WEIGHTS_PATH):
            data = np.load(FLTRUST_PREV_WEIGHTS_PATH, allow_pickle=True)
            weights = [data[f'arr_{i}'] for i in range(len(data.files))]
            logger.info(f"FLTrust: loaded prev weights from {FLTRUST_PREV_WEIGHTS_PATH} ({len(weights)} layers)")
            return weights
    except Exception as e:
        logger.warning(f"FLTrust: failed to load prev weights: {e}")
    return None


def _save_fltrust_prev_weights(weights: list):
    try:
        save_dir = os.path.dirname(FLTRUST_PREV_WEIGHTS_PATH)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez(FLTRUST_PREV_WEIGHTS_PATH, *weights)
        logger.info(f"FLTrust: saved prev weights to {FLTRUST_PREV_WEIGHTS_PATH}")
    except Exception as e:
        logger.warning(f"FLTrust: failed to save prev weights: {e}")


def _fltrust(weights_list: list, w_prev: list, w_server: list, f: int = 0) -> list:
    n = len(weights_list)

    g0_layers = [ws - wp for ws, wp in zip(w_server, w_prev)]
    g0_flat   = np.concatenate([g.flatten() for g in g0_layers])
    g0_norm   = np.linalg.norm(g0_flat)

    if g0_norm < 1e-10:
        logger.warning("FLTrust: server gradient near-zero β€” fallback to FedAvg.")
        return _fedavg(weights_list, [1.0] * n)

    trust_scores      = []
    clipped_gradients = []

    for i, w_client in enumerate(weights_list):
        g_i_layers = [wc - wp for wc, wp in zip(w_client, w_prev)]
        g_i_flat   = np.concatenate([g.flatten() for g in g_i_layers])
        g_i_norm   = np.linalg.norm(g_i_flat)

        if g_i_norm < 1e-10:
            cos_sim = 0.0
            ts      = 0.0
            clipped = [np.zeros_like(g) for g in g_i_layers]
        else:
            cos_sim = float(np.dot(g0_flat, g_i_flat) / (g0_norm * g_i_norm))
            ts      = max(0.0, cos_sim)
            scale   = g0_norm / g_i_norm
            clipped = [g * scale for g in g_i_layers]

        trust_scores.append(ts)
        clipped_gradients.append(clipped)
        logger.info(f"FLTrust: client {i} | cosine_sim={cos_sim:.4f} | TS={ts:.4f}")

    total_ts = sum(trust_scores)

    if total_ts < 1e-10:
        logger.warning("FLTrust: all trust scores zero β€” fallback to FedAvg.")
        return _fedavg(weights_list, [1.0] * n)

    logger.info(
        f"FLTrust: n={n} | total_TS={total_ts:.4f} | "
        f"active={sum(ts > 0 for ts in trust_scores)}/{n}"
    )

    aggregated = []
    for layer_idx in range(len(w_prev)):
        weighted_grad = sum(
            trust_scores[i] * clipped_gradients[i][layer_idx]
            for i in range(n)
        ) / total_ts
        aggregated.append(w_prev[layer_idx] + weighted_grad)

    return aggregated


# β”€β”€ FedEMA helpers β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean_cosine_per_node(flat: list) -> list:
    n = len(flat)
    result = []
    for i in range(n):
        sims = [_cosine_similarity(flat[i], flat[j]) for j in range(n) if j != i]
        result.append(float(np.mean(sims)))
    return result


def _exponential_curve(t: np.ndarray, c: float, lam: float) -> np.ndarray:
    return c * (1.0 - np.exp(-lam * t))


def _fit_curve(rounds: np.ndarray, values: np.ndarray):
    try:
        params, _ = curve_fit(
            _exponential_curve,
            rounds, values,
            p0=[1.0, 0.3],
            bounds=([0, 0], [2.0, 10.0]),
            maxfev=FEDEMA_REFIT_STEPS
        )
        return params
    except Exception:
        return None


def _ransac_inliers(node_ids: list, histories: dict) -> list:
    n          = len(node_ids)
    num_rounds = len(next(iter(histories.values())))
    rounds_arr = np.arange(1, num_rounds + 1, dtype=float)
    best_inliers = []

    for _ in range(FEDEMA_RANSAC_ITERS):
        sample_size = min(FEDEMA_SAMPLE_SIZE, n)
        sample_ids  = list(np.random.choice(node_ids, size=sample_size, replace=False))

        sample_mean = np.mean([histories[nid] for nid in sample_ids], axis=0)
        params = _fit_curve(rounds_arr, sample_mean)
        if params is None:
            continue

        c, lam    = params
        predicted = _exponential_curve(rounds_arr, c, lam)

        predicted_last  = float(predicted[-1])
        curve_range     = abs(float(predicted[-1]) - float(predicted[0]))
        tolerance       = max(0.005, min(FEDEMA_THRESHOLD_K, (curve_range / 0.8) * FEDEMA_THRESHOLD_K))
        threshold       = max(predicted_last * tolerance, 0.005)

        residuals = [
            (nid, float(np.mean(np.abs(np.array(histories[nid]) - predicted))))
            for nid in node_ids
        ]
        inliers = [nid for nid, res in residuals if res <= threshold]

        if len(inliers) < n // 2:
            continue

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if not best_inliers:
        logger.warning("FedEMA RANSAC: no inliers found β€” returning all nodes.")
        return node_ids

    return best_inliers


def _cosine_cluster_inliers(node_ids: list, mean_cos: list) -> list:
    pairs   = sorted(zip(node_ids, mean_cos), key=lambda x: x[1], reverse=True)
    inliers = [nid for nid, cos in pairs if cos >= FEDEMA_COS_THRESHOLD]

    if len(inliers) < max(1, len(node_ids) // 2):
        half    = max(1, len(pairs) // 2)
        inliers = [nid for nid, _ in pairs[:half]]

    return inliers


def _fedema(weights_list: list, node_ids: list) -> list:
    global _fedema_round, _fedema_history

    _fedema_round += 1
    n = len(weights_list)
    logger.info(f"FedEMA: round={_fedema_round} | n_clients={n}")

    flat     = [np.concatenate([w.flatten() for w in wl]) for wl in weights_list]
    mean_cos = _mean_cosine_per_node(flat)

    for nid, cos in zip(node_ids, mean_cos):
        logger.info(f"FedEMA: node={nid} | mean_cos_sim={cos:.4f}")

    if _fedema_round > 1:
        for nid, cos in zip(node_ids, mean_cos):
            _fedema_history.setdefault(nid, []).append(cos)

    if _fedema_round < FEDEMA_MIN_ROUNDS:
        best_idx   = int(np.argmax(mean_cos))
        best_id    = node_ids[best_idx]
        logger.info(f"FedEMA: warmup round {_fedema_round} β€” selecting best cosine node: {best_id} (cos={mean_cos[best_idx]:.4f})")
        inlier_ids = [best_id]
    else:
        history_len = _fedema_round - 1
        valid_ids = [
            nid for nid in node_ids
            if len(_fedema_history.get(nid, [])) == history_len
        ]
        if len(valid_ids) < FEDEMA_SAMPLE_SIZE:
            logger.warning("FedEMA: not enough valid nodes for RANSAC β€” using cosine clustering.")
            inlier_ids = _cosine_cluster_inliers(node_ids, mean_cos)
        else:
            inlier_ids = _ransac_inliers(valid_ids, _fedema_history)

    outlier_ids = [nid for nid in node_ids if nid not in inlier_ids]
    logger.info(f"FedEMA: inliers={inlier_ids} ({len(inlier_ids)}/{n})")
    if outlier_ids:
        logger.warning(f"FedEMA: excluded outliers={outlier_ids}")

    inlier_weights = [weights_list[node_ids.index(nid)] for nid in inlier_ids]
    result = [np.zeros_like(w) for w in inlier_weights[0]]
    for wl in inlier_weights:
        for i, w in enumerate(wl):
            result[i] += w / len(inlier_weights)

    return result


# β”€β”€ FedConsensus (original) β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€

def _fedconsensus(weights_list: list, node_ids: list, w_prev: list) -> list:
    n = len(weights_list)

    gradients  = []
    flat_grads = []
    for w_client in weights_list:
        g_layers = [wc - wp for wc, wp in zip(w_client, w_prev)]
        gradients.append(g_layers)
        flat_grads.append(np.concatenate([g.flatten() for g in g_layers]))

    grad_norms = np.array([np.linalg.norm(f) for f in flat_grads])

    trust_matrix = np.zeros((n, n))

    for ref_idx in range(n):
        g_ref_norm = grad_norms[ref_idx]
        ref_id     = node_ids[ref_idx]

        logger.info(f"FedConsensus [ref={ref_id}] | grad_norm={g_ref_norm:.6f}")

        if g_ref_norm < 1e-10:
            logger.info(f"FedConsensus [ref={ref_id}]: skipped (near-zero gradient)")
            continue

        for i in range(n):
            if grad_norms[i] < 1e-10:
                trust_matrix[ref_idx][i] = 0.0
            else:
                cos = float(np.dot(flat_grads[ref_idx], flat_grads[i])
                            / (g_ref_norm * grad_norms[i]))
                trust_matrix[ref_idx][i] = max(0.0, cos)

        ts_str = " | ".join(f"{node_ids[i]}={trust_matrix[ref_idx][i]:.3f}" for i in range(n))
        logger.info(f"FedConsensus [ref={ref_id}]: {ts_str}")

    mean_received = np.array([
        np.mean([trust_matrix[j][i] for j in range(n) if j != i])
        for i in range(n)
    ])

    adaptive_threshold = float(np.mean(mean_received))
    logger.info(
        f"FedConsensus: mean_received={[f'{mean_received[i]:.3f}' for i in range(n)]} "
        f"| adaptive_threshold={adaptive_threshold:.3f}"
    )

    trusted_mask = []
    for i in range(n):
        trust_count = sum(
            trust_matrix[j][i] > adaptive_threshold
            for j in range(n) if j != i
        )
        is_trusted = trust_count > n // 2
        trusted_mask.append(is_trusted)
        logger.info(
            f"FedConsensus voting: node={node_ids[i]} | "
            f"mean_score={mean_received[i]:.3f} | "
            f"trusted_by={trust_count}/{n-1} peers | "
            f"status={'TRUSTED' if is_trusted else 'EXCLUDED'}"
        )

    trusted_indices = [i for i, t in enumerate(trusted_mask) if t]

    if not trusted_indices:
        logger.warning("FedConsensus: no trusted nodes found β€” FedAvg fallback.")
        return _fedavg(weights_list, [1.0] * n)

    logger.info(
        f"FedConsensus: trusted={[node_ids[i] for i in trusted_indices]} | "
        f"excluded={[node_ids[i] for i in range(n) if not trusted_mask[i]]}"
    )

    trust_scores      = []
    clipped_gradients = []
    median_norm = float(np.median([grad_norms[j] for j in trusted_indices]))

    for i in trusted_indices:
        g_i_norm = grad_norms[i]
        peer_scores = [trust_matrix[j][i] for j in trusted_indices if j != i]
        ts = float(np.mean(peer_scores)) if peer_scores else 1.0

        if g_i_norm > 1e-10:
            scale   = median_norm / g_i_norm
            clipped = [g * scale for g in gradients[i]]
        else:
            scale   = 0.0
            clipped = [np.zeros_like(g) for g in gradients[i]]

        trust_scores.append(ts)
        clipped_gradients.append(clipped)
        logger.info(
            f"FedConsensus agg: node={node_ids[i]} | "
            f"peer_trust={ts:.4f} | grad_norm={g_i_norm:.6f} | scale={scale:.4f}"
        )

    total_ts = sum(trust_scores)

    if total_ts < 1e-10:
        logger.warning("FedConsensus: all trust scores zero β€” FedAvg fallback.")
        return _fedavg(weights_list, [1.0] * n)

    logger.info(f"FedConsensus: total_TS={total_ts:.4f} | active={len(trusted_indices)}/{n}")

    aggregated = []
    for layer_idx in range(len(w_prev)):
        weighted_grad = sum(
            trust_scores[k] * clipped_gradients[k][layer_idx]
            for k in range(len(trusted_indices))
        ) / total_ts
        aggregated.append(w_prev[layer_idx] + weighted_grad)

    return aggregated


# β”€β”€ FedConsensus-SV (Soft Voting) β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€

def _fedconsensus_sv(weights_list: list, node_ids: list, w_prev: list) -> list:
    """
    FedConsensus-SV: per-voter ΟƒΟ‡ΞµΟ„ΞΉΞΊΟ ΞΊΞ±Ο„ΟΟ†Ξ»ΞΉ + ΟƒΟ…Ξ½ΞµΟ‡Ξ® Ξ²Ξ¬ΟΞ·.

    Ξ”ΞΉΞ±Ο†ΞΏΟΞ­Ο‚ Ξ±Ο€Ο Ο„ΞΏ original:
    1. Ξ ΞΊΟΞΌΞ²ΞΏΟ‚ j ΟΞ·Ο†Ξ―Ξ¶ΞµΞΉ Ο„ΞΏΞ½ i Ξ±Ξ½ T[j,i] > mean(T[j,:]) β€” ΟƒΟ‡ΞµΟ„ΞΉΞΊΟ,
       ΟΟ‡ΞΉ global ΞΊΞ±Ο„ΟΟ†Ξ»ΞΉ. ΞΞ¬ΞΈΞµ ΞΊΟΞΌΞ²ΞΏΟ‚ ΞΊΟΞ―Ξ½ΞµΞΉ ΞΌΞµ Ξ²Ξ¬ΟƒΞ· Ο,Ο„ΞΉ Ξ²Ξ»Ξ­Ο€ΞµΞΉ ΞΏ Ξ―Ξ΄ΞΉΞΏΟ‚.
    2. Ξ’Ξ¬ΟΞΏΟ‚ ΞΊΟΞΌΞ²ΞΏΟ… i = votes_iΒ² / Ξ£votesΒ² β€” ΟƒΟ…Ξ½ΞµΟ‡Ξ­Ο‚, ΟΟ‡ΞΉ Ξ΄Ο…Ξ±Ξ΄ΞΉΞΊΟ.
       Ξ attacker Ο€Ξ±Ξ―ΟΞ½ΞµΞΉ Ξ»Ξ―Ξ³ΞµΟ‚ ΟΞ®Ο†ΞΏΟ…Ο‚ β†’ ΞΌΞΉΞΊΟΟ Ξ²Ξ¬ΟΞΏΟ‚ β†’ ΞΌΞΉΞΊΟΞ® ΞµΟ€ΞΉΟΟΞΏΞ®.
    3. Trust matrix ΞΌΞµ Ξ•ΞΞ‘ matmul (vectorized) Ξ±Ξ½Ο„Ξ― Ξ³ΞΉΞ± per-pair loops.

    Drop-in Ξ±Ξ½Ο„ΞΉΞΊΞ±Ο„Ξ¬ΟƒΟ„Ξ±ΟƒΞ·: Ξ―Ξ΄ΞΉΞ± ΞµΞ―ΟƒΞΏΞ΄ΞΏΟ‚/Ξ­ΞΎΞΏΞ΄ΞΏΟ‚ ΞΌΞµ _fedconsensus.
    """
    n = len(weights_list)

    # gradients + flat matrix
    gradients, flat_grads = [], []
    for w_client in weights_list:
        g_layers = [wc - wp for wc, wp in zip(w_client, w_prev)]
        gradients.append(g_layers)
        flat_grads.append(np.concatenate([g.flatten() for g in g_layers]))

    F = np.stack(flat_grads)                                 # (n, d)
    norms = np.linalg.norm(F, axis=1)

    # vectorized ReLU-cosine trust matrix
    Fn = F / np.maximum(norms[:, None], 1e-12)
    T = np.clip(Fn @ Fn.T, 0.0, None)
    T[norms < 1e-10, :] = 0.0
    T[:, norms < 1e-10] = 0.0
    np.fill_diagonal(T, 0.0)

    # per-voter relative threshold: j votes for i if T[j,i] > mean(T[j,:])
    votes = np.zeros(n)
    for j in range(n):
        row_mean = T[j].sum() / max(n - 1, 1)
        votes += (T[j] > row_mean).astype(float)

    for i in range(n):
        w_share = (votes[i] ** 2) / max((votes ** 2).sum(), 1e-12)
        logger.info(
            f"FedConsensus-SV: node={node_ids[i]} | "
            f"votes={votes[i]:.0f}/{n-1} | weight_share={w_share:.3f}"
        )

    w = votes ** 2
    if w.sum() < 1e-10:
        logger.warning("FedConsensus-SV: zero votes everywhere β€” FedAvg fallback.")
        return _fedavg(weights_list, [1.0] * n)
    w = w / w.sum()

    voted = np.where(votes > 0)[0]
    median_norm = float(np.median(norms[voted])) if len(voted) else float(np.median(norms))

    # norm clipping + weighted aggregation
    aggregated = []
    for layer_idx in range(len(w_prev)):
        weighted_grad = np.zeros_like(w_prev[layer_idx], dtype=float)
        for k in range(n):
            if norms[k] > 1e-10:
                scale = median_norm / norms[k]
                weighted_grad += w[k] * gradients[k][layer_idx] * scale
            # norms[k] β‰ 0: contributes 0 (already zeroed)
        aggregated.append(w_prev[layer_idx] + weighted_grad)

    logger.info(
        f"FedConsensus-SV: n={n} | active(votes>0)={len(voted)}/{n} "
        f"| median_norm={median_norm:.6f}"
    )
    return aggregated


# β”€β”€ FedConsensus-P (Personalized) β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€

def _load_fcp_prev_weights(node_id: str, fallback: list) -> list:
    """Ξ¦ΞΏΟΟ„ΟΞ½ΞµΞΉ Ο„ΞΏ Ο€ΟΞΏΟƒΟ‰Ο€ΞΉΞΊΟ ΞΌΞΏΞ½Ο„Ξ­Ξ»ΞΏ Ο„ΞΏΟ… node_id, Ξ±Ξ»Ξ»ΞΉΟΟ‚ ΞµΟ€ΞΉΟƒΟ„ΟΞ­Ο†ΞµΞΉ Ο„ΞΏ fallback."""
    path = os.path.join(FCP_MODELS_DIR, f"fcp_model_{node_id}.npz")
    try:
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            return [data[f'arr_{i}'] for i in range(len(data.files))]
    except Exception as e:
        logger.warning(f"FedConsensus-P: could not load personal model for {node_id}: {e}")
    return fallback


def _save_fcp_weights(node_id: str, weights: list):
    try:
        os.makedirs(FCP_MODELS_DIR, exist_ok=True)
        path = os.path.join(FCP_MODELS_DIR, f"fcp_model_{node_id}.npz")
        np.savez(path, *weights)
    except Exception as e:
        logger.warning(f"FedConsensus-P: could not save personal model for {node_id}: {e}")


def _fedconsensus_p(weights_list: list, node_ids: list, w_global_prev: list) -> list:
    """
    FedConsensus-P (Personalized).

    ΞΞ¬ΞΈΞµ ΞΊΟΞΌΞ²ΞΏΟ‚ Ξ΄ΞΉΞ±Ο„Ξ·ΟΞµΞ― Ο€ΟΞΏΟƒΟ‰Ο€ΞΉΞΊΟ ΞΌΞΏΞ½Ο„Ξ­Ξ»ΞΏ (Ξ±Ο€ΞΏΞΈΞ·ΞΊΞµΟ…ΞΌΞ­Ξ½ΞΏ ΟƒΟ„ΞΏ FCP_MODELS_DIR).
    Aggregation: ΞΊΞ¬ΞΈΞµ i ΞµΞ½Ξ·ΞΌΞµΟΟΞ½ΞµΟ„Ξ±ΞΉ ΞΌΞµ weighted sum ΟΞ»Ο‰Ξ½ Ο„Ο‰Ξ½ gradients,
    ΞΌΞµ Ξ²Ξ¬ΟΞ· Ξ±Ο€Ο EMA trust matrix (trust[i,j]Β²).

    Ξ£Ξ—ΞΞ‘ΞΞ¤Ξ™ΞΞ Ξ³ΞΉΞ± integration:
    - Ξ•Ο€ΞΉΟƒΟ„ΟΞ­Ο†ΞµΞΉ Ξ•ΞΞ‘ global aggregated weights list (ΞΏ ΞΊΟΞΌΞ²ΞΏΟ‚ ΞΌΞµ Ο„ΞΏ Ο…ΟΞ·Ξ»ΟΟ„ΞµΟΞΏ
      trust score Ο„Ξ± "ΞµΞΊΟ€ΟΞΏΟƒΟ‰Ο€ΞµΞ―") Ξ³ΞΉΞ± Ξ½Ξ± ΟƒΟ‰ΞΈΞµΞ― ΟƒΟ„ΞΏ FOG_MODEL_FILE_PATH Ο‰Ο‚
      ΟƒΟ…ΞΌΞ²Ξ±Ο„ΟΟ„Ξ·Ο„Ξ± ΞΌΞµ Ο„ΞΏ Ο…Ο€Ξ¬ΟΟ‡ΞΏΞ½ pipeline β€” Ξ±Ξ»Ξ»Ξ¬ Ο„Ξ± per-node ΞΌΞΏΞ½Ο„Ξ­Ξ»Ξ±
      Ξ±Ο€ΞΏΞΈΞ·ΞΊΞµΟΞΏΞ½Ο„Ξ±ΞΉ ΞµΟƒΟ‰Ο„ΞµΟΞΉΞΊΞ¬ ΟƒΟ„ΞΏ FCP_MODELS_DIR.
    - Ξ¤ΞΏ routing (Ο€ΞΏΞΉΞΏΟ‚ edge Ο€Ξ±Ξ―ΟΞ½ΞµΞΉ Ο€ΞΏΞΉΞΏ ΞΌΞΏΞ½Ο„Ξ­Ξ»ΞΏ) Ξ³Ξ―Ξ½ΞµΟ„Ξ±ΞΉ Ξ±Ο…Ο„ΟΞΌΞ±Ο„Ξ±:
      Ο„ΞΏ ΞΊΞ¬ΞΈΞµ edge ΞΊΞΏΞΉΟ„Ξ¬ΞµΞΉ Ο„ΞΏ fcp_model_{node_id}.npz ΞΊΞ±Ο„Ξ¬ Ο„Ξ·Ξ½ Ξ±ΟΟ‡ΞΉΞΊΞΏΟ€ΞΏΞ―Ξ·ΟƒΞ·.
    - Ξ‘Ξ½ Ξ΄ΞµΞ½ Ο…Ο€Ξ¬ΟΟ‡ΞµΞΉ Ξ±Ο€ΞΏΞΈΞ·ΞΊΞµΟ…ΞΌΞ­Ξ½ΞΏ ΞΌΞΏΞ½Ο„Ξ­Ξ»ΞΏ Ξ³ΞΉΞ± ΞΊΟΞΌΞ²ΞΏ (round 1), Ο‡ΟΞ·ΟƒΞΉΞΌΞΏΟ€ΞΏΞΉΞµΞ―
      Ο„ΞΏ global w_prev Ο‰Ο‚ Ξ±ΟΟ‡ΞΉΞΊΞ® Ο„ΞΉΞΌΞ®.
    """
    global _fcp_trust
    n = len(weights_list)

    # Ο†ΟΟΟ„Ο‰ΟƒΞµ per-node w_prev (round 1: fallback ΟƒΟ„ΞΏ global)
    w_prev_per_node = {
        nid: _load_fcp_prev_weights(nid, w_global_prev)
        for nid in node_ids
    }

    # gradients Ο‰Ο‚ Ο€ΟΞΏΟ‚ Ο„ΞΏ Ο€ΟΞΏΟƒΟ‰Ο€ΞΉΞΊΟ w_prev Ο„ΞΏΟ… ΞΊΞ¬ΞΈΞµ ΞΊΟΞΌΞ²ΞΏΟ…
    gradients, flat_grads = [], []
    for w_client, nid in zip(weights_list, node_ids):
        wp = w_prev_per_node[nid]
        g_layers = [wc - wpl for wc, wpl in zip(w_client, wp)]
        gradients.append(g_layers)
        flat_grads.append(np.concatenate([g.flatten() for g in g_layers]))

    F = np.stack(flat_grads)
    norms = np.linalg.norm(F, axis=1)

    # sanitize + global norm clipping
    bad = ~np.isfinite(F).all(axis=1)
    if bad.any():
        logger.warning(
            f"FedConsensus-P: non-finite updates from "
            f"{[node_ids[i] for i in np.where(bad)[0]]} β€” zeroed."
        )
        F[bad] = 0.0
        norms[bad] = 0.0

    good = norms > 0
    median_norm = float(np.median(norms[good])) if good.any() else 1.0
    scale = np.minimum(1.0, median_norm / np.maximum(norms, 1e-12))
    F = F * scale[:, None]
    gradients = [[g * scale[i] for g in gradients[i]] for i in range(n)]
    norms = norms * scale

    # vectorized ReLU-cosine similarity
    Fn = F / np.maximum(norms[:, None], 1e-12)
    S = np.clip(Fn @ Fn.T, 0.0, None)
    np.fill_diagonal(S, 1.0)   # self-trust = 1

    # EMA trust matrix (reset Ξ±Ξ½ Ξ±Ξ»Ξ»Ξ¬ΞΎΞΏΟ…Ξ½ ΞΏΞΉ ΟƒΟ…ΞΌΞΌΞµΟ„Ξ­Ο‡ΞΏΞ½Ο„ΞµΟ‚)
    if _fcp_trust["T"] is None or _fcp_trust["ids"] != list(node_ids):
        if _fcp_trust["T"] is not None:
            logger.warning("FedConsensus-P: participant set changed β€” resetting EMA trust.")
        _fcp_trust["T"] = np.eye(n)
        _fcp_trust["ids"] = list(node_ids)

    T = (1.0 - FCP_EMA_BETA) * _fcp_trust["T"] + FCP_EMA_BETA * S
    _fcp_trust["T"] = T

    # per-node personalized aggregation
    results = {}
    for i, nid in enumerate(node_ids):
        w = np.clip(T[i], 0.0, None) ** 2
        w = w / max(w.sum(), 1e-12)

        top = np.argsort(w)[::-1][:3]
        logger.info(
            f"FedConsensus-P: node={nid} | "
            + "top_peers=" + ", ".join(f"{node_ids[j]}:{w[j]:.3f}" for j in top)
        )

        new_weights = []
        for layer_idx in range(len(w_prev_per_node[nid])):
            weighted_grad = sum(
                w[j] * gradients[j][layer_idx] for j in range(n)
            )
            new_weights.append(w_prev_per_node[nid][layer_idx] + weighted_grad)

        results[nid] = new_weights
        _save_fcp_weights(nid, new_weights)

    logger.info(f"FedConsensus-P: updated {n} personal models | median_norm={median_norm:.6f}")

    # ΞµΟ€ΞΉΟƒΟ„ΟΞ­Ο†ΞµΞΉ Ο„ΞΏ ΞΌΞΏΞ½Ο„Ξ­Ξ»ΞΏ Ο„ΞΏΟ… most-trusted ΞΊΟΞΌΞ²ΞΏΟ… Ο‰Ο‚ global (Ξ³ΞΉΞ± pipeline compatibility)
    best_idx = int(np.argmax([T[:, i].mean() for i in range(n)]))
    best_nid = node_ids[best_idx]
    logger.info(f"FedConsensus-P: global representative = {best_nid}")
    return results[best_nid]


# β”€β”€ Main aggregation function β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€

def aggregate_models_with_metrics(edge_models_cache: dict, fog_weight: float = 1.0,
                                   round_start_time: float = None):
    t_receive_end = time.perf_counter()

    cpu_temp_before = _read_cpu_temp()
    ram_monitor     = PeakRamMonitor(interval=0.5)
    ram_monitor.start()
    t_total_start   = time.perf_counter()

    weights_list, mse_scores, loaded_paths, node_ids = [], [], [], []
    for map_id, entry in edge_models_cache.items():
        model_path = entry["model_path"]
        metrics    = entry.get("metrics", {})
        mse = (metrics.get("after_training", {}).get("mse") or metrics.get("mse", 1.0))
        if not _wait_for_file(model_path, timeout_s=5.0):
            logger.warning(f"[fog] WARN: model file not ready: {model_path}")
            continue
        try:
            weights_list.append(_safe_load_model(model_path).get_weights())
            mse_scores.append(float(mse))
            loaded_paths.append(model_path)
            node_ids.append(map_id)
            logger.info(f"Loaded edge model: {map_id} | MSE={mse:.4f}")
        except Exception as e:
            logger.warning(f"[fog] WARN: failed to load edge model {model_path}: {e}")

    if not weights_list:
        ram_monitor.stop()
        logger.error("No edge models loaded β€” cannot aggregate.")
        return None

    n = len(weights_list)
    f = NUM_BYZANTINE
    inv_mse_scores = [1.0 / (m + 1e-8) for m in mse_scores]

    cpu_usage_samples = []
    stop_cpu_mon = threading.Event()

    def _cpu_mon():
        psutil.cpu_percent(interval=None)
        while not stop_cpu_mon.is_set():
            try:
                cpu_usage_samples.append(psutil.cpu_percent(interval=None))
            except Exception:
                pass
            time.sleep(0.5)

    cpu_thread = threading.Thread(target=_cpu_mon, daemon=True)
    cpu_thread.start()

    ram_before_agg = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    t_agg_start = time.perf_counter()

    # β”€β”€ Aggregation dispatch β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€β”€
    if AGGREGATION_STRATEGY == 'krum':
        aggregated_weights = _krum(weights_list, f)

    elif AGGREGATION_STRATEGY == 'multi_krum':
        m_param = MULTI_KRUM_M if MULTI_KRUM_M > 0 else max(1, n - f)
        aggregated_weights = _multi_krum(weights_list, f, m=m_param)

    elif AGGREGATION_STRATEGY == 'tmean':
        aggregated_weights = _trimmed_mean(weights_list, f, trim_fraction=TMEAN_TRIM_FRACTION)

    elif AGGREGATION_STRATEGY == 'fedema':
        aggregated_weights = _fedema(weights_list, node_ids)

    elif AGGREGATION_STRATEGY == 'fltrust':
        fog_model_path  = FogResourcesPaths.FOG_MODEL_FILE_PATH.value
        _is_first_round = not os.path.exists(FLTRUST_ROUND_FLAG_PATH)

        w_prev = None
        if os.path.exists(fog_model_path):
            try:
                w_prev = _safe_load_model(fog_model_path).get_weights()
                _save_fltrust_prev_weights(w_prev)
                logger.info(f"FLTrust: loaded w_prev from fog model (round1={_is_first_round})")
            except Exception as _e:
                logger.warning(f"FLTrust: could not load fog model: {_e} β€” trying saved prev weights.")
                w_prev = _load_fltrust_prev_weights()
        else:
            logger.warning("FLTrust: fog model not found β€” trying saved prev weights.")
            w_prev = _load_fltrust_prev_weights()

        if _is_first_round:
            try:
                open(FLTRUST_ROUND_FLAG_PATH, 'w').close()
                logger.info(f"FLTrust: Round 1 β€” created round flag at {FLTRUST_ROUND_FLAG_PATH}")
            except Exception as _e:
                logger.warning(f"FLTrust: could not create round flag: {_e}")

        if w_prev is None:
            logger.warning("FLTrust: no w_prev available β€” FedAvg fallback.")
            aggregated_weights = _fedavg(weights_list, inv_mse_scores)
        elif FLTRUST_SERVER_NODE not in node_ids:
            logger.warning(f"FLTrust: server node '{FLTRUST_SERVER_NODE}' not found β€” FedAvg fallback.")
            aggregated_weights = _fedavg(weights_list, inv_mse_scores)
        else:
            server_idx    = node_ids.index(FLTRUST_SERVER_NODE)
            w_server_node = weights_list[server_idx]
            logger.info(f"FLTrust: server={FLTRUST_SERVER_NODE} (idx={server_idx}) | n={n} | f={f} | round1={_is_first_round}")
            aggregated_weights = _fltrust(weights_list, w_prev, w_server_node, f=f)

    elif AGGREGATION_STRATEGY == 'fltrust2':
        fog_model_path  = FogResourcesPaths.FOG_MODEL_FILE_PATH.value
        _is_first_round = not os.path.exists(FLTRUST_ROUND_FLAG_PATH)

        if _is_first_round:
            logger.info("FLTrust2: round 1 cold start -- using trimmed-mean fallback.")
            aggregated_weights = _trimmed_mean(weights_list, f, trim_fraction=TMEAN_TRIM_FRACTION)
            try:
                open(FLTRUST_ROUND_FLAG_PATH, 'w').close()
                logger.info(f"FLTrust2: Round 1 -- created round flag at {FLTRUST_ROUND_FLAG_PATH}")
            except Exception as _e:
                logger.warning(f"FLTrust2: could not create round flag: {_e}")
        else:
            w_prev = None
            if os.path.exists(fog_model_path):
                try:
                    w_prev = _safe_load_model(fog_model_path).get_weights()
                    _save_fltrust_prev_weights(w_prev)
                    logger.info(f"FLTrust2: loaded w_prev from fog model (round1={_is_first_round})")
                except Exception as _e:
                    logger.warning(f"FLTrust2: could not load fog model: {_e} -- trying saved prev weights.")
                    w_prev = _load_fltrust_prev_weights()
            else:
                logger.warning("FLTrust2: fog model not found -- trying saved prev weights.")
                w_prev = _load_fltrust_prev_weights()

            if w_prev is None:
                logger.warning("FLTrust2: no w_prev available -- FedAvg fallback.")
                aggregated_weights = _fedavg(weights_list, inv_mse_scores)
            else:
                logger.info("FLTrust2: training independent server model on fog...")
                w_server = fltrust2_train_server(fog_model_path)

                if w_server is None:
                    logger.warning("FLTrust2: server training failed -- FedAvg fallback.")
                    aggregated_weights = _fedavg(weights_list, inv_mse_scores)
                else:
                    logger.info(
                        f"FLTrust2: independent server ready | n={n} | f={f} | "
                        f"round1={_is_first_round}"
                    )
                    aggregated_weights = _fltrust(weights_list, w_prev, w_server, f=f)

    elif AGGREGATION_STRATEGY == 'fedconsensus':
        fog_model_path = FogResourcesPaths.FOG_MODEL_FILE_PATH.value
        w_prev_fc = None

        if os.path.exists(fog_model_path):
            try:
                w_prev_fc = _safe_load_model(fog_model_path).get_weights()
                logger.info("FedConsensus: loaded w_prev from fog model.")
            except Exception as _e:
                logger.warning(f"FedConsensus: could not load fog model: {_e}")

        if w_prev_fc is None:
            logger.warning("FedConsensus: no w_prev available β€” FedAvg fallback.")
            aggregated_weights = _fedavg(weights_list, inv_mse_scores)
        else:
            aggregated_weights = _fedconsensus(weights_list, node_ids, w_prev_fc)

    elif AGGREGATION_STRATEGY == 'fedconsensus_sv':
        fog_model_path = FogResourcesPaths.FOG_MODEL_FILE_PATH.value
        w_prev_fc = None

        if os.path.exists(fog_model_path):
            try:
                w_prev_fc = _safe_load_model(fog_model_path).get_weights()
                logger.info("FedConsensus-SV: loaded w_prev from fog model.")
            except Exception as _e:
                logger.warning(f"FedConsensus-SV: could not load fog model: {_e}")

        if w_prev_fc is None:
            logger.warning("FedConsensus-SV: no w_prev available β€” FedAvg fallback.")
            aggregated_weights = _fedavg(weights_list, inv_mse_scores)
        else:
            aggregated_weights = _fedconsensus_sv(weights_list, node_ids, w_prev_fc)

    elif AGGREGATION_STRATEGY == 'fedconsensus_p':
        fog_model_path = FogResourcesPaths.FOG_MODEL_FILE_PATH.value
        w_prev_fc = None

        if os.path.exists(fog_model_path):
            try:
                w_prev_fc = _safe_load_model(fog_model_path).get_weights()
                logger.info("FedConsensus-P: loaded global w_prev from fog model.")
            except Exception as _e:
                logger.warning(f"FedConsensus-P: could not load fog model: {_e}")

        if w_prev_fc is None:
            logger.warning("FedConsensus-P: no w_prev available β€” FedAvg fallback.")
            aggregated_weights = _fedavg(weights_list, inv_mse_scores)
        else:
            aggregated_weights = _fedconsensus_p(weights_list, node_ids, w_prev_fc)

    else:
        aggregated_weights = _fedavg(weights_list, inv_mse_scores)

    aggregation_time_s = round(time.perf_counter() - t_agg_start, 4)

    ram_after_agg = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    agg_ram_mb    = round(max(0.0, ram_after_agg - ram_before_agg), 2)

    stop_cpu_mon.set()
    cpu_thread.join()
    peak_ram_mb      = ram_monitor.stop()
    cpu_temp_after   = _read_cpu_temp()
    avg_cpu_usage    = round(sum(cpu_usage_samples) / len(cpu_usage_samples) if cpu_usage_samples else 0.0, 1)
    estimated_energy = _estimate_energy_wh(avg_cpu_usage, aggregation_time_s)

    template_model = _safe_load_model(loaded_paths[0])
    template_model.set_weights(aggregated_weights)
    template_model.save(FogResourcesPaths.FOG_MODEL_FILE_PATH.value, include_optimizer=False)
    delete_files_containing(FogResourcesPaths.MODELS_FOLDER_PATH.value, "edge", [".keras"])

    total_time_s = round(time.perf_counter() - t_total_start, 4)

    full_round_time_s = (
        round(time.perf_counter() - round_start_time, 4)
        if round_start_time is not None else None
    )
    wait_time_s = (
        round(t_receive_end - round_start_time, 4)
        if round_start_time is not None else None
    )

    logger.info(
        f"[AGG METRICS] strategy={AGGREGATION_STRATEGY.upper()} | "
        f"n_clients={n} | f={f} | "
        f"agg_time={aggregation_time_s}s | "
        f"total_time={total_time_s}s | "
        f"wait_time={wait_time_s}s | "
        f"full_round_time={full_round_time_s}s | "
        f"peak_ram={peak_ram_mb}MB | "
        f"agg_ram={agg_ram_mb}MB | "
        f"cpu_usage={avg_cpu_usage}% | "
        f"cpu_temp_before={cpu_temp_before}C | "
        f"cpu_temp_after={cpu_temp_after}C | "
        f"estimated_energy={estimated_energy}Wh"
    )

    return template_model
