"""
shared/crypto_ckks.py
---------------------
CKKS Homomorphic Encryption για μεταφορά βαρών μοντέλων σε Federated Learning.

Διαφορά από AES (crypto.py):
  - AES:  edge encrypt → fog DECRYPT → FedAvg → fog encrypt → cloud decrypt
  - CKKS: edge encrypt → fog FedAvg ΣΤΟ CIPHERTEXT → cloud decrypt
  Ο fog δεν βλέπει ΠΟΤΕ τα βάρη σε plaintext.

Serialization: JSON + base64 (cross-platform, αντί για pickle)
"""

import os
import time
import base64
import json
import numpy as np
from typing import List, Tuple

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False

from shared.logging_config import logger

# ---------------------------------------------------------------------------
# Σταθερές
# ---------------------------------------------------------------------------

POLY_MOD_DEGREE     = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
MAX_VECTOR_SIZE     = 4096
SCALE               = 2 ** 40

CKKS_KEYS_DIR   = "/app/shared/ckks_keys"
PUBLIC_CTX_PATH = f"{CKKS_KEYS_DIR}/ckks_public_context.bin"
SECRET_CTX_PATH = f"{CKKS_KEYS_DIR}/ckks_secret_context.bin"

# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

def create_context() -> bytes:
    """Δημιουργεί νέο TenSEAL CKKS context με secret key."""
    if not TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL not installed.")

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.generate_galois_keys()
    ctx.global_scale = SCALE

    ctx_bytes = ctx.serialize(save_secret_key=True)
    logger.info(
        "CKKS: created new context | poly_mod=%d | scale=2^40 | size=%.1fKB",
        POLY_MOD_DEGREE, len(ctx_bytes) / 1024
    )
    return ctx_bytes


def get_public_context(ctx_bytes: bytes) -> bytes:
    """Εξάγει το public context (χωρίς secret key) για διανομή σε edge/fog."""
    if not TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL not installed.")
    ctx = ts.context_from(ctx_bytes)
    pub_ctx_bytes = ctx.serialize(save_secret_key=False)
    logger.info("CKKS: exported public context | size=%.1fKB", len(pub_ctx_bytes) / 1024)
    return pub_ctx_bytes


def load_context(ctx_bytes: bytes) -> "ts.Context":
    """Φορτώνει TenSEAL context από bytes."""
    if not TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL not installed.")
    return ts.context_from(ctx_bytes)


def load_public_context_from_file() -> bytes:
    """Φορτώνει το public context από το προκαθορισμένο path."""
    with open(PUBLIC_CTX_PATH, "rb") as f:
        return f.read()


def load_secret_context_from_file() -> bytes:
    """Φορτώνει το secret context από το προκαθορισμένο path (cloud only)."""
    with open(SECRET_CTX_PATH, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Weights → flat vector → chunks
# ---------------------------------------------------------------------------

def _flatten_weights(weights: List[np.ndarray]) -> Tuple[np.ndarray, List[dict]]:
    meta = []
    parts = []
    for arr in weights:
        meta.append({
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size":  int(arr.size),
        })
        parts.append(arr.flatten().astype(np.float32))
    flat = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return flat, meta


def _reconstruct_weights(flat: np.ndarray, meta: List[dict]) -> List[np.ndarray]:
    weights = []
    offset = 0
    for m in meta:
        size = m["size"]
        arr = flat[offset:offset + size].reshape(m["shape"]).astype(m["dtype"])
        weights.append(arr)
        offset += size
    return weights


def _chunk_vector(flat: np.ndarray, chunk_size: int = MAX_VECTOR_SIZE) -> List[np.ndarray]:
    return [flat[i:i + chunk_size] for i in range(0, len(flat), chunk_size)]


# ---------------------------------------------------------------------------
# Κρυπτογράφηση
# ---------------------------------------------------------------------------

def encrypt_weights(
    weights: List[np.ndarray],
    ctx_bytes: bytes,
) -> Tuple[bytes, dict]:
    """Κρυπτογραφεί λίστα numpy arrays με CKKS.

    Χρησιμοποιεί JSON + base64 για cross-platform serialization
    (αντί για pickle που είναι platform-dependent).
    """
    if not TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL not installed.")

    t_start = time.perf_counter()

    t_ctx_start = time.perf_counter()
    ctx = load_context(ctx_bytes)
    t_ctx_load = time.perf_counter() - t_ctx_start

    flat, shape_meta = _flatten_weights(weights)
    total_params = len(flat)

    chunks = _chunk_vector(flat)

    t_enc_start = time.perf_counter()
    enc_chunks = []
    for chunk in chunks:
        vec = ts.ckks_vector(ctx, chunk.tolist())
        enc_chunks.append(base64.b64encode(vec.serialize()).decode("utf-8"))
    t_encrypt = time.perf_counter() - t_enc_start

    t_serial_start = time.perf_counter()
    payload = {
        "enc_chunks":   enc_chunks,
        "shape_meta":   shape_meta,
        "total_params": total_params,
        "chunk_size":   MAX_VECTOR_SIZE,
        "n_chunks":     len(enc_chunks),
    }
    # JSON encoding — cross-platform (αντί για pickle)
    payload_bytes = json.dumps(payload).encode("utf-8")
    t_serial = time.perf_counter() - t_serial_start

    metrics = {
        "ctx_load_time_s":   round(t_ctx_load, 4),
        "encrypt_time_s":    round(t_encrypt, 4),
        "serialize_time_s":  round(t_serial, 4),
        "total_time_s":      round(t_ctx_load + t_encrypt + t_serial, 4),
        "total_params":      total_params,
        "n_chunks":          len(enc_chunks),
        "plaintext_size_kb": round(flat.nbytes / 1024, 1),
        "payload_size_kb":   round(len(payload_bytes) / 1024, 1),
        "overhead_x":        round(len(payload_bytes) / max(flat.nbytes, 1), 1),
    }

    logger.info(
        "CKKS: encrypted %d params in %d chunks | "
        "ctx_load=%.3fs | encrypt=%.2fs | serialize=%.3fs | "
        "total=%.2fs | payload=%.1fKB (%.0fx overhead)",
        total_params, len(enc_chunks),
        metrics["ctx_load_time_s"], metrics["encrypt_time_s"],
        metrics["serialize_time_s"], metrics["total_time_s"],
        metrics["payload_size_kb"], metrics["overhead_x"],
    )

    return payload_bytes, metrics


def encrypt_weights_to_b64(
    weights: List[np.ndarray],
    ctx_bytes: bytes,
) -> Tuple[str, dict]:
    """Κρυπτογραφεί και επιστρέφει base64 string."""
    payload_bytes, metrics = encrypt_weights(weights, ctx_bytes)
    return base64.b64encode(payload_bytes).decode("utf-8"), metrics


# ---------------------------------------------------------------------------
# HE Aggregation (fog - χωρίς αποκρυπτογράφηση)
# ---------------------------------------------------------------------------

def he_aggregate(
    payload_bytes_list: List[bytes],
    ctx_bytes: bytes,
    sample_counts: List[int] = None,
) -> Tuple[bytes, dict]:
    """FedAvg aggregation πάνω σε CKKS-κρυπτογραφημένα βάρη."""
    if not TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL not installed.")
    if not payload_bytes_list:
        raise ValueError("CKKS HE aggregate: empty payload list")

    t_start = time.perf_counter()
    n_clients = len(payload_bytes_list)

    if sample_counts is None:
        weights = [1.0 / n_clients] * n_clients
    else:
        total = sum(sample_counts)
        weights = [s / total for s in sample_counts]

    t_ctx_start = time.perf_counter()
    ctx = load_context(ctx_bytes)
    t_ctx_load = time.perf_counter() - t_ctx_start

    # JSON decoding — cross-platform
    payloads = [json.loads(p.decode("utf-8")) for p in payload_bytes_list]

    n_chunks = payloads[0]["n_chunks"]
    shape_meta = payloads[0]["shape_meta"]
    for i, p in enumerate(payloads):
        if p["n_chunks"] != n_chunks:
            raise ValueError(
                f"CKKS HE aggregate: chunk count mismatch "
                f"(client 0: {n_chunks}, client {i}: {p['n_chunks']})"
            )

    t_he_start = time.perf_counter()
    agg_chunks = []
    for chunk_idx in range(n_chunks):
        enc_bytes_0 = base64.b64decode(payloads[0]["enc_chunks"][chunk_idx])
        agg_vec = ts.ckks_vector_from(ctx, enc_bytes_0)
        agg_vec *= weights[0]

        for client_idx in range(1, n_clients):
            enc_bytes_i = base64.b64decode(
                payloads[client_idx]["enc_chunks"][chunk_idx]
            )
            vec_i = ts.ckks_vector_from(ctx, enc_bytes_i)
            vec_i *= weights[client_idx]
            agg_vec += vec_i

        agg_chunks.append(base64.b64encode(agg_vec.serialize()).decode("utf-8"))

    t_he = time.perf_counter() - t_he_start
    t_aggregate = time.perf_counter() - t_start

    result_payload = {
        "enc_chunks":   agg_chunks,
        "shape_meta":   shape_meta,
        "total_params": payloads[0]["total_params"],
        "chunk_size":   payloads[0]["chunk_size"],
        "n_chunks":     n_chunks,
        "aggregated":   True,
        "n_clients":    n_clients,
        "weights":      weights,
    }
    # JSON encoding — cross-platform
    result_bytes = json.dumps(result_payload).encode("utf-8")

    metrics = {
        "ctx_load_time_s":  round(t_ctx_load, 4),
        "he_ops_time_s":    round(t_he, 4),
        "aggregate_time_s": round(t_aggregate, 4),
        "n_clients":        n_clients,
        "n_chunks":         n_chunks,
        "result_size_kb":   round(len(result_bytes) / 1024, 1),
    }

    logger.info(
        "CKKS HE aggregate: %d clients | %d chunks | "
        "ctx_load=%.3fs | he_ops=%.2fs | total=%.2fs | size=%.1fKB",
        n_clients, n_chunks,
        metrics["ctx_load_time_s"], metrics["he_ops_time_s"],
        metrics["aggregate_time_s"], metrics["result_size_kb"],
    )

    return result_bytes, metrics


# ---------------------------------------------------------------------------
# Αποκρυπτογράφηση (cloud only)
# ---------------------------------------------------------------------------

def decrypt_weights(
    payload_bytes: bytes,
    ctx_bytes: bytes,
) -> Tuple[List[np.ndarray], dict]:
    """Αποκρυπτογραφεί CKKS payload — καλείται ΜΟΝΟ από το cloud."""
    if not TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL not installed.")

    t_start = time.perf_counter()

    t_ctx_start = time.perf_counter()
    ctx = load_context(ctx_bytes)
    t_ctx_load = time.perf_counter() - t_ctx_start

    # JSON decoding — cross-platform
    payload = json.loads(payload_bytes.decode("utf-8"))
    shape_meta   = payload["shape_meta"]
    n_chunks     = payload["n_chunks"]
    total_params = payload["total_params"]

    t_dec_start = time.perf_counter()
    flat_parts = []
    for chunk_idx in range(n_chunks):
        enc_bytes = base64.b64decode(payload["enc_chunks"][chunk_idx])
        vec = ts.ckks_vector_from(ctx, enc_bytes)
        decrypted = vec.decrypt()
        flat_parts.extend(decrypted)
    t_dec = time.perf_counter() - t_dec_start

    flat = np.array(flat_parts[:total_params], dtype=np.float32)
    weights = _reconstruct_weights(flat, shape_meta)

    t_total = time.perf_counter() - t_start

    metrics = {
        "ctx_load_time_s": round(t_ctx_load, 4),
        "decrypt_time_s":  round(t_dec, 4),
        "total_time_s":    round(t_total, 4),
        "total_params":    total_params,
        "n_chunks":        n_chunks,
        "n_layers":        len(weights),
    }

    logger.info(
        "CKKS: decrypted %d params (%d layers) | "
        "ctx_load=%.3fs | decrypt=%.2fs | total=%.2fs",
        total_params, len(weights),
        metrics["ctx_load_time_s"], metrics["decrypt_time_s"], metrics["total_time_s"],
    )

    return weights, metrics


def decrypt_weights_from_b64(
    payload_b64: str,
    ctx_bytes: bytes,
) -> Tuple[List[np.ndarray], dict]:
    """Αποκρυπτογράφηση από base64 string."""
    payload_bytes = base64.b64decode(payload_b64)
    return decrypt_weights(payload_bytes, ctx_bytes)