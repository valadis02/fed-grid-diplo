"""
shared/crypto.py
----------------
AES-256-GCM κρυπτογράφηση/αποκρυπτογράφηση για μεταφορά βαρών μοντέλων.

Χρησιμοποιείται στο Πείραμα 4 για μέτρηση overhead κρυπτογράφησης
σε συσκευές διαφορετικών δυνατοτήτων (PC, RPi5, RPi4).

AES-256-GCM:
  - 256-bit κλειδί (32 bytes)
  - 96-bit IV/nonce (12 bytes) — τυχαίος για κάθε μήνυμα
  - 128-bit authentication tag (16 bytes)
  - Παρέχει και εμπιστευτικότητα ΚΑΙ integrity (authenticated encryption)

Payload μετά κρυπτογράφηση: [12 bytes IV] + [ciphertext] + [16 bytes tag]
Overhead vs plaintext: +28 bytes (αμελητέο για μοντέλα ~2MB)
"""

import os
import time
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from shared.logging_config import logger

# ---------------------------------------------------------------------------
# Σταθερές
# ---------------------------------------------------------------------------

AES_KEY_SIZE   = 32   # 256 bits
AES_NONCE_SIZE = 12   # 96 bits (recommended για GCM)
AES_TAG_SIZE   = 16   # 128 bits (default για GCM)

# Environment variable για το κλειδί (base64-encoded 32 bytes)
# Παράδειγμα δημιουργίας: python -c "import os,base64; print(base64.b64encode(os.urandom(32)).decode())"
_ENV_KEY_VAR = "AES_ENCRYPTION_KEY"


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def get_key() -> bytes:
    """
    Φορτώνει το AES-256 κλειδί από environment variable AES_ENCRYPTION_KEY.
    Αν δεν υπάρχει, χρησιμοποιεί default κλειδί (ΜΟΝΟ για testing).
    """
    key_b64 = os.getenv(_ENV_KEY_VAR)
    if key_b64:
        try:
            key = base64.b64decode(key_b64)
            if len(key) != AES_KEY_SIZE:
                raise ValueError(f"AES key must be {AES_KEY_SIZE} bytes, got {len(key)}")
            return key
        except Exception as e:
            logger.error("Crypto: invalid AES_ENCRYPTION_KEY: %s", e)
            raise

    # Default κλειδί για testing — ΜΗΝ χρησιμοποιείς σε production
    logger.warning(
        "Crypto: AES_ENCRYPTION_KEY not set — using default test key. "
        "Set AES_ENCRYPTION_KEY env var for real deployments."
    )
    return b'\x00' * AES_KEY_SIZE


# ---------------------------------------------------------------------------
# Κρυπτογράφηση / Αποκρυπτογράφηση
# ---------------------------------------------------------------------------

def encrypt(plaintext: bytes, key: bytes = None) -> tuple[bytes, dict]:
    """
    Κρυπτογραφεί plaintext με AES-256-GCM.

    Args:
        plaintext: τα bytes προς κρυπτογράφηση (π.χ. βάρη μοντέλου)
        key:       32-byte κλειδί (αν None, χρησιμοποιεί get_key())

    Returns:
        (ciphertext_with_iv_and_tag, metrics)
        όπου metrics = {
            'encrypt_time_s':    χρόνος κρυπτογράφησης σε δευτερόλεπτα,
            'plaintext_size_b':  μέγεθος plaintext σε bytes,
            'ciphertext_size_b': μέγεθος ciphertext (με IV+tag) σε bytes,
            'overhead_bytes':    επιπλέον bytes λόγω κρυπτογράφησης,
        }

    Format: [12 bytes nonce] + [len(plaintext) bytes ciphertext] + [16 bytes tag]
    """
    if key is None:
        key = get_key()

    t_start = time.perf_counter()

    nonce = os.urandom(AES_NONCE_SIZE)
    aesgcm = AESGCM(key)
    # encrypt() επιστρέφει ciphertext + tag μαζί (tag στο τέλος)
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    t_end = time.perf_counter()
    encrypt_time = t_end - t_start

    # Συνδυάζουμε: nonce + ciphertext+tag
    result = nonce + ciphertext_with_tag

    metrics = {
        'encrypt_time_s':    round(encrypt_time, 6),
        'plaintext_size_b':  len(plaintext),
        'ciphertext_size_b': len(result),
        'overhead_bytes':    len(result) - len(plaintext),  # = AES_NONCE_SIZE + AES_TAG_SIZE = 28
    }

    logger.debug(
        "Crypto: encrypted %d bytes → %d bytes in %.4fs (overhead: +%d bytes)",
        metrics['plaintext_size_b'],
        metrics['ciphertext_size_b'],
        metrics['encrypt_time_s'],
        metrics['overhead_bytes'],
    )

    return result, metrics


def decrypt(ciphertext_with_iv: bytes, key: bytes = None) -> tuple[bytes, dict]:
    """
    Αποκρυπτογραφεί με AES-256-GCM.

    Args:
        ciphertext_with_iv: [12 bytes nonce] + [ciphertext] + [16 bytes tag]
        key:                32-byte κλειδί (αν None, χρησιμοποιεί get_key())

    Returns:
        (plaintext, metrics)
        όπου metrics = {
            'decrypt_time_s':    χρόνος αποκρυπτογράφησης σε δευτερόλεπτα,
            'ciphertext_size_b': μέγεθος εισόδου σε bytes,
            'plaintext_size_b':  μέγεθος plaintext σε bytes,
        }

    Raises:
        cryptography.exceptions.InvalidTag: αν το μήνυμα έχει τροποποιηθεί
        ValueError: αν το ciphertext είναι πολύ μικρό
    """
    if key is None:
        key = get_key()

    if len(ciphertext_with_iv) < AES_NONCE_SIZE + AES_TAG_SIZE:
        raise ValueError(
            f"Crypto: ciphertext too short ({len(ciphertext_with_iv)} bytes), "
            f"minimum is {AES_NONCE_SIZE + AES_TAG_SIZE} bytes"
        )

    t_start = time.perf_counter()

    nonce           = ciphertext_with_iv[:AES_NONCE_SIZE]
    ciphertext_tag  = ciphertext_with_iv[AES_NONCE_SIZE:]

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext_tag, associated_data=None)

    t_end = time.perf_counter()
    decrypt_time = t_end - t_start

    metrics = {
        'decrypt_time_s':    round(decrypt_time, 6),
        'ciphertext_size_b': len(ciphertext_with_iv),
        'plaintext_size_b':  len(plaintext),
    }

    logger.debug(
        "Crypto: decrypted %d bytes → %d bytes in %.4fs",
        metrics['ciphertext_size_b'],
        metrics['plaintext_size_b'],
        metrics['decrypt_time_s'],
    )

    return plaintext, metrics


# ---------------------------------------------------------------------------
# Βοηθητικές συναρτήσεις για base64 (για JSON payloads)
# ---------------------------------------------------------------------------

def encrypt_to_b64(plaintext: bytes, key: bytes = None) -> tuple[str, dict]:
    """
    Κρυπτογραφεί και επιστρέφει base64 string (για JSON payloads).
    """
    ciphertext, metrics = encrypt(plaintext, key)
    return base64.b64encode(ciphertext).decode('utf-8'), metrics


def decrypt_from_b64(ciphertext_b64: str, key: bytes = None) -> tuple[bytes, dict]:
    """
    Αποκωδικοποιεί base64 και αποκρυπτογραφεί.
    """
    ciphertext = base64.b64decode(ciphertext_b64)
    return decrypt(ciphertext, key)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== AES-256-GCM Crypto Test ===")

    # Δοκιμή με τυχαία δεδομένα μεγέθους ~2MB (μέγεθος μοντέλου)
    test_sizes = [1024, 1024 * 100, 1024 * 1024, 2164 * 1024]  # 1KB, 100KB, 1MB, 2.1MB
    key = get_key()

    for size in test_sizes:
        data = os.urandom(size)
        enc_data, enc_metrics = encrypt(data, key)
        dec_data, dec_metrics = decrypt(enc_data, key)

        assert dec_data == data, "Decryption mismatch!"

        print(
            f"  Size: {size/1024:.1f} KB | "
            f"Encrypt: {enc_metrics['encrypt_time_s']*1000:.2f}ms | "
            f"Decrypt: {dec_metrics['decrypt_time_s']*1000:.2f}ms | "
            f"Overhead: +{enc_metrics['overhead_bytes']} bytes"
        )

    print("\nAll tests passed!")
    sys.exit(0)