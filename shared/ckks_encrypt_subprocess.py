"""
shared/ckks_encrypt_subprocess.py
----------------------------------
Εκτελείται ως ξεχωριστό subprocess για να αποφευχθεί το
TensorFlow + TenSEAL memory conflict.

Χρήση:
  python ckks_encrypt_subprocess.py <weights_path> <output_path>

Επιστρέφει:
  - Γράφει το encrypted payload στο output_path
  - Γράφει JSON metrics στο stdout
  - Exit code 0 = επιτυχία, 1 = σφάλμα
"""
import sys
import os
import json
import time
import base64

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": f"Usage: {sys.argv[0]} <weights_path> <output_path>"}))
        sys.exit(1)

    weights_path = sys.argv[1]
    output_path  = sys.argv[2]

    # Imports εδώ — ΧΩΡΙΣ TensorFlow
    import numpy as np
    from shared.crypto_ckks import encrypt_weights_to_b64, load_public_context_from_file

    t_total_start = time.perf_counter()

    # 1. Φόρτωσε public context από hardcoded path
    t_ctx_start = time.perf_counter()
    pub_ctx_bytes = load_public_context_from_file()
    ctx_load_time_s = round(time.perf_counter() - t_ctx_start, 4)

    # 2. Φόρτωσε weights από .npz (χωρίς TF)
    data = np.load(weights_path, allow_pickle=True)
    weights = [data[f"arr_{i}"] for i in range(len(data.files))]

    # 3. CKKS κρυπτογράφηση
    t_enc_start = time.perf_counter()
    payload_b64, enc_metrics = encrypt_weights_to_b64(weights, pub_ctx_bytes)
    encrypt_time_s = round(time.perf_counter() - t_enc_start, 4)

    # 4. Γράψε payload στο output file
    payload_bytes = base64.b64decode(payload_b64)
    with open(output_path, "wb") as f:
        f.write(payload_bytes)

    total_time_s = round(time.perf_counter() - t_total_start, 4)

    # 5. Επέστρεψε metrics στο stdout
    result = {
        "ctx_load_time_s": ctx_load_time_s,
        "encrypt_time_s":  encrypt_time_s,
        "total_time_s":    total_time_s,
        "total_params":    enc_metrics["total_params"],
        "payload_size_kb": enc_metrics["payload_size_kb"],
        "overhead_x":      enc_metrics["overhead_x"],
        "output_path":     output_path,
    }
    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()