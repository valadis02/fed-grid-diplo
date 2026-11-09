"""
Επιθέσεις πάνω στο update που εκτελεί ο ίδιος ο (κακόβουλος) client.
Κοινός κώδικας για το edge (testbed) και το tools/offline_sim.py.
Οι adaptive επιθέσεις (ALIE / IPM / Min-Max) είναι στο fog/model/fog_attacks.py.
"""
import numpy as np

EDGE_UPDATE_ATTACKS = ("sign_flipping", "sign_flipping_weights", "gaussian_noise")


def attack_update(weights: list, w_t: list, attack: str, zeta: float, sigma: float) -> list:
    """
    weights : w_i μετά την τοπική εκπαίδευση,  w_t : global από το οποίο ξεκίνησε ο γύρος.
    - sign_flipping         : g̃ = −ζ·g          → w̃ = w_t − ζ·(w_i − w_t)
    - sign_flipping_weights : w̃ = −w            (εκδοχή διπλωματικής)
    - gaussian_noise        : g̃ = g + N(0, σ²I) → w̃ = w_i + N(0, σ²I)   (χρησιμοποιεί το np.random)
    """
    if attack == "sign_flipping":
        return [(wt - zeta * (wi - wt)).astype(wi.dtype) for wi, wt in zip(weights, w_t)]
    if attack == "sign_flipping_weights":
        return [-w for w in weights]
    if attack == "gaussian_noise":
        return [w + np.random.normal(0, sigma, w.shape).astype(w.dtype) for w in weights]
    raise ValueError(f"unknown update attack '{attack}'")

