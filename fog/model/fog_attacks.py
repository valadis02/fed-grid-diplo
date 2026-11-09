"""
Adaptive (defence-aware) Byzantine attacks, simulated at the fog node.

Μοντέλο απειλής: omniscient, συνεργαζόμενοι επιτιθέμενοι. Οι επιτιθέμενοι edges
εκπαιδεύονται κανονικά (ATTACK_TYPE=none στο edge). Ο fog, ΠΡΙΝ το aggregation,
υπολογίζει τα updates g_i = w_i - w_prev όλων των clients, κατασκευάζει ένα κοινό
κακόβουλο update g̃ και αντικαθιστά με αυτό τα updates των επιτιθέμενων.
(Paper: «η επίθεση εφαρμόζεται στον fog για να μοντελοποιήσει συνεργαζόμενους
επιτιθέμενους που βλέπουν τα τίμια updates».)

Τα μ, σ υπολογίζονται πάνω σε ΟΛΑ τα (τίμια, πριν την επίθεση) updates του fog —
και οι επιτιθέμενοι έχουν εκπαιδευτεί τίμια, άρα αυτό είναι το omniscient σενάριο.

Env:
  FOG_ATTACK_TYPE   none | alie | ipm | minmax
  FOG_ATTACKERS     λίστα edge names, π.χ. "edge_node_2,edge_node_3"
  FOG_ATTACK_F      f για το z_max του ALIE (default: πλήθος επιτιθέμενων στον fog)
  ALIE_Z            αν δοθεί, αντικαθιστά το z_max
  IPM_EPS           ε του IPM (default 2.0)
  MINMAX_PERTURB    std | unit | sign   (κατεύθυνση ∇p, default std)
"""
import os
import time
import numpy as np
from scipy.stats import norm
from shared.logging_config import logger

FOG_ATTACK_TYPE = os.getenv('FOG_ATTACK_TYPE', 'none').lower()
FOG_ATTACKERS   = [x.strip() for x in os.getenv('FOG_ATTACKERS', '').split(',') if x.strip()]
FOG_ATTACK_F    = int(os.getenv('FOG_ATTACK_F', '0'))
ALIE_Z          = os.getenv('ALIE_Z', '')
IPM_EPS         = float(os.getenv('IPM_EPS', '2.0'))
MINMAX_PERTURB  = os.getenv('MINMAX_PERTURB', 'std').lower()


def _flatten(wl: list) -> np.ndarray:
    return np.concatenate([w.ravel().astype(np.float64) for w in wl])


def _unflatten(vec: np.ndarray, template: list) -> list:
    out, k = [], 0
    for w in template:
        out.append(vec[k:k + w.size].reshape(w.shape).astype(w.dtype))
        k += w.size
    return out


def alie_z(n: int, f: int) -> float:
    """z_max του Baruch et al. (2019): s = floor(n/2 + 1) - f, z = Φ^{-1}((n - s) / n)."""
    s = int(np.floor(n / 2 + 1)) - f
    p = min((n - s) / n, (n - 0.5) / n)      # s <= 0 μόνο για f >= n/2 (εκτός του grid): αποφυγή z = inf
    return float(norm.ppf(p))


def _alie(G: np.ndarray, f: int) -> tuple:
    mu, sd = G.mean(0), G.std(0)
    z = float(ALIE_Z) if ALIE_Z else alie_z(G.shape[0], f)
    return mu - z * sd, {'z': round(z, 4)}


def _ipm(G: np.ndarray) -> tuple:
    return -IPM_EPS * G.mean(0), {'eps': IPM_EPS}


def _sq_dists_to(G: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sum((G - x) ** 2, axis=1)


def _minmax(G: np.ndarray) -> tuple:
    """
    Min-Max (Shejwalkar & Houmansadr, NDSS 2021), AGR-agnostic:
    g̃ = μ + γ ∇p με το μεγαλύτερο γ ώστε  max_i ||g̃ - g_i|| <= max_{i,j} ||g_i - g_j||.
    """
    mu, sd = G.mean(0), G.std(0)
    if MINMAX_PERTURB == 'unit':
        p = -mu / (np.linalg.norm(mu) + 1e-12)
    elif MINMAX_PERTURB == 'sign':
        p = -np.sign(mu)
    else:  # 'std'
        p = -sd / (np.linalg.norm(sd) + 1e-12)

    sq = np.sum(G ** 2, axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (G @ G.T), 0.0)
    thr = D2.max()

    ok = lambda g: _sq_dists_to(G, mu + g * p).max() <= thr
    lo, hi = 0.0, 1.0
    while ok(hi) and hi < 1e8:          # βρες άνω φράγμα
        lo, hi = hi, hi * 2.0
    for _ in range(60):                 # διχοτόμηση
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return mu + lo * p, {'gamma': float(f'{lo:.6g}'), 'perturb': MINMAX_PERTURB}


def apply_fog_attack(weights_list: list, node_ids: list, w_prev: list | None,
                     attack_type: str | None = None, attackers: list | None = None, f_override: int | None = None) -> list:
    """Επιστρέφει νέα weights_list με τα updates των επιτιθέμενων αντικατεστημένα.
    attack_type / attackers / f_override: για το offline_sim (αλλιώς από env)."""
    atype = (attack_type or FOG_ATTACK_TYPE).lower()
    attackers = FOG_ATTACKERS if attackers is None else attackers
    if atype == 'none' or not attackers:
        return weights_list
    atk_idx = [i for i, nid in enumerate(node_ids) if nid in attackers]
    if not atk_idx:
        logger.info(f"FogAttack[{atype}]: κανένας επιτιθέμενος σε αυτόν τον fog.")
        return weights_list
    if w_prev is None:
        logger.warning(f"FogAttack[{atype}]: δεν υπάρχει w_prev — η επίθεση ΔΕΝ εφαρμόστηκε.")
        return weights_list

    t0 = time.perf_counter()
    base = _flatten(w_prev)
    G = np.stack([_flatten(wl) - base for wl in weights_list])
    f = f_override if f_override else (FOG_ATTACK_F if FOG_ATTACK_F > 0 else len(atk_idx))

    if atype == 'alie':
        g_mal, info = _alie(G, f)
    elif atype == 'ipm':
        g_mal, info = _ipm(G)
    elif atype == 'minmax':
        g_mal, info = _minmax(G)
    else:
        logger.warning(f"FogAttack: άγνωστος τύπος '{atype}' — καμία επίθεση.")
        return weights_list

    w_mal = _unflatten(base + g_mal, weights_list[0])
    out = list(weights_list)
    for i in atk_idx:
        out[i] = [w.copy() for w in w_mal]

    mu_norm = float(np.linalg.norm(G.mean(0)))
    logger.warning(
        f"FogAttack[{atype.upper()}]: {len(atk_idx)}/{len(node_ids)} updates αντικαταστάθηκαν "
        f"({[node_ids[i] for i in atk_idx]}) | f={f} | {info} | "
        f"||g_mal||={np.linalg.norm(g_mal):.4g} ||mu||={mu_norm:.4g} | "
        f"{time.perf_counter() - t0:.3f}s"
    )
    return out

