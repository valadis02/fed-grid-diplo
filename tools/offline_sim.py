"""
Offline προσομοίωση του testbed για το E1 (robustness).

Ίδιος κώδικας με το testbed, χωρίς δίκτυο/containers:
  - δεδομένα, χαρακτηριστικά, κανονικοποίηση, LF : shared/forecasting.py
  - μοντέλο και κοινό w0                         : shared/model_architectures.create_model (MODEL_INIT_SEED)
  - επιθέσεις στο update (SF, GN)                : shared/attacks.attack_update
  - adaptive επιθέσεις (ALIE, IPM, Min-Max)      : fog/model/fog_attacks.apply_fog_attack
  - aggregators, ανά-γύρο CSV                    : fog/model/model_aggregation_service
  - FLTrust2 server training                     : fog/model/fltrust2_server_training
  - αξιολόγηση (feeder / fog / house)            : cloud/model/cloud_evaluation_service._evaluate_dayahead
Τοπολογία: 2 fogs × 25 σπίτια, cloud = mean(προηγούμενο cloud, fog1, fog2) — όπως στο testbed.

Διαφορές από το testbed (να δηλωθούν αν χρειαστεί):
  - seeds: tf.keras.utils.set_random_seed(seed·10⁶ + γύρος·10³ + σπίτι) πριν από κάθε τοπική εκπαίδευση
    (το testbed σπέρνει μία φορά ανά edge process). Ίδια κατανομή, όχι ίδια ακολουθία.
  - ο fog δεν «περιμένει» edges: όλοι οι clients συμμετέχουν σε κάθε γύρο.

Χρήση:
  python tools/offline_sim.py --data lcl50_diverse --strategy krum --attack alie --frac 30 --seed 1
Έξοδος: <out>/<tag>.json (μετρικές ανά γύρο) και <out>/agg_log.csv (ίδιο format με τον fog).
"""
import argparse
import json
import os
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="lcl50_similar")
    ap.add_argument("--strategy", required=True, choices=["fedavg", "krum", "multi_krum", "tmean", "median", "fltrust2"])
    ap.add_argument("--attack", default="none",
                    choices=["none", "label_flipping", "sign_flipping", "gaussian_noise", "alie", "ipm", "minmax"])
    ap.add_argument("--frac", type=int, default=30, choices=[10, 20, 30, 40])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--houses", type=int, default=50, help="2 fogs × houses/2")
    ap.add_argument("--zeta", type=float, default=2.6)
    ap.add_argument("--sigma", type=float, default=30.0)
    ap.add_argument("--ipm-eps", type=float, default=2.0)
    ap.add_argument("--minmax-perturb", default="std", choices=["std", "unit", "sign"])
    ap.add_argument("--tmean-beta", type=float, default=0.0, help="0 = κόβει f (πραγματικοί Byzantine του fog) από κάθε άκρο· >0 = σταθερό β (ablation)")
    ap.add_argument("--krum-f", type=int, default=-1, help="-1 = πλήθος επιτιθέμενων του fog για το --frac")
    ap.add_argument("--fltrust-epochs", type=int, default=5)
    ap.add_argument("--root-houses", default="", help="root household ανά fog (FLTrust2), π.χ. 19,45· κενό = medoid του fog για το dataset (DEFAULT_ROOTS, από tools/diag_root.py)")
    ap.add_argument("--train-date", default="2012-10-15")
    ap.add_argument("--train-days", type=int, default=382)
    ap.add_argument("--eval-days", type=int, default=92)
    ap.add_argument("--out", default="results/offline")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true", help="ξανατρέχει ακόμα κι αν υπάρχει το json")
    return ap.parse_args()


a = parse()
# Ρυθμίσεις που διαβάζουν τα modules του repo κατά το import — ίδιες με το gen_exp50.py
os.environ.update({
    "FORECAST_MODE": "dayahead", "NORMALIZATION": "per_house", "MODEL_INIT_SEED": str(a.seed),
    "TRAINING_DATE": a.train_date, "TRAIN_DAYS": str(a.train_days), "EVAL_DAYS": str(a.eval_days),
    "BATCH_SIZE": str(a.batch), "FLTRUST_EPOCHS": str(a.fltrust_epochs),
    "IPM_EPS": str(a.ipm_eps), "MINMAX_PERTURB": a.minmax_perturb,
    "FOG_GROUPS": f"1-{a.houses // 2},{a.houses // 2 + 1}-{a.houses}",
    "TF_CPP_MIN_LOG_LEVEL": "3", "AGG_LOG_PATH": os.path.join(a.out, "agg_log.csv"),
})
sys.path.insert(0, ROOT)

import logging                                                          # noqa: E402
import numpy as np                                                      # noqa: E402
import pandas as pd                                                     # noqa: E402
import tensorflow as tf                                                 # noqa: E402
from shared.logging_config import logger                                # noqa: E402
from shared import forecasting as fc                                    # noqa: E402
from shared.model_architectures import create_model                     # noqa: E402
from shared.attacks import attack_update, EDGE_UPDATE_ATTACKS           # noqa: E402
from shared.attackers import attackers_for, PER_FOG                     # noqa: E402
import fog.model.model_aggregation_service as agg                       # noqa: E402
import fog.model.fltrust2_server_training as fst                        # noqa: E402
from fog.model.fog_attacks import apply_fog_attack                      # noqa: E402
from cloud.model.cloud_evaluation_service import _evaluate_dayahead     # noqa: E402

logger.setLevel(logging.WARNING)
if os.getenv("TF_NUM_INTRAOP_THREADS"):
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_NUM_INTRAOP_THREADS"]))
    tf.config.threading.set_inter_op_parallelism_threads(1)
FOG_ATTACKS = ("alie", "ipm", "minmax")
# Medoid κάθε fog (μέση cosine των updates, γύροι 1-3, χωρίς επίθεση), εκτός επιτιθέμενων σε όλα τα ποσοστά
DEFAULT_ROOTS = {"lcl50_similar": "9,38", "lcl50_diverse": "6,34"}


def load_house(i):
    """Κρατά μόνο τη σειρά του train (μικρή) και τα eval arrays. Τα train arrays (~75 MB/σπίτι)
    χτίζονται τη στιγμή της εκπαίδευσης, ώστε η μνήμη να μην κλιμακώνεται με τα 50 σπίτια."""
    path = os.path.join(ROOT, a.data, f"house_{i}", "input_data.csv")
    t0, t1, e1 = fc.windows(a.train_date, a.train_days, a.eval_days)
    tr = fc.load_series(path, t0, t1)
    ev = fc.load_series(path, t1 - pd.Timedelta(days=fc.HISTORY_DAYS), e1)
    stats = fc.fit_stats(tr)
    A_ev = fc.build_arrays(ev, stats, target_start=t1, target_end=e1, issue_at_midnight=True)
    A_ev.pop("ts", None)
    return tr, stats, A_ev


def train_arrays(i):
    tr, stats, _ = data[i]
    A = fc.build_arrays(tr, stats)
    if i in ATT and a.attack == "label_flipping":
        fc.label_flip_targets(A, tr)
    return A


def local_train(model, w_global, A, seed):
    import gc
    tf.keras.utils.set_random_seed(seed)
    model.set_weights(w_global)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")   # νέος Adam κάθε γύρο, όπως στο edge
    model.fit([A["Xh"], A["Xf"]], A["y"], epochs=a.epochs, batch_size=a.batch, shuffle=True, verbose=0)
    w = model.get_weights()
    gc.collect()
    return w


def fog_aggregate(k, wl, ids, w_prev, model, tmp_model_path, rnd):
    n = len(wl)
    f = a.krum_f if a.krum_f >= 0 else PER_FOG[a.frac][k - 1]
    agg._AGG_INFO.clear()
    agg.FOG_ATTACKERS = [x for x in ids if int(x.rsplit("_", 1)[1]) in ATT]
    agg.FOG_NAME_ENV, agg.RUN_TAG, agg._ROUND_CTR[0] = f"FOG_NODE_{k}", TAG, rnd
    agg.AGGREGATION_STRATEGY = a.strategy

    if a.attack in FOG_ATTACKS:
        wl = apply_fog_attack(wl, ids, w_prev, attack_type=a.attack, attackers=agg.FOG_ATTACKERS)

    t0 = time.perf_counter()
    s = a.strategy
    if s == "fedavg":
        out = agg._fedavg(wl, [1.0] * n)
    elif s == "krum":
        out = agg._krum(wl, f)
    elif s == "multi_krum":
        out = agg._multi_krum(wl, f, m=max(1, n - f))
    elif s == "tmean":
        out = agg._trimmed_mean(wl, f, trim_fraction=a.tmean_beta)
    elif s == "median":
        out = agg._median(wl)
    elif s == "fltrust2":
        fst.FLTRUST_ROOT_DATA_PATH = os.path.join(ROOT, a.data, f"house_{ROOTS[k - 1]}", "input_data.csv")
        model.set_weights(w_prev)
        model.save(tmp_model_path)
        tf.keras.utils.set_random_seed(a.seed * 10**6 + rnd * 10**3 + 900 + k)
        w_srv = fst._train_server_dayahead(tmp_model_path)
        out = agg._fltrust(wl, w_prev, w_srv) if w_srv is not None else agg._fedavg(wl, [1.0] * n)
    agg._write_agg_log(ids, n, f, round(time.perf_counter() - t0, 4), "", "", "")
    return [np.asarray(w, dtype=np.float32) for w in out]


def main():
    global ATT, TAG, ROOTS
    ATT = [] if a.attack == "none" else [i for i in attackers_for(a.frac) if i <= a.houses]
    dname = os.path.basename(a.data.rstrip('/'))
    roots = a.root_houses or DEFAULT_ROOTS[dname]
    ROOTS = [int(x) for x in roots.split(",")]
    rsuf = f"_root{roots.replace(',', '-')}" if (a.strategy == "fltrust2" and a.root_houses and a.root_houses != DEFAULT_ROOTS[dname]) else ""
    beta = f"_b{a.tmean_beta:g}" if (a.strategy == "tmean" and a.tmean_beta > 0) else ""
    TAG = a.tag or f"{a.strategy}_{a.attack}_f{a.frac}{beta}{rsuf}_s{a.seed}_{dname}"
    os.makedirs(a.out, exist_ok=True)
    out_json = os.path.join(a.out, f"{TAG}.json")
    if os.path.exists(out_json) and not a.force:
        print(f"[{TAG}] υπάρχει ήδη — skip"); return

    t_start = time.time()
    H = list(range(1, a.houses + 1))
    global data
    data = {i: load_house(i) for i in H}
    houses_eval = [(i, f"house_{i}", data[i][2]) for i in H]
    fogs = {1: H[:a.houses // 2], 2: H[a.houses // 2:]}

    model = create_model(os.getenv("MODEL_ARCHITECTURE", "simple_lstm_two_gates"))
    w_glob = [w.copy() for w in model.get_weights()]          # κοινό seeded w0
    tmp_model_path = os.path.join(tempfile.mkdtemp(), "ref.keras")
    hist = []
    for r in range(1, a.rounds + 1):
        fog_models = []
        for k, members in fogs.items():
            wl, ids = [], []
            for i in members:
                A = train_arrays(i)
                w = local_train(model, w_glob, A, a.seed * 10**6 + r * 10**3 + i)
                del A
                if i in ATT and a.attack in EDGE_UPDATE_ATTACKS:
                    np.random.seed(a.seed * 10**6 + r * 10**3 + 500 + i)
                    w = attack_update(w, w_glob, a.attack, a.zeta, a.sigma)
                wl.append(w); ids.append(f"edge_node_{i}")
            fog_models.append(fog_aggregate(k, wl, ids, w_glob, model, tmp_model_path, r))
        w_glob = [np.mean([g, f1, f2], axis=0).astype(np.float32)
                  for g, f1, f2 in zip(w_glob, fog_models[0], fog_models[1])]
        model.set_weights(w_glob)
        ev = _evaluate_dayahead(model, houses_eval)
        ev.pop("per_house", None)
        ev["round"] = r
        hist.append(ev)
        print(f"[{TAG}] r{r} feeder nMAE={ev['feeder']['nmae'] * 100:.2f}% R2={ev['feeder']['r2']:.3f} "
              f"house_med={ev['house_nmae_median'] * 100:.1f}% ({time.time() - t_start:.0f}s)", flush=True)

    json.dump(dict(tag=TAG, args=vars(a), attackers=ATT, rounds=hist, secs=round(time.time() - t_start, 1)),
              open(out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

    