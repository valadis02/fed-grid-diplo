"""
Ουρά του E1 (robustness) για το tools/offline_sim.py.

Core grid (paper, Πίνακας 5):
  diverse : 6 κανόνες × 6 επιθέσεις × {10,20,30,40}% + 6 runs χωρίς επίθεση, × seeds
  similar : μόνο 30% (6 × 6 + 6), × seeds      — έλεγχος ευαισθησίας στην ετερογένεια
Τα runs χωρίς επίθεση τρέχουν με --frac 30 (επηρεάζει μόνο το f του Krum / M-Krum).

  python tools/run_grid.py --seeds 1,2 --jobs 4            # τρέχει ό,τι λείπει
  python tools/run_grid.py --seeds 1,2 --list               # μόνο λίστα
  python tools/run_grid.py --seeds 3 --only krum,multi_krum # 3ο seed σε επιλεγμένους κανόνες
Τα runs που έχουν ήδη json παραλείπονται, άρα η ουρά συνεχίζει από όπου σταμάτησε.
"""
import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RULES = ["fedavg", "krum", "multi_krum", "tmean", "median", "fltrust2"]
ATTACKS = ["label_flipping", "sign_flipping", "gaussian_noise", "alie", "ipm", "minmax"]
FRACS = [10, 20, 30, 40]


def grid(seeds, only):
    rules = [r for r in RULES if not only or r in only]
    runs = []
    for seed in seeds:
        for data, fracs in (("lcl50_diverse", FRACS), ("lcl50_similar", [30])):
            runs += [(data, r, "none", 30, seed) for r in rules]
            runs += [(data, r, atk, f, seed) for r, atk, f in itertools.product(rules, ATTACKS, fracs)]
    return runs


def tag(data, rule, atk, frac, seed):
    return f"{rule}_{atk}_f{frac}_s{seed}_{data}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0, help="TF threads ανά run (0 = cores / jobs)")
    ap.add_argument("--only", default="", help="περιορισμός σε κανόνες, π.χ. krum,multi_krum")
    ap.add_argument("--out", default="results/offline")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    runs = grid([int(s) for s in a.seeds.split(",")], [x for x in a.only.split(",") if x])
    todo = [r for r in runs if not os.path.exists(os.path.join(ROOT, a.out, tag(*r) + ".json"))]
    print(f"{len(runs)} runs στο grid, {len(runs) - len(todo)} έτοιμα, {len(todo)} μένουν")
    if a.list:
        for r in todo:
            print(tag(*r))
        return

    threads = a.threads or max(1, (os.cpu_count() or 2) // a.jobs)
    env = dict(os.environ, TF_NUM_INTRAOP_THREADS=str(threads), TF_NUM_INTEROP_THREADS="1",
               OMP_NUM_THREADS=str(threads))
    os.makedirs(os.path.join(ROOT, a.out, "logs"), exist_ok=True)

    def run(r):
        data, rule, atk, frac, seed = r
        log = os.path.join(ROOT, a.out, "logs", tag(*r) + ".log")
        cmd = [sys.executable, os.path.join(ROOT, "tools", "offline_sim.py"), "--data", data, "--strategy", rule,
               "--attack", atk, "--frac", str(frac), "--seed", str(seed), "--out", os.path.join(ROOT, a.out)]
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        print(f"{'OK ' if rc == 0 else 'ERR'} {tag(*r)}", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run, todo))


if __name__ == "__main__":
    main()

    