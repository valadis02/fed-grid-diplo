"""
Ουρά του E1 (robustness) για το tools/offline_sim.py.

Core grid (paper, Πίνακας 5):
  similar : 6 κανόνες × 6 επιθέσεις × {10,20,30,40}% + 6 runs χωρίς επίθεση, × seeds   (σπίτια ενός cluster)
  (diverse: εκτός grid προς το παρόν)
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


def grid(seeds, only, fracs=FRACS, sets=("lcl50_similar", "lcl50_diverse")):
    rules = [r for r in RULES if not only or r in only]
    runs = []
    for seed in seeds:
        for data, fr in (("lcl50_similar", fracs),):
            if data not in sets:
                continue
            if 30 in fr:
                runs += [(data, r, "none", 30, seed) for r in rules]
            runs += [(data, r, atk, f, seed) for r, atk, f in itertools.product(rules, ATTACKS, [f for f in fr if f in FRACS])]
    # Ablation: Trimmed Mean με σταθερό β = 0.2 (f άγνωστο) μόνο στο 40%
    if not only or "tmean" in only:
        runs += [(data, "tmean", atk, 40, seed, 0.2) for seed in seeds for atk in ATTACKS for data in sets if 40 in fracs]
    return runs


def tag(data, rule, atk, frac, seed, beta=0):
    b = f"_b{beta:g}" if beta else ""
    return f"{rule}_{atk}_f{frac}{b}_s{seed}_{data}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0, help="TF threads ανά run (0 = cores / jobs)")
    ap.add_argument("--only", default="", help="περιορισμός σε κανόνες, π.χ. krum,multi_krum")
    ap.add_argument("--fracs", default="10,20,30,40", help="ποσοστά για το similar, π.χ. 10,20,40")
    ap.add_argument("--sets", default="lcl50_similar")
    ap.add_argument("--out", default="results/offline")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    runs = grid([int(s) for s in a.seeds.split(",")], [x for x in a.only.split(",") if x],
                [int(f) for f in a.fracs.split(",")], a.sets.split(","))
    todo = [r for r in runs if not os.path.exists(os.path.join(ROOT, a.out, tag(*r) + ".json"))]
    print(f"{len(runs)} runs στο grid, {len(runs) - len(todo)} έτοιμα, {len(todo)} μένουν")
    if a.list:
        for r in todo:
            print(tag(*r))
        return

    # Preflight: πακέτα και δεδομένα, ώστε να μη «σκάσουν» όλα τα runs με το ίδιο σφάλμα
    chk = subprocess.run([sys.executable, "-c", "import tensorflow, pandas, sklearn, scipy, psutil"],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        sys.exit("Λείπουν πακέτα στο Python που τρέχει το run_grid:\n" + chk.stderr.strip().splitlines()[-1]
                 + "\nΔες το README του tools/ (pip install tensorflow==2.20.0 pandas scikit-learn scipy psutil).")
    for data in {r[0] for r in todo}:
        if not os.path.exists(os.path.join(ROOT, data, "house_1", "input_data.csv")):
            sys.exit(f"Λείπουν δεδομένα: {os.path.join(ROOT, data)}")

    threads = a.threads or max(1, (os.cpu_count() or 2) // a.jobs)
    env = dict(os.environ, TF_NUM_INTRAOP_THREADS=str(threads), TF_NUM_INTEROP_THREADS="1",
               OMP_NUM_THREADS=str(threads))
    os.makedirs(os.path.join(ROOT, a.out, "logs"), exist_ok=True)

    def run(r):
        data, rule, atk, frac, seed = r[:5]
        beta = r[5] if len(r) > 5 else 0
        log = os.path.join(ROOT, a.out, "logs", tag(*r) + ".log")
        cmd = [sys.executable, os.path.join(ROOT, "tools", "offline_sim.py"), "--data", data, "--strategy", rule,
               "--attack", atk, "--frac", str(frac), "--seed", str(seed), "--tmean-beta", str(beta), "--out", os.path.join(ROOT, a.out)]
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        (ok.__setitem__(0, ok[0] + 1) if rc == 0 else fails.append(tag(*r)))
        print(f"{'OK ' if rc == 0 else 'ERR'} {tag(*r)}", flush=True)

    fails = []

    def run_guarded(r):
        if len(fails) >= 3 * a.jobs and ok[0] == 0:
            return                       # όλα αποτυγχάνουν — σταμάτα, δες τα logs
        run(r)

    ok = [0]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run_guarded, todo))
    if fails and ok[0] == 0:
        print(f"Σταμάτησε: {len(fails)} αποτυχίες χωρίς κανένα επιτυχημένο run. Δες {os.path.join(a.out, 'logs')}")


if __name__ == "__main__":
    main()

    