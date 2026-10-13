#!/usr/bin/env python3
"""
Build lcl_data50/house_1..50/input_data.csv with 50 DISTINCT LCL households
(12 months: 2012-02-01 .. 2013-01-31, 17,568 half-hours, format: timestamp,consumption_kwh).

- house_1..10  : input_data.csv copied verbatim from the existing lcl_data3/house_1..10.
- house_11..50 : 40 new Std-tariff households with COMPLETE coverage (0 missing slots),
                 random sample with seed 42 (reproducible).

Format safety: before writing anything, house_1 (MAC000049) is re-extracted from the full LCL
and compared with the existing lcl_data3/house_1/input_data.csv. If not identical -> stop.

Usage (PowerShell, laptop A):
  python make_lcl_data50.py --lcl "<...>\\CC_LCL-FullData.csv" --src <...>\\lcl_data3 --out <...>\\lcl_data50
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict

import pandas as pd

START, END_EXCL = "2012-02-01", "2013-02-01"
SLOTS = 366 * 48
N_TOTAL, N_EXISTING, SEED, CHUNK = 50, 10, 42, 5_000_000
EXISTING = ["MAC000049", "MAC000085", "MAC000105", "MAC000216", "MAC000245",
            "MAC004458", "MAC004504", "MAC004562", "MAC004888", "MAC005047"]
ID, TARIFF, DT = "LCLid", "stdorToU", "DateTime"
FNAME = "input_data.csv"


def kwh_column(lcl):
    cols = pd.read_csv(lcl, nrows=0).columns.tolist()
    k = [c for c in cols if c.strip().lower().startswith("kwh")]
    if not k or ID not in cols or DT not in cols or TARIFF not in cols:
        sys.exit(f"Unexpected LCL header: {cols}")
    return k[0]


def pass1(lcl, kwh, cache):
    if os.path.exists(cache):
        print(f"[pass1] using cache {cache}")
        return json.load(open(cache))
    n, tar = defaultdict(int), {}
    for k, ch in enumerate(pd.read_csv(lcl, usecols=[ID, TARIFF, DT, kwh], dtype=str, chunksize=CHUNK)):
        ch = ch[(ch[DT] >= START) & (ch[DT] < END_EXCL)]
        ch = ch[pd.to_numeric(ch[kwh], errors="coerce").notna()]
        for lid, c in ch.groupby(ID).size().items():
            n[lid] += int(c)
        for lid, t in ch.groupby(ID)[TARIFF].first().items():
            tar.setdefault(lid, t)
        print(f"[pass1] chunk {k + 1}", flush=True)
    res = {lid: [n[lid], tar.get(lid)] for lid in n}
    json.dump(res, open(cache, "w"))
    return res


def pass2(lcl, kwh, macs):
    want, parts = set(macs), defaultdict(list)
    for k, ch in enumerate(pd.read_csv(lcl, usecols=[ID, DT, kwh], dtype=str,
                                       chunksize=CHUNK, keep_default_na=False)):
        ch = ch[ch[ID].isin(want) & (ch[DT] >= START) & (ch[DT] < END_EXCL)]
        for lid, g in ch.groupby(ID):
            parts[lid].append(g)
        print(f"[pass2] chunk {k + 1}", flush=True)
    return {lid: pd.concat(p, ignore_index=True) for lid, p in parts.items()}


def shape(raw, kwh, value_mode):
    d = raw.copy()
    d["timestamp"] = d[DT].str.strip().str.replace(r"\.\d+$", "", regex=True)
    v = d[kwh].str.strip()
    num = pd.to_numeric(v, errors="coerce")
    d = d[num.notna()].copy()
    if value_mode == "text":
        d["consumption_kwh"] = v[num.notna()]
    else:  # float repr, as pandas would write it
        d["consumption_kwh"] = num[num.notna()].astype(float).map(repr)
    d = d.drop_duplicates("timestamp").sort_values("timestamp")
    return d[["timestamp", "consumption_kwh"]].reset_index(drop=True)


MAX_MONTH_ZERO_FRAC = 0.5    # π.χ. κενό σπίτι για έναν μήνα -> απόρριψη
MIN_EVAL_TRAIN_RATIO = 0.25  # μέσος eval / μέσος train: κάτω από αυτό -> απόρριψη
EVAL_START = "2012-11-02"


def usable(df):
    """Απορρίπτει σπίτια που είναι (σχεδόν) άδεια για κάποιο διάστημα,
    π.χ. MAC που έμεινε κενό τον Δεκέμβριο (100% μηδενικά στο eval)."""
    v = df["consumption_kwh"].astype(float)
    t = pd.to_datetime(df["timestamp"])
    zero_by_month = (v == 0).groupby(t.dt.to_period("M")).mean()
    if zero_by_month.max() > MAX_MONTH_ZERO_FRAC:
        return False, f"month with {zero_by_month.max():.0%} zeros ({zero_by_month.idxmax()})"
    tr, ev = v[t < EVAL_START].mean(), v[t >= EVAL_START].mean()
    if tr <= 0 or ev / tr < MIN_EVAL_TRAIN_RATIO:
        return False, f"eval/train mean ratio {ev / max(tr, 1e-9):.2f}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcl", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    kwh = kwh_column(a.lcl)
    ref = pd.read_csv(os.path.join(a.src, "house_1", FNAME), dtype=str, keep_default_na=False)
    if list(ref.columns) != ["timestamp", "consumption_kwh"]:
        sys.exit(f"Unexpected reference columns: {list(ref.columns)}")
    for n in range(1, N_EXISTING + 1):
        if not os.path.isfile(os.path.join(a.src, f"house_{n}", FNAME)):
            sys.exit(f"missing {a.src}/house_{n}/{FNAME}")

    cov = pass1(a.lcl, kwh, os.path.join(os.path.dirname(os.path.abspath(a.out)), "lcl_coverage_cache.json"))
    cand = sorted(l for l, (n, t) in cov.items() if t == "Std" and n >= SLOTS and l not in EXISTING)
    print(f"[select] Std households with full {SLOTS}-slot coverage: {len(cand)}")
    need = N_TOTAL - N_EXISTING
    if len(cand) < need:
        sys.exit("Not enough candidates.")
    # pass1 counts include LCL duplicate timestamps, so many "full" houses miss 1-9 slots
    # after dedupe -> extract ALL candidates, in a seeded random order, and take the first 40 complete.
    pool = random.Random(SEED).sample(cand, len(cand))

    data = pass2(a.lcl, kwh, pool + [EXISTING[0]])

    mode = None
    for m in ("text", "float"):
        if shape(data[EXISTING[0]], kwh, m).equals(ref):
            mode = m
            break
    if mode is None:
        s = shape(data[EXISTING[0]], kwh, "text")
        print("FORMAT CHECK FAILED for house_1 (MAC000049).")
        print("existing:\n", ref.head(5).to_string(), "\nrows", len(ref))
        print("re-extracted:\n", s.head(5).to_string(), "\nrows", len(s))
        diff = (s.ne(ref)).any(axis=1) if len(s) == len(ref) else None
        if diff is not None and diff.any():
            i = diff.idxmax()
            print("first diff row", i, "\n", ref.iloc[i].to_dict(), "\n", s.iloc[i].to_dict())
        sys.exit(1)
    print(f"[format] OK: re-extracted house_1 identical to existing input_data.csv (mode={mode})")

    os.makedirs(a.out, exist_ok=True)
    manifest = []
    for n in range(1, N_EXISTING + 1):
        dst = os.path.join(a.out, f"house_{n}")
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(a.src, f"house_{n}", FNAME), os.path.join(dst, FNAME))
        manifest.append((n, EXISTING[n - 1], "thesis house"))
        ok, why = usable(pd.read_csv(os.path.join(dst, FNAME)))
        if not ok:
            print(f"[warn] thesis house_{n} ({EXISTING[n - 1]}): {why}")

    n = N_EXISTING
    for mac in pool:
        if n == N_TOTAL:
            break
        if mac not in data:
            continue
        df = shape(data[mac], kwh, mode)
        if len(df) != SLOTS:
            print(f"[skip] {mac}: {SLOTS - len(df)} missing after dedupe")
            continue
        ok, why = usable(df)
        if not ok:
            print(f"[skip] {mac}: {why}")
            continue
        n += 1
        dst = os.path.join(a.out, f"house_{n}")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        df.to_csv(os.path.join(dst, FNAME), index=False, lineterminator="\n")
        manifest.append((n, mac, "new"))
        print(f"[write] house_{n} <- {mac}")

    if n < N_TOTAL:
        sys.exit(f"Only {n} houses written.")
    pd.DataFrame(manifest, columns=["house", "LCLid", "note"]).to_csv(
        os.path.join(a.out, "houses_manifest.csv"), index=False)
    print(f"Done: {N_TOTAL} distinct houses in {a.out}")


if __name__ == "__main__":
    main()

    