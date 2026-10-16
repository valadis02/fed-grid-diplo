#!/usr/bin/env python3
"""Ένα πέρασμα στο LCL: πόσα μισάωρα έχει κάθε σπίτι ανά μήνα (+ tariff).
Έξοδος: lcl_monthly_coverage.csv (LCLid, tariff, 2011-11, 2011-12, ..., 2014-02) — λίγα KB, στείλ' το μου.

Χρήση:
  python lcl_coverage_scan.py --lcl "<...>\\CC_LCL-FullData.csv" --out lcl_monthly_coverage.csv
"""
import argparse
from collections import defaultdict

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--lcl", required=True)
ap.add_argument("--out", default="lcl_monthly_coverage.csv")
a = ap.parse_args()

cols = pd.read_csv(a.lcl, nrows=0).columns.tolist()
kwh = [c for c in cols if c.strip().lower().startswith("kwh")][0]
cnt, tar = defaultdict(int), {}
for k, ch in enumerate(pd.read_csv(a.lcl, usecols=["LCLid", "stdorToU", "DateTime", kwh], dtype=str, chunksize=5_000_000)):
    ch = ch[pd.to_numeric(ch[kwh], errors="coerce").notna()]
    ch = ch.assign(m=ch["DateTime"].str[:7])
    for (lid, m), c in ch.groupby(["LCLid", "m"]).size().items():
        cnt[(lid, m)] += int(c)
    for lid, t in ch.groupby("LCLid")["stdorToU"].first().items():
        tar.setdefault(lid, t)
    print(f"[scan] chunk {k + 1}", flush=True)

df = pd.Series(cnt).unstack(fill_value=0).sort_index(axis=1)
df.insert(0, "tariff", pd.Series(tar))
df.index.name = "LCLid"
df.to_csv(a.out)
print(f"Done: {len(df)} houses, months {df.columns[1]} .. {df.columns[-1]} -> {a.out}")
