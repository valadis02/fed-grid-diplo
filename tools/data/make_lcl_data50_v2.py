#!/usr/bin/env python3
"""
Φτιάχνει lcl_data50_v2/house_1..50/input_data.csv με 50 ΠΡΑΓΜΑΤΙΚΑ σπίτια του LCL,
ΚΑΝΕΝΑ από τα 10 της διπλωματικής.

Περίοδος (default): 2012-10-15 .. 2014-02-27 (~16,5 μήνες). Οι περισσότεροι μετρητές του LCL μπήκαν
μέσα στο 2012· σε αυτή την περίοδο ~2.600 Std σπίτια έχουν (σχεδόν) πλήρη δεδομένα.
  - ιστορικό : 15–31 Οκτ 2012 (για lags / προφίλ 7 ημερών)
  - train    : Νοε 2012 – Οκτ 2013 (12 μήνες, με ολόκληρο χειμώνα)
  - eval     : Νοε 2013 – Ιαν 2014 (3 μήνες, χειμώνας· ο Φεβ 2014 περισσεύει / προαιρετικός)

Κριτήρια επιλογής:
  - Std tariff, όχι σπίτι της διπλωματικής
  - το πολύ MAX_MISSING κενά μισάωρα σε όλη την περίοδο (γεμίζουν με γραμμική παρεμβολή)
  - όχι «άδειο» σπίτι: κανένας μήνας με >50% μηδενικά ή με μέση κατανάλωση <25% του διάμεσου μήνα
  - seed 42 παντού (αναπαραγώγιμο)

Έλεγχος format: το MAC000049 (house_1 της διπλωματικής) ξαναβγαίνει από το LCL και συγκρίνεται
γραμμή-γραμμή με το υπάρχον input_data.csv στο διάστημα που αυτό καλύπτει. Αν διαφέρει -> stop.
(Μόνο έλεγχος· το σπίτι αυτό ΔΕΝ μπαίνει στα 50.)

Έξοδοι:
  <out>/house_N/input_data.csv     (timestamp,consumption_kwh — ίδιο format με πριν)
  <out>/houses_manifest.csv        (house, LCLid, Acorn_grouped, filled_slots, mean_kwh)
  <out>/holidays.csv               (αργίες Αγγλίας/Ουαλίας στην περίοδο)

Επιλογή σπιτιών (--select). Για similar/stratified τρέχει πρώτα k-means (default k=6) σε ΟΛΑ
τα υποψήφια σπίτια του LCL με βάση τη ΣΥΜΠΕΡΙΦΟΡΑ (κανονικοποιημένο ημερήσιο προφίλ καθημερινής
και Σ/Κ + λόγος χειμώνα/καλοκαιριού) και γράφει cluster_summary.csv, cluster_assignments.csv.
  similar    : 50 σπίτια με παρόμοια συμπεριφορά: ένα cluster (default το μεγαλύτερο, ή --cluster N),
               χωρίς τα ακραία σε επίπεδο κατανάλωσης (εκτός P10–P90 του cluster),
               τα πιο κοντινά στο κέντρο του cluster
  stratified : 50 σπίτια από ΟΛΑ τα clusters, ανάλογα με το μέγεθός τους (αντιπροσωπευτικό δείγμα)
  random     : απλή τυχαία επιλογή (χωρίς clustering)

Χρήση (PowerShell):
  python make_lcl_data50_v2.py --lcl "<...>\\CC_LCL-FullData.csv" --src "<...>\\lcl_data3" --out "<...>\\lcl_data50_v2"
  python make_lcl_data50_v2.py ... --out "<...>\\lcl50_similar" --select similar
  python make_lcl_data50_v2.py ... --out "<...>\\lcl50_diverse" --select stratified
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

N_TOTAL, SEED, CHUNK, BATCH = 50, 42, 5_000_000, 300
MAX_MISSING = 48                     # ≤ 1 μέρα κενά συνολικά
THESIS = ["MAC000049", "MAC000085", "MAC000105", "MAC000216", "MAC000245",
          "MAC004458", "MAC004504", "MAC004562", "MAC004888", "MAC005047"]
ID, TARIFF, DT = "LCLid", "stdorToU", "DateTime"
FNAME = "input_data.csv"
BANK_HOLIDAYS = ["2011-12-26", "2011-12-27", "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07",
                 "2012-06-04", "2012-06-05", "2012-08-27", "2012-12-25", "2012-12-26", "2013-01-01",
                 "2013-03-29", "2013-04-01", "2013-05-06", "2013-05-27", "2013-08-26",
                 "2013-12-25", "2013-12-26", "2014-01-01"]


def header(lcl):
    cols = pd.read_csv(lcl, nrows=0).columns.tolist()
    k = [c for c in cols if c.strip().lower().startswith("kwh")]
    if not k or ID not in cols or DT not in cols or TARIFF not in cols:
        sys.exit(f"Unexpected LCL header: {cols}")
    return k[0], ("Acorn_grouped" if "Acorn_grouped" in cols else None)


def pass1(lcl, kwh, acorn, start, end, cache):
    """Μετράει μισάωρα ανά σπίτι στην περίοδο (με cache, γιατί το LCL είναι ~8 GB)."""
    if os.path.exists(cache):
        print(f"[pass1] using cache {cache}")
        return json.load(open(cache))
    n, tar, ac = defaultdict(int), {}, {}
    cols = [ID, TARIFF, DT, kwh] + ([acorn] if acorn else [])
    for k, ch in enumerate(pd.read_csv(lcl, usecols=cols, dtype=str, chunksize=CHUNK)):
        ch = ch[(ch[DT] >= start) & (ch[DT] < end)]
        ch = ch[pd.to_numeric(ch[kwh], errors="coerce").notna()]
        for lid, c in ch.groupby(ID).size().items():
            n[lid] += int(c)
        for lid, t in ch.groupby(ID)[TARIFF].first().items():
            tar.setdefault(lid, t)
        if acorn:
            for lid, a in ch.groupby(ID)[acorn].first().items():
                ac.setdefault(lid, a)
        print(f"[pass1] chunk {k + 1}", flush=True)
    res = {lid: [n[lid], tar.get(lid), ac.get(lid, "")] for lid in n}
    json.dump(res, open(cache, "w"))
    return res


def pass2(lcl, kwh, macs, start, end):
    want, parts = set(macs), defaultdict(list)
    for k, ch in enumerate(pd.read_csv(lcl, usecols=[ID, DT, kwh], dtype=str,
                                       chunksize=CHUNK, keep_default_na=False)):
        ch = ch[ch[ID].isin(want) & (ch[DT] >= start) & (ch[DT] < end)]
        for lid, g in ch.groupby(ID):
            parts[lid].append(g)
        print(f"[pass2] chunk {k + 1} ({len(want)} houses)", flush=True)
    return {lid: pd.concat(p, ignore_index=True) for lid, p in parts.items()}


def shape(raw, kwh, value_mode):
    d = raw.copy()
    d["timestamp"] = d[DT].str.strip().str.replace(r"\.\d+$", "", regex=True)
    v = d[kwh].str.strip()
    num = pd.to_numeric(v, errors="coerce")
    d = d[num.notna()].copy()
    if value_mode == "text":
        d["consumption_kwh"] = v[num.notna()]
    else:
        d["consumption_kwh"] = num[num.notna()].astype(float).map(repr)
    d = d.drop_duplicates("timestamp").sort_values("timestamp")
    return d[["timestamp", "consumption_kwh"]].reset_index(drop=True)


def complete(df, start, end):
    """Πλήρες πλέγμα μισάωρων· τα λίγα κενά γεμίζουν με γραμμική παρεμβολή. Επιστρέφει (df, #κενά)."""
    idx = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(minutes=30), freq="30min")
    s = df.set_index(pd.to_datetime(df["timestamp"]))["consumption_kwh"]
    txt = s.reindex(idx)
    missing = int(txt.isna().sum())
    if missing:
        num = pd.to_numeric(txt, errors="coerce").interpolate(limit_direction="both")
        txt = txt.where(txt.notna(), num.map(lambda x: f"{x:.3f}"))
    out = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"), "consumption_kwh": txt.values})
    return out, missing


def usable(df):
    v = df["consumption_kwh"].astype(float)
    m = pd.to_datetime(df["timestamp"]).dt.to_period("M")
    zero = (v == 0).groupby(m).mean()
    if zero.max() > 0.5:
        return False, f"month with {zero.max():.0%} zeros ({zero.idxmax()})"
    mm = v.groupby(m).mean()
    if mm.min() < 0.25 * mm.median():
        return False, f"near-empty month {mm.idxmin()} ({mm.min():.3f} vs median {mm.median():.3f})"
    return True, ""


def pass_features(lcl, kwh, macs, start, end, cache):
    """Ένα πέρασμα στο LCL: για κάθε υποψήφιο σπίτι, μέση κατανάλωση ανά (μισάωρο, καθημερινή/Σ/Κ) και ανά μήνα."""
    if os.path.exists(cache):
        print(f"[features] using cache {cache}")
        z = json.load(open(cache))
        return z["macs"], np.array(z["X"]), z["info"]
    want = set(macs)
    s_slot, n_slot, s_mon, n_mon = defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)
    wd_cache = {}
    for k, ch in enumerate(pd.read_csv(lcl, usecols=[ID, DT, kwh], dtype=str, chunksize=CHUNK)):
        ch = ch[ch[ID].isin(want) & (ch[DT] >= start) & (ch[DT] < end)]
        v = pd.to_numeric(ch[kwh], errors="coerce")
        ch = ch[v.notna()].assign(v=v[v.notna()])
        if ch.empty:
            continue
        dt = ch[DT].str.strip()
        day = dt.str[:10]
        for d in day.unique():
            if d not in wd_cache:
                wd_cache[d] = int(pd.Timestamp(d).dayofweek >= 5)
        ch = ch.assign(slot=dt.str[11:13].astype(int) * 2 + (dt.str[14:16].astype(int) >= 30),
                       we=day.map(wd_cache), mon=dt.str[5:7].astype(int))
        for key, val in ch.groupby([ID, "we", "slot"])["v"].agg(["sum", "count"]).iterrows():
            s_slot[key] += val["sum"]; n_slot[key] += int(val["count"])
        for key, val in ch.groupby([ID, "mon"])["v"].agg(["sum", "count"]).iterrows():
            s_mon[key] += val["sum"]; n_mon[key] += int(val["count"])
        print(f"[features] chunk {k + 1}", flush=True)
    out_m, X, info = [], [], {}
    for m in macs:
        prof = np.array([[s_slot.get((m, we, sl), 0) / max(n_slot.get((m, we, sl), 0), 1) for sl in range(48)]
                         for we in (0, 1)])
        mon = {mo: s_mon[(m, mo)] / n_mon[(m, mo)] for mo in range(1, 13) if n_mon.get((m, mo))}
        mean = prof.mean()
        if mean <= 0 or not all(x in mon for x in (1, 7)):
            continue
        winter = np.mean([mon[x] for x in (12, 1, 2) if x in mon]); summer = np.mean([mon[x] for x in (6, 7, 8) if x in mon])
        ratio = winter / max(summer, 1e-6)
        out_m.append(m)
        X.append(list(prof[0] / mean) + list(prof[1] / mean) + [np.log(ratio)])
        info[m] = dict(mean=float(mean), peak=float(np.argmax(prof[0]) / 2), we_ratio=float(prof[1].mean() / max(prof[0].mean(), 1e-6)),
                       ws_ratio=float(ratio))
    json.dump(dict(macs=out_m, X=X, info=info), open(cache, "w"))
    return out_m, np.array(X), info


def kmeans(X, k, seed=SEED, restarts=10, iters=100):
    """Απλό k-means (k-means++), χωρίς sklearn."""
    rng = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        C = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), 1)
            C.append(X[rng.choice(len(X), p=d / d.sum())])
        C = np.array(C)
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcl", required=True)
    ap.add_argument("--src", required=True, help="φάκελος με house_1/input_data.csv της διπλωματικής (έλεγχος format)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2012-10-15")
    ap.add_argument("--end", default="2014-02-28", help="exclusive")
    ap.add_argument("--select", choices=["random", "stratified", "similar"], default="random")
    ap.add_argument("--k", type=int, default=6, help="αριθμός clusters (μόνο για --select cluster)")
    ap.add_argument("--cluster", type=int, default=-1, help="για --select similar: ποιο cluster (default: το μεγαλύτερο)")
    a = ap.parse_args()
    start, end = a.start, a.end
    slots = len(pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(minutes=30), freq="30min"))
    print(f"[period] {start} .. {end} (excl), {slots} half-hours")

    kwh, acorn = header(a.lcl)
    ref = pd.read_csv(os.path.join(a.src, "house_1", FNAME), dtype=str, keep_default_na=False)
    if list(ref.columns) != ["timestamp", "consumption_kwh"]:
        sys.exit(f"Unexpected reference columns: {list(ref.columns)}")

    cache = os.path.join(os.path.dirname(os.path.abspath(a.out)),
                         f"lcl_coverage_{start}_{end}.json".replace(":", ""))
    cov = pass1(a.lcl, kwh, acorn, start, end, cache)
    cand = sorted(l for l, (n, t, _) in cov.items()
                  if t == "Std" and n >= slots - MAX_MISSING and l not in THESIS)
    print(f"[select] Std, non-thesis, ≤{MAX_MISSING} missing: {len(cand)} candidates")
    if len(cand) < N_TOTAL:
        sys.exit("Not enough candidates — try a shorter/later period")
    group_of, quota, clus_of = {m: 0 for m in cand}, {0: N_TOTAL}, {}
    if a.select in ("similar", "stratified"):
        fcache = cache.replace("lcl_coverage_", "lcl_features_")
        macs, X, info = pass_features(a.lcl, kwh, cand, start, end, fcache)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Xs[:, -1] *= np.sqrt(48)          # ο λόγος χειμώνα/καλοκαιριού να «ζυγίζει» όσο ένα ημερήσιο προφίλ
        lab = kmeans(Xs, a.k)
        cent = np.array([Xs[lab == j].mean(0) for j in range(a.k)])
        dist = np.sqrt(((Xs - cent[lab]) ** 2).sum(1))
        rows = []
        for j in range(a.k):
            mm = [m for m, l in zip(macs, lab) if l == j]
            ac = pd.Series([cov[m][2] for m in mm]).value_counts()
            rows.append(dict(cluster=j, size=len(mm),
                             mean_kwh=round(np.median([info[m]["mean"] for m in mm]), 3),
                             weekday_peak_hour=round(np.median([info[m]["peak"] for m in mm]), 1),
                             weekend_vs_weekday=round(np.median([info[m]["we_ratio"] for m in mm]), 2),
                             winter_vs_summer=round(np.median([info[m]["ws_ratio"] for m in mm]), 2),
                             top_acorn=(f"{ac.index[0]} ({ac.iloc[0] / len(mm):.0%})" if len(ac) and ac.index[0] else "")))
        summ = pd.DataFrame(rows).sort_values("size", ascending=False)
        print("[cluster] summary (median ανά cluster):\n" + summ.to_string(index=False))
        meta = pd.DataFrame({"LCLid": macs, "cluster": lab, "dist_to_centroid": dist.round(3),
                             "mean_kwh": [round(info[m]["mean"], 4) for m in macs]})
        clus_of = dict(zip(meta.LCLid, meta.cluster.astype(int)))
        if a.select == "similar":
            # ένα cluster· εκτός τα ακραία σε επίπεδο κατανάλωσης (εκτός P10–P90 του cluster)·
            # από τα υπόλοιπα, τα πιο κοντινά στο κέντρο (πιο «τυπική» συμπεριφορά) πρώτα
            pick = a.cluster if a.cluster >= 0 else int(summ.iloc[0]["cluster"])
            sub = meta[meta.cluster == pick]
            lo, hi = sub.mean_kwh.quantile(0.10), sub.mean_kwh.quantile(0.90)
            sub = sub[(sub.mean_kwh >= lo) & (sub.mean_kwh <= hi)].sort_values("dist_to_centroid")
            print(f"[similar] cluster {pick}: {int((meta.cluster == pick).sum())} houses, "
                  f"{len(sub)} after removing level extremes (mean kWh outside {lo:.3f}–{hi:.3f})")
            if len(sub) < N_TOTAL + 10:
                sys.exit("Cluster too small — try a smaller --k or another --cluster")
            pool = sub.LCLid.tolist()
            group_of = {m: 0 for m in pool}
        else:
            # αναλογική δειγματοληψία: από κάθε cluster όσα σπίτια αντιστοιχούν στο μέγεθός του
            vc = meta.cluster.value_counts()
            tiny = vc[vc < max(3, 0.02 * len(meta))].index          # clusters-«ακραία» (π.χ. 1–2 σπίτια)
            if len(tiny):
                print(f"[stratified] ignoring tiny clusters {sorted(int(t) for t in tiny)}")
                meta = meta[~meta.cluster.isin(tiny)]
            share = meta.cluster.value_counts(normalize=True) * N_TOTAL
            q = share.astype(int)
            for j in (share - q).sort_values(ascending=False).index[:N_TOTAL - q.sum()]:
                q[j] += 1
            quota = {int(j): int(v) for j, v in q.items()}
            print(f"[stratified] houses per cluster: {dict(sorted(quota.items()))}")
            group_of = dict(zip(meta.LCLid, meta.cluster.astype(int)))
            pool = random.Random(SEED).sample(list(meta.LCLid), len(meta))
        os.makedirs(a.out + "_meta", exist_ok=True)
        summ.to_csv(os.path.join(a.out + "_meta", "cluster_summary.csv"), index=False)
        meta.to_csv(os.path.join(a.out + "_meta", "cluster_assignments.csv"), index=False)
    else:
        pool = random.Random(SEED).sample(cand, len(cand))

    # format check με το house_1 της διπλωματικής (μόνο για έλεγχο)
    first = pass2(a.lcl, kwh, [THESIS[0]] + pool[:BATCH], start, end)
    mode = None
    for md in ("text", "float"):
        s = shape(first[THESIS[0]], kwh, md)
        # σύγκριση μόνο στο κοινό διάστημα της περιόδου μας και του υπάρχοντος αρχείου
        r0, r1 = max(ref.timestamp.iloc[0], s.timestamp.iloc[0]), min(ref.timestamp.iloc[-1], s.timestamp.iloc[-1])
        if r0 >= r1:
            sys.exit("No overlap with the reference file for the format check.")
        s = s[(s.timestamp >= r0) & (s.timestamp <= r1)].reset_index(drop=True)
        rr = ref[(ref.timestamp >= r0) & (ref.timestamp <= r1)].reset_index(drop=True)
        if len(rr) > 1000 and s.equals(rr):
            mode = md
            break
    if mode is None:
        print("FORMAT CHECK FAILED for MAC000049 — στείλε μου αυτό το output.")
        print("existing:\n", rr.head(3).to_string(), "\nre-extracted:\n", s.head(3).to_string())
        sys.exit(1)
    print(f"[format] OK (mode={mode})")

    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    manifest, n, pos, data = [], 0, 0, first
    taken, used, relaxed = defaultdict(int), set(), False
    while n < N_TOTAL:
        batch = [m for m in pool[pos:pos + BATCH] if m not in used]
        if pos >= len(pool):
            if relaxed or a.select != "stratified":
                sys.exit(f"Only {n} houses found.")
            print("[stratified] some clusters could not fill their quota — filling the rest from any cluster")
            relaxed, pos, data = True, 0, None
            quota = defaultdict(lambda: N_TOTAL)
            continue
        if not batch:
            pos += BATCH
            continue
        if data is None:
            data = pass2(a.lcl, kwh, batch, start, end)
        for mac in batch:
            if n == N_TOTAL:
                break
            if mac in used or mac not in data or taken[group_of[mac]] >= quota[group_of[mac]]:
                continue
            used.add(mac)
            df, miss = complete(shape(data[mac], kwh, mode), start, end)
            if miss > MAX_MISSING:
                print(f"[skip] {mac}: {miss} missing")
                continue
            ok, why = usable(df)
            if not ok:
                print(f"[skip] {mac}: {why}")
                continue
            n += 1
            taken[group_of[mac]] += 1
            dst = os.path.join(a.out, f"house_{n}")
            os.makedirs(dst)
            df.to_csv(os.path.join(dst, FNAME), index=False, lineterminator="\n")
            mean = float(df.consumption_kwh.astype(float).mean())
            manifest.append((n, mac, cov[mac][2], clus_of.get(mac, ""), miss, round(mean, 4)))
            print(f"[write] house_{n} <- {mac} (filled {miss}, mean {mean:.3f})")
        pos += BATCH
        data = None

    pd.DataFrame(manifest, columns=["house", "LCLid", "Acorn_grouped", "cluster", "filled_slots", "mean_kwh"]).to_csv(
        os.path.join(a.out, "houses_manifest.csv"), index=False)
    hol = [d for d in BANK_HOLIDAYS if start <= d < end]
    pd.DataFrame({"date": hol}).to_csv(os.path.join(a.out, "holidays.csv"), index=False)
    if a.select != "random":
        for f in ("cluster_summary.csv", "cluster_assignments.csv"):
            shutil.copy2(os.path.join(a.out + "_meta", f), os.path.join(a.out, f))
        shutil.rmtree(a.out + "_meta")
    assert not set(m[1] for m in manifest) & set(THESIS)
    print(f"Done: {N_TOTAL} houses (none from the thesis) in {a.out}")


if __name__ == "__main__":
    main()

    