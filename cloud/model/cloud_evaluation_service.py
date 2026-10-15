"""
cloud_evaluation_service.py
============================
Αξιολογεί το global model μετά από κάθε FL aggregation round.
FORECAST_MODE=dayahead: πρόβλεψη επόμενης μέρας (00:00) για κάθε σπίτι, μετρικές στο άθροισμα (feeder:
σύνολο + ανά fog) και ανά σπίτι, μαζί με baselines (προφίλ 7 ημερών, ίδια ώρα πριν 1 εβδομάδα) και skill.
Υποστηρίζει και LCL data (αρχεία απευθείας στο /app/data/) και
συνθετικά data (house_1/input_data.csv δομή).
"""
import os
import json
import time
import glob
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from cloud.communication.cloud_resources_paths import CloudResourcesPaths
from shared.logging_config import logger
from shared import normalization as norm
from shared import forecasting as fc

# ── Configuration ─────────────────────────────────────────────
SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH', 48))
BATCH_SIZE      = 256
WINDOWS         = [3, 6, 12, 24]

# Ίδιο train/eval παράθυρο με τα edges (βλ. model_training_service).
# Διόρθωση: παλιότερα το test set ήταν οι γραμμές 20*48..29*48 (λογική του
# 1-μηνιαίου dataset) — με 12 μήνες αυτές πέφτουν ΜΕΣΑ στο training period,
# και αξιολογούνταν μόνο τα house_1..10.
TRAINING_DATE = os.getenv('TRAINING_DATE', '2012-02-01')
TRAIN_DAYS    = int(os.getenv('TRAIN_DAYS', 274))
EVAL_DAYS     = int(os.getenv('EVAL_DAYS', 92))

FEATURE_COLS = [
    "value_diff",
    "value_rolling_mean_3",  "value_volatility_3",  "value_ewm_3",
    "value_rolling_mean_6",  "value_volatility_6",  "value_ewm_6",
    "value_rolling_mean_12", "value_volatility_12", "value_ewm_12",
    "value_rolling_mean_24", "value_volatility_24", "value_ewm_24",
    "drift_flag", "time_since_last_spike",
]

# Τα δεδομένα αξιολόγησης δεν αλλάζουν από γύρο σε γύρο: preprocessing ΜΙΑ φορά
# και cache. Επίσης μόνο ένα evaluation τη φορά (αλλιώς τα threads στοιβάζονται
# και κλέβουν CPU από τα edges που τρέχουν στο ίδιο μηχάνημα).
_HOUSES_CACHE = None
_EVAL_LOCK = threading.Lock()
EVAL_BATCH = int(os.getenv('CLOUD_EVAL_BATCH', 4096))

RESULTS_FILE = os.path.join(
    CloudResourcesPaths.STATUS_FOLDER_PATH.value,
    "fl_global_model_metrics.json"
)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"timestamp": "datetime", "consumption_kwh": "value"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "value"])
    df["value"] = df["value"].astype(float)

    for w in WINDOWS:
        df[f"value_rolling_mean_{w}"] = (
            df["value"].rolling(w).mean().interpolate().ffill().bfill()
        )

    df["value_diff"] = df["value"].diff().fillna(0)
    for w in WINDOWS:
        df[f"value_ewm_{w}"]        = df["value"].ewm(span=w, adjust=False).mean()
        df[f"value_volatility_{w}"] = df["value"].rolling(w).std().fillna(0)

    baseline_vol     = df["value"].rolling(6).std().fillna(0)
    df["drift_flag"] = (df["value"].diff().abs() > baseline_vol * 2).astype(int)

    counter, time_since = 0, []
    for flag in df["drift_flag"]:
        counter = 0 if flag == 1 else counter + 1
        time_since.append(counter)
    df["time_since_last_spike"] = time_since

    return df


def _find_csv_files(base: str) -> list:
    """Όλα τα house_N/input_data.csv (ταξινομημένα κατά N), αλλιώς *.csv."""
    paths = glob.glob(os.path.join(base, "house_*", "input_data.csv"))

    def _num(p):
        try:
            return int(os.path.basename(os.path.dirname(p)).split("_")[1])
        except Exception:
            return 10 ** 9
    paths = sorted(paths, key=_num)
    if not paths:
        paths = sorted(glob.glob(os.path.join(base, "*.csv")))
    return paths


def _windows():
    t0 = pd.to_datetime(TRAINING_DATE)
    tr = (t0, t0 + pd.Timedelta(days=TRAIN_DAYS))
    ev = (t0 + pd.Timedelta(days=TRAIN_DAYS + 1),
          t0 + pd.Timedelta(days=TRAIN_DAYS + 1 + EVAL_DAYS))
    return tr, ev


def _load_houses() -> list:
    """Επιστρέφει [(name, X_eval_norm, y_eval_norm, stats)] για κάθε σπίτι."""
    (tr0, tr1), (ev0, ev1) = _windows()
    out = []
    for base in ["/app/data", "./data"]:
        if not os.path.isdir(base):
            continue
        csv_files = _find_csv_files(base)
        if not csv_files:
            continue
        logger.info("Cloud eval: found %d houses in %s | eval window %s -> %s",
                    len(csv_files), base, ev0.date(), ev1.date())
        for path in csv_files:
            name = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
            try:
                raw = pd.read_csv(path)
                ts = pd.to_datetime(raw["timestamp"], errors="coerce")
                tr_df = _preprocess(raw[(ts >= tr0) & (ts < tr1)].copy())
                ev_df = _preprocess(raw[(ts >= ev0) & (ts < ev1)].copy())
                if len(ev_df) <= SEQUENCE_LENGTH:
                    logger.warning("Cloud eval: %s has only %d eval rows, skipping.", name, len(ev_df))
                    continue
                stats = norm.fit_stats(tr_df) if norm.enabled() else None
                ev_n = norm.transform_df(ev_df, stats)
                ev_n = ev_n.dropna(subset=FEATURE_COLS + ["value"])
                X = ev_n[FEATURE_COLS].astype("float32").values
                y = ev_n["value"].astype("float32").values
                Xw, yw = _windows_of(X, y)
                out.append((name, Xw, yw, stats))
            except Exception as e:
                logger.error("Cloud eval: failed to load %s: %s", path, e)
        if out:
            break
    if not out:
        logger.warning("Cloud eval: no test data found — skipping evaluation.")
    return out


def _windows_of(X, y):
    """Ίδια αντιστοίχιση με timeseries_dataset_from_array (target y[i] για X[i:i+L]),
    αλλά vectorized: ένας πίνακας παραθύρων αντί για tf.data loop."""
    L = SEQUENCE_LENGTH
    n = len(X) - L + 1
    Xw = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)[:n]   # (n, F, L)
    Xw = np.ascontiguousarray(np.transpose(Xw, (0, 2, 1)), dtype=np.float32)  # (n, L, F)
    return Xw, y[:n].astype(np.float32)


def _predict(model, Xw, yw):
    yp = model.predict(Xw, batch_size=EVAL_BATCH, verbose=0).flatten()
    return yw, yp


def _evaluate(model, houses: list) -> dict:
    yt_kwh, yp_kwh, yt_n, yp_n, per_house = [], [], [], [], {}
    for name, Xw, yw, stats in houses:
        t, p = _predict(model, Xw, yw)
        tk, pk = norm.inverse_target(t, stats), norm.inverse_target(p, stats)
        yt_kwh.append(tk); yp_kwh.append(pk); yt_n.append(t); yp_n.append(p)
        per_house[name] = {
            "mse": float(mean_squared_error(tk, pk)),
            "r2": float(r2_score(tk, pk)),
        }
    if not yt_kwh:
        return {}
    tk, pk = np.concatenate(yt_kwh), np.concatenate(yp_kwh)
    tn, pn = np.concatenate(yt_n), np.concatenate(yp_n)
    r2s = [v["r2"] for v in per_house.values()]
    return {
        # pooled, σε kWh (ίδια μονάδα με τα παλιά runs)
        "mse": float(mean_squared_error(tk, pk)),
        "mae": float(mean_absolute_error(tk, pk)),
        "r2":  float(r2_score(tk, pk)),
        # pooled στον κανονικοποιημένο χώρο
        "mse_norm": float(mean_squared_error(tn, pn)),
        "r2_norm":  float(r2_score(tn, pn)),
        # ανά σπίτι
        "r2_house_median": float(np.median(r2s)),
        "r2_house_min":    float(np.min(r2s)),
        "n_houses": len(per_house),
        "per_house": per_house,
    }


# ── Day-ahead (FORECAST_MODE=dayahead) ─────────────────────────────────────

def _fog_groups(nums: list) -> dict:
    """FOG_GROUPS="1-25,26-50" → {'fog1': [...], 'fog2': [...]} (αριθμοί σπιτιών)."""
    spec = os.getenv("FOG_GROUPS", "1-25,26-50")
    groups = {}
    for k, part in enumerate(x for x in spec.split(",") if x.strip()):
        a, b = (int(v) for v in part.split("-"))
        groups[f"fog{k + 1}"] = [n for n in nums if a <= n <= b]
    return {k: v for k, v in groups.items() if v}


def _load_houses_dayahead() -> list:
    """[(num, name, arrays)] με στόχους τις μέρες [T1, T1+EVAL_DAYS), ίδια παράθυρα με τα edges."""
    t0, t1, e1 = fc.windows(TRAINING_DATE, TRAIN_DAYS, EVAL_DAYS)
    out = []
    for base in ["/app/data", "./data"]:
        if not os.path.isdir(base):
            continue
        csv_files = _find_csv_files(base)
        if not csv_files:
            continue
        logger.info("Cloud eval (dayahead): %d houses in %s | train %s -> %s | eval days %s -> %s",
                    len(csv_files), base, t0.date(), t1.date(), t1.date(), e1.date())
        for path in csv_files:
            name = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
            try:
                num = int(name.split("_")[1])
            except Exception:
                num = len(out) + 1
            try:
                ser = fc.load_series(path, t0, e1)
                tr = ser[ser["ts"] < t1]
                ev = ser[ser["ts"] >= t1 - pd.Timedelta(days=fc.HISTORY_DAYS)].reset_index(drop=True)
                stats = fc.fit_stats(tr) if norm.enabled() else None
                A = fc.build_arrays(ev, stats, target_start=t1, target_end=e1, issue_at_midnight=True)
                if A["n"] == 0:
                    logger.warning("Cloud eval: %s has no eval days, skipping.", name)
                    continue
                A.pop("ts", None)
                out.append((num, name, A))
            except Exception as e:
                logger.error("Cloud eval: failed to load %s: %s", path, e)
        if out:
            break
    if out:
        n_days = {A["n"] for _, _, A in out}
        if len(n_days) > 1:
            m = min(n_days)
            logger.warning("Cloud eval: unequal eval days %s — truncating to %d", n_days, m)
            for _, _, A in out:
                for k in ("Xh", "Xf", "y", "base_n", "y_kwh", "prof7_kwh", "week_kwh"):
                    A[k] = A[k][:m]
                A["n"] = m
    else:
        logger.warning("Cloud eval: no test data found — skipping evaluation.")
    return out


def _group_metrics(Y, P, nums, groups):
    tot = fc.metrics(Y.sum(0), P.sum(0))
    fogs = {g: fc.metrics(Y[[nums.index(n) for n in ns]].sum(0), P[[nums.index(n) for n in ns]].sum(0))
            for g, ns in groups.items()}
    per = [fc.metrics(Y[i], P[i]) for i in range(len(nums))]
    return tot, fogs, per


def _evaluate_dayahead(model, houses: list) -> dict:
    nums = [n for n, _, _ in houses]
    groups = _fog_groups(nums)
    Y = np.stack([A["y_kwh"] for _, _, A in houses])                 # (houses, days, 48)
    P = np.stack([fc.predict_kwh(model, A, EVAL_BATCH) for _, _, A in houses])
    tot, fogs, per = _group_metrics(Y, P, nums, groups)
    res = {
        "forecast_mode": "dayahead",
        "feeder": tot,
        "fogs": fogs,
        "house_nmae_median": float(np.median([m["nmae"] for m in per])),
        "house_r2_median": float(np.median([m["r2"] for m in per])),
        "house_r2_min": float(np.min([m["r2"] for m in per])),
        "n_houses": len(houses),
        "per_house": {name: {"nmae": m["nmae"], "r2": m["r2"]} for (_, name, _), m in zip(houses, per)},
        # συμβατότητα με τα παλιά runs: r2/mse/mae = feeder
        "r2": tot["r2"], "mse": tot["mse"], "mae": tot["mae"],
    }
    for bname, key in (("baseline_prof7", "prof7_kwh"), ("baseline_week", "week_kwh")):
        B = np.stack([A[key] for _, _, A in houses])
        bt, bf, bp = _group_metrics(Y, B, nums, groups)
        res[bname] = {"feeder": bt, "fogs": bf,
                      "house_nmae_median": float(np.median([m["nmae"] for m in bp])),
                      "house_r2_median": float(np.median([m["r2"] for m in bp]))}
    b = res["baseline_prof7"]["feeder"]
    res["skill_nmae"] = float(1 - tot["nmae"] / b["nmae"])
    res["skill_day"] = float(1 - tot["day_err"] / b["day_err"])
    res["skill_house_nmae"] = float(1 - res["house_nmae_median"] / res["baseline_prof7"]["house_nmae_median"])
    return res


def evaluate_global_model(round_num: int):
    model_path = CloudResourcesPaths.CLOUD_MODEL_FILE_PATH.value
    if not os.path.exists(model_path):
        logger.warning("Cloud eval: model not found at %s.", model_path)
        return

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        logger.error("Cloud eval: failed to load model: %s", e)
        return

    global _HOUSES_CACHE
    if not _EVAL_LOCK.acquire(blocking=False):
        logger.warning("Cloud eval [Round %d]: previous evaluation still running — skipped.", round_num)
        return
    try:
        t0 = time.perf_counter()
        if _HOUSES_CACHE is None:
            _HOUSES_CACHE = _load_houses_dayahead() if fc.dayahead() else _load_houses()
        houses = _HOUSES_CACHE
        if not houses:
            return
        metrics = _evaluate_dayahead(model, houses) if fc.dayahead() else _evaluate(model, houses)
        if metrics:
            metrics["eval_time_s"] = round(time.perf_counter() - t0, 2)
    finally:
        _EVAL_LOCK.release()
    if not metrics:
        return

    metrics["round"] = round_num
    metrics["timestamp"] = int(time.time())

    if fc.dayahead():
        f, b = metrics["feeder"], metrics["baseline_prof7"]["feeder"]
        logger.info(
            "Cloud eval [Round %d] day-ahead: feeder nMAE=%.2f%% (baseline %.2f%%) | R²=%.3f (baseline %.3f) | "
            "day err=%.2f%% (baseline %.2f%%) | fog nMAE=%s | house nMAE med=%.1f%% R² med=%.3f | "
            "skill nMAE=%.3f day=%.3f | houses=%d | %.1fs",
            round_num, f["nmae"] * 100, b["nmae"] * 100, f["r2"], b["r2"], f["day_err"] * 100, b["day_err"] * 100,
            "/".join(f"{g['nmae'] * 100:.2f}%" for g in metrics["fogs"].values()),
            metrics["house_nmae_median"] * 100, metrics["house_r2_median"],
            metrics["skill_nmae"], metrics["skill_day"], metrics["n_houses"], metrics.get("eval_time_s", -1))
        _append_metrics(metrics)
        return

    logger.info(
        "Cloud eval [Round %d]: MSE=%.4f | MAE=%.4f | R²=%.4f (kWh, pooled) | "
        "R²_norm=%.4f | median house R²=%.4f | min house R²=%.4f | houses=%d | %.1fs",
        round_num, metrics["mse"], metrics["mae"], metrics["r2"],
        metrics["r2_norm"], metrics["r2_house_median"], metrics["r2_house_min"],
        metrics["n_houses"], metrics.get("eval_time_s", -1),
    )

    _append_metrics(metrics)


def _append_metrics(metrics: dict):
    try:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

        existing = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                existing = json.load(f)

        existing.append(metrics)

        with open(RESULTS_FILE, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info("Cloud eval: metrics saved to %s", RESULTS_FILE)

    except Exception as e:
        logger.error("Cloud eval: failed to save metrics: %s", e)

        