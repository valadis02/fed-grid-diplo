"""
shared/forecasting.py
=====================
Day-ahead forecasting (FORECAST_MODE=dayahead) — κοινός κώδικας για edge, fog (FLTrust2) και cloud eval.

Η εκδοχή που επιλέχθηκε στη Φάση 1 (offline προσομοίωση σε 50 πραγματικά σπίτια LCL):
  * μία πρόβλεψη τη μέρα, στις 00:00 (μετά το τελευταίο γνωστό μισάωρο 23:30),
    για ΟΛΑ τα 48 μισάωρα της επόμενης μέρας (multi-output, Dense(48))
  * το μοντέλο προβλέπει την ΑΠΟΚΛΙΣΗ από το «προφίλ 7 ημερών»
    (μέσος όρος της ίδιας ώρας τις 7 προηγούμενες μέρες) — το προφίλ προστίθεται πίσω στην πρόβλεψη
  * είσοδοι: (α) παράθυρο 48 μισάωρων με τα 15 features του pipeline (ίδια με πριν, κανονικοποιημένα ανά σπίτι)
             (β) για κάθε ένα από τα 48 μισάωρα-στόχους: ώρα (sin/cos), μέρα εβδομάδας (sin/cos), αργία,
                 προφίλ 7 ημερών, τιμή ίδια ώρα πριν μία εβδομάδα  → 48 x 7 = 336 τιμές
  * loss MSE (εκτιμά τον μέσο → σωστά ημερήσια σύνολα)

Στην εκπαίδευση χρησιμοποιούνται όλες οι αφετηρίες (κάθε μισάωρο) ως επαύξηση δεδομένων·
στην αξιολόγηση μόνο οι αφετηρίες των 00:00, δηλ. ακριβώς το σενάριο λειτουργίας.

FORECAST_MODE=legacy (default) κρατά την παλιά συμπεριφορά ανέπαφη.
"""
import os

import numpy as np
import pandas as pd

from shared import normalization as norm

FORECAST_MODE = os.getenv("FORECAST_MODE", "legacy").lower()
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 48))
HORIZON = 48                       # μισάωρα της επόμενης μέρας
HISTORY_DAYS = 14                  # ιστορικό πριν τον πρώτο στόχο (lags 1 εβδομάδας + προφίλ 7 ημερών)
FIRST_ORIGIN = HISTORY_DAYS * 48 - 1
WINDOWS = (3, 6, 12, 24)

HIST_FEATURES = [
    "value_diff",
    "value_rolling_mean_3", "value_volatility_3", "value_ewm_3",
    "value_rolling_mean_6", "value_volatility_6", "value_ewm_6",
    "value_rolling_mean_12", "value_volatility_12", "value_ewm_12",
    "value_rolling_mean_24", "value_volatility_24", "value_ewm_24",
    "drift_flag", "time_since_last_spike",
]
FUT_FEATURES = ["hour_sin", "hour_cos", "wd_sin", "wd_cos", "holiday", "prof7", "week_ago"]
N_HIST = len(HIST_FEATURES)
N_FUT = HORIZON * len(FUT_FEATURES)

# Αργίες Αγγλίας/Ουαλίας (bank holidays) — καλύπτουν όλο το LCL (2011-11 .. 2014-02).
# Override: HOLIDAYS_FILE=/path/holidays.csv (στήλη 'date').
UK_BANK_HOLIDAYS = [
    "2011-12-26", "2011-12-27", "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07", "2012-06-04",
    "2012-06-05", "2012-08-27", "2012-12-25", "2012-12-26", "2013-01-01", "2013-03-29", "2013-04-01",
    "2013-05-06", "2013-05-27", "2013-08-26", "2013-12-25", "2013-12-26", "2014-01-01",
]


def dayahead() -> bool:
    return FORECAST_MODE == "dayahead"


def _holidays():
    f = os.getenv("HOLIDAYS_FILE", "")
    days = pd.read_csv(f)["date"].tolist() if f and os.path.exists(f) else UK_BANK_HOLIDAYS
    return set(pd.to_datetime(days).date)


HOLIDAYS = _holidays()


# ── Δεδομένα ────────────────────────────────────────────────────────────────

def load_series(path_or_df, start=None, end=None) -> pd.DataFrame:
    """input_data.csv (timestamp, consumption_kwh) → DataFrame [ts, value] σε πλήρες πλέγμα μισάωρων."""
    d = pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    tcol = "timestamp" if "timestamp" in d.columns else "datetime"
    vcol = "consumption_kwh" if "consumption_kwh" in d.columns else "value"
    s = pd.Series(pd.to_numeric(d[vcol], errors="coerce").values,
                  index=pd.to_datetime(d[tcol], errors="coerce")).dropna()
    s = s[~s.index.duplicated()].sort_index()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index < pd.Timestamp(end)]
    if s.empty:
        return pd.DataFrame(columns=["ts", "value"])
    grid = pd.date_range(s.index[0].floor("30min"), s.index[-1], freq="30min")
    s = s.reindex(s.index.union(grid)).interpolate(limit_direction="both").reindex(grid)
    return pd.DataFrame({"ts": grid, "value": s.values.astype("float64")})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ίδια features με edge/model/data_preprocessing.preprocess_data (στη μνήμη)."""
    df = df.copy()
    v = df["value"]
    for w in WINDOWS:
        df[f"value_rolling_mean_{w}"] = v.rolling(w).mean().interpolate().ffill().bfill()
    df["value_diff"] = v.diff().fillna(0)
    for w in WINDOWS:
        df[f"value_ewm_{w}"] = v.ewm(span=w, adjust=False).mean()
        df[f"value_volatility_{w}"] = v.rolling(w).std().fillna(0)
    base_vol = v.rolling(6).std().fillna(0)
    df["drift_flag"] = (v.diff().abs() > base_vol * 2).astype(int)
    tsls, c = [], 0
    for f in df["drift_flag"].values:
        c = 0 if f == 1 else c + 1
        tsls.append(c)
    df["time_since_last_spike"] = tsls
    return df


def fit_stats(train_series: pd.DataFrame) -> dict:
    """μ, σ από το training split του σπιτιού (ίδιο με shared.normalization)."""
    return norm.fit_stats(train_series[["value"]])


def _calendar(ts: pd.DatetimeIndex) -> np.ndarray:
    slot = ts.hour * 2 + ts.minute // 30
    wd = ts.dayofweek
    hol = np.array([t.date() in HOLIDAYS for t in ts], dtype="float32")
    return np.stack([np.sin(2 * np.pi * slot / 48), np.cos(2 * np.pi * slot / 48),
                     np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7), hol], axis=1).astype("float32")


def build_arrays(series: pd.DataFrame, stats: dict, target_start=None, target_end=None,
                 issue_at_midnight: bool = False) -> dict:
    """
    series: [ts, value] (με HISTORY_DAYS ιστορικό πριν το target_start).
    Επιστρέφει είσοδο/στόχους για όλες τις αφετηρίες o (τελευταίο γνωστό μισάωρο) με στόχους o+1..o+48:
      Xh (n, L, 15), Xf (n, 336), y (n, 48) = κανονικοποιημένη απόκλιση από το προφίλ 7 ημερών,
      base_n (n, 48), y_kwh (n, 48), prof7_kwh, week_kwh (baselines), ts (n*48)
    issue_at_midnight=True: μόνο αφετηρίες 23:30 → πρόβλεψη 00:00–23:30 της επόμενης μέρας.
    """
    mu, sd = (stats["mu"], stats["sd"]) if (stats and norm.enabled()) else (0.0, 1.0)
    df = add_features(series)
    ts = pd.DatetimeIndex(df["ts"])
    raw = df["value"].values.astype("float64")
    hist = df[HIST_FEATURES].copy()
    if stats and norm.enabled():
        hist = norm.transform_df(hist, stats)
    hist = hist.values.astype("float32")
    L = SEQUENCE_LENGTH

    o = np.arange(max(FIRST_ORIGIN, L - 1), len(df) - HORIZON)
    if target_start is not None:
        o = o[ts[o + 1] >= pd.Timestamp(target_start)]
    if target_end is not None:
        o = o[ts[o + HORIZON] < pd.Timestamp(target_end)]
    if issue_at_midnight:
        o = o[(ts[o].hour == 23) & (ts[o].minute == 30)]
    if len(o) == 0:
        return {"n": 0}

    tgt = o[:, None] + np.arange(1, HORIZON + 1)[None, :]
    rn = (raw - mu) / sd
    prof = np.stack([raw[tgt - 48 * k] for k in range(1, 8)], 0).mean(0)
    week = raw[tgt - 336]
    prof_n, week_n = (prof - mu) / sd, (week - mu) / sd
    cal = _calendar(ts)[tgt]                                            # (n, 48, 5)
    Xf = np.concatenate([cal, prof_n[..., None], week_n[..., None]], axis=2).reshape(len(o), -1)
    Xh = np.lib.stride_tricks.sliding_window_view(hist, L, axis=0)[o - L + 1].transpose(0, 2, 1)
    return {
        "n": len(o),
        "Xh": np.ascontiguousarray(Xh, dtype="float32"),
        "Xf": Xf.astype("float32"),
        "y": (rn[tgt] - prof_n).astype("float32"),
        "base_n": prof_n.astype("float32"),
        "y_kwh": raw[tgt],
        "prof7_kwh": prof,
        "week_kwh": week,
        "ts": ts[tgt.ravel()],
        "mu": mu, "sd": sd,
    }


def predict_kwh(model, arr: dict, batch_size: int = 4096) -> np.ndarray:
    """Πρόβλεψη σε kWh (n, 48): έξοδος μοντέλου + προφίλ 7 ημερών, αποκανονικοποίηση, ≥ 0."""
    p = model.predict([arr["Xh"], arr["Xf"]], batch_size=batch_size, verbose=0).reshape(arr["n"], HORIZON)
    return np.clip((p + arr["base_n"]) * arr["sd"] + arr["mu"], 0, None)


def windows(training_date: str, train_days: int, eval_days: int):
    """
    Χρονικά παράθυρα day-ahead:
      train: δεδομένα [T0, T1), στόχοι από T0 + HISTORY_DAYS
      eval : στόχοι [T1, T1 + eval_days), με ιστορικό από T1 - HISTORY_DAYS
    """
    t0 = pd.Timestamp(training_date)
    t1 = t0 + pd.Timedelta(days=train_days)
    e1 = t1 + pd.Timedelta(days=eval_days)
    return t0, t1, e1


def metrics(y_kwh: np.ndarray, p_kwh: np.ndarray) -> dict:
    """Μετρικές σε kWh για ένα σπίτι ή ένα άθροισμα (πίνακες (n_days, 48) ή επίπεδοι)."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y, p = np.ravel(y_kwh), np.ravel(p_kwh)
    if len(y) == 0:
        return {}
    m = float(np.mean(y)) or 1e-9
    out = {"mse": float(mean_squared_error(y, p)), "mae": float(mean_absolute_error(y, p)),
           "r2": float(r2_score(y, p)), "nmae": float(mean_absolute_error(y, p) / m),
           "nrmse": float(np.sqrt(mean_squared_error(y, p)) / m), "bias": float(np.sum(p) / max(np.sum(y), 1e-9))}
    if np.ndim(y_kwh) == 2 and np.shape(y_kwh)[1] == HORIZON:
        dy, dp = np.sum(y_kwh, 1), np.sum(p_kwh, 1)
        out["day_err"] = float(np.mean(np.abs(dp - dy) / np.maximum(dy, 1e-9)))
    return out

