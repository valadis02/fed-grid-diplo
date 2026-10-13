"""
shared/normalization.py
=======================
Per-house (per-client) normalization για το LCL forecasting pipeline.

Γιατί: τα σπίτια έχουν πολύ διαφορετική κλίμακα κατανάλωσης (π.χ. μέσος
0.08 kWh έως >1 kWh). Χωρίς κανονικοποίηση, το global μοντέλο κυριαρχείται
από λίγα σπίτια και το MLP καταρρέει σε σταθερή πρόβλεψη (dead ReLUs).

Κανόνες (στατιστικά ΜΟΝΟ από το training split του ίδιου σπιτιού, χωρίς
διαρροή στο evaluation· κάθε client υπολογίζει τα δικά του μ, σ τοπικά):
  value, value_rolling_mean_*, value_ewm_*   ->  (x - μ) / σ
  value_volatility_*, value_diff             ->  x / σ
  time_since_last_spike                      ->  x / TSLS_SCALE (48 = 1 μέρα)
  drift_flag                                 ->  αμετάβλητο (0/1)

Το target ('value') κανονικοποιείται με τον ίδιο τρόπο· οι προβλέψεις
γυρίζουν σε kWh με inverse_target() για metrics συγκρίσιμα με προηγούμενα runs.

Ενεργοποίηση/απενεργοποίηση: env NORMALIZATION=per_house (default) | none
"""
import json
import os

import numpy as np
import pandas as pd

NORMALIZATION = os.getenv("NORMALIZATION", "per_house").lower()
TSLS_SCALE = float(os.getenv("TSLS_SCALE", "48"))
_EPS = 1e-6


def enabled() -> bool:
    return NORMALIZATION == "per_house"


def fit_stats(train_df: pd.DataFrame) -> dict:
    """Υπολογίζει μ, σ του 'value' στο training split."""
    v = pd.to_numeric(train_df["value"], errors="coerce").dropna()
    mu = float(v.mean())
    sd = float(v.std())
    if not np.isfinite(sd) or sd < _EPS:
        sd = 1.0
    return {"mu": mu, "sd": sd, "tsls_scale": TSLS_SCALE, "mode": NORMALIZATION}


def transform_df(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Εφαρμόζει την κανονικοποίηση σε DataFrame με τις στήλες του pipeline."""
    if not enabled() or stats is None:
        return df
    df = df.copy()
    mu, sd = stats["mu"], stats["sd"]
    for c in df.columns:
        if c == "value" or c.startswith("value_rolling_mean_") or c.startswith("value_ewm_"):
            df[c] = (df[c].astype(float) - mu) / sd
        elif c.startswith("value_volatility_") or c == "value_diff":
            df[c] = df[c].astype(float) / sd
        elif c == "time_since_last_spike":
            df[c] = df[c].astype(float) / stats.get("tsls_scale", TSLS_SCALE)
    return df


def transform_csv(path: str, stats: dict) -> None:
    """Κανονικοποιεί in-place ένα προεπεξεργασμένο CSV (μετά το preprocess_data)."""
    if not enabled() or stats is None:
        return
    df = pd.read_csv(path)
    transform_df(df, stats).to_csv(path, index=False)


def inverse_target(y, stats: dict):
    """Επιστρέφει τιμές target/προβλέψεις σε kWh."""
    y = np.asarray(y, dtype=np.float64)
    if not enabled() or stats is None:
        return y
    return y * stats["sd"] + stats["mu"]


def save_stats(stats: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

        