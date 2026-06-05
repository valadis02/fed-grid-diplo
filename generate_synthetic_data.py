"""
generate_synthetic_data.py
Generates synthetic energy consumption data for 10 households.
Used for Chapter 7 experiments (FL vs Centralized, Hardware, Encryption, Pruning).

Output: data/house_X/input_data.csv for X in 1..10
Format: timestamp (15-min intervals), consumption_kwh
Duration: 4 days (2 training + 2 evaluation)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 4 days of 15-minute intervals
dates = pd.date_range(start="2024-01-01", periods=4*24*4, freq="15min")

def generate_house(profile, seed):
    np.random.seed(seed)
    n = len(dates)
    hours = np.array([d.hour + d.minute/60 for d in dates])

    if profile == "small_apartment":
        # Low consumption, peaks morning and evening
        base = 0.2 + 0.15 * np.sin((hours - 7) * np.pi / 6) + \
               0.12 * np.sin((hours - 19) * np.pi / 4)

    elif profile == "large_house":
        # High consumption, broad peaks
        base = 0.5 + 0.3 * np.sin((hours - 8) * np.pi / 7) + \
               0.25 * np.sin((hours - 20) * np.pi / 5)

    elif profile == "work_from_home":
        # Steady daytime usage
        daytime = ((hours >= 8) & (hours <= 18)).astype(float)
        base = 0.3 + 0.4 * daytime + 0.1 * np.sin((hours - 12) * np.pi / 6)

    elif profile == "elderly_couple":
        # Early morning and early evening peaks
        base = 0.25 + 0.2 * np.sin((hours - 6) * np.pi / 5) + \
               0.15 * np.sin((hours - 17) * np.pi / 4)

    elif profile == "night_profile":
        # Low daytime, high nighttime
        base = 0.15 + 0.35 * (1 - np.sin((hours - 14) * np.pi / 10))

    elif profile == "young_professional":
        # Low morning (leaves early), spike evening after work
        morning_low = np.exp(-0.5 * ((hours - 7) / 1.5) ** 2) * 0.1
        evening_spike = np.exp(-0.5 * ((hours - 20) / 2.0) ** 2) * 0.6
        base = 0.1 + morning_low + evening_spike

    elif profile == "family_with_kids":
        # High morning rush, afternoon after school, evening dinner
        morning = np.exp(-0.5 * ((hours - 7.5) / 1.5) ** 2) * 0.4
        afternoon = np.exp(-0.5 * ((hours - 15.5) / 1.5) ** 2) * 0.3
        evening = np.exp(-0.5 * ((hours - 19) / 2.0) ** 2) * 0.5
        base = 0.2 + morning + afternoon + evening

    elif profile == "ev_owner":
        # Normal household + large nighttime spike for EV charging
        household = 0.3 + 0.2 * np.sin((hours - 8) * np.pi / 8)
        ev_charging = np.exp(-0.5 * ((hours - 23) / 1.5) ** 2) * 1.2
        base = household + ev_charging

    elif profile == "remote_worker_heavy":
        # High steady daytime + multiple appliance peaks
        daytime = ((hours >= 7) & (hours <= 20)).astype(float)
        base = 0.4 + 0.5 * daytime + 0.2 * np.sin((hours - 12) * np.pi / 5) + \
               0.15 * np.sin((hours - 17) * np.pi / 3)

    elif profile == "irregular":
        # Unpredictable pattern with high variance
        base = 0.3 + 0.4 * np.random.rand(n) + \
               0.2 * np.sin((hours - np.random.uniform(0, 12)) * np.pi / 6)

    else:
        base = 0.3 * np.ones(n)

    noise = np.random.normal(0, 0.02, n)
    consumption = np.clip(base + noise, 0.01, None)
    return consumption


profiles = [
    ("house_1",  "small_apartment",     1),
    ("house_2",  "large_house",         2),
    ("house_3",  "work_from_home",      3),
    ("house_4",  "elderly_couple",      4),
    ("house_5",  "night_profile",       5),
    ("house_6",  "young_professional",  6),
    ("house_7",  "family_with_kids",    7),
    ("house_8",  "ev_owner",            8),
    ("house_9",  "remote_worker_heavy", 9),
    ("house_10", "irregular",          10),
]

out_root = Path("data")
for house_id, profile, seed in profiles:
    folder = out_root / house_id
    folder.mkdir(parents=True, exist_ok=True)
    consumption = generate_house(profile, seed)
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "consumption_kwh": np.round(consumption, 4)
    })
    out_path = folder / "input_data.csv"
    df.to_csv(out_path, index=False)
    print(f"[{house_id}] {profile:<22} | {len(df)} rows | mean={consumption.mean():.3f} kWh")

print("\nDone! Data generated in ./data/")