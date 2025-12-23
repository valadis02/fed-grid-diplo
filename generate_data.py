import pandas as pd
import numpy as np

start = pd.Timestamp("2024-01-01 00:00:00")
end = pd.Timestamp("2024-01-07 00:00:00")
timestamps = pd.date_range(start, end, freq="10min", inclusive="left")

base = 1.2
noise = np.random.normal(0, 0.05, len(timestamps))
trend = np.linspace(0, 0.3, len(timestamps))
values = base + noise + trend

df = pd.DataFrame({
    "datetime": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
    "apparent power (kWh)": values.round(3),
    "value": values.round(3)
})

df.to_csv("edge/data/input_data.csv", index=False)
print("CSV generated:", len(df), "rows")
