import os
import pandas as pd
import numpy as np
from shared.utils import required_columns
from shared.logging_config import logger


# -------------------------
# Helpers (DEFENSIVE)
# -------------------------

def _get_value_series(df):
    """
    Defensive accessor for 'value' column.
    Handles pandas edge-case where duplicate columns
    cause df['value'] to return a DataFrame.
    """
    value = df['value']
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value


def normalize_columns(df, timedate_column_name, value_column_name):
    if timedate_column_name in df.columns and timedate_column_name != "datetime":
        df = df.rename(columns={timedate_column_name: "datetime"})
    if value_column_name in df.columns and value_column_name != "value":
        df = df.rename(columns={value_column_name: "value"})
    return df


# -------------------------
# Feature extraction
# -------------------------

def extract_date_features(dataframe):
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'], errors='coerce')
    dataframe = dataframe.dropna(subset=['datetime'])

    dataframe['year'] = dataframe['datetime'].dt.year
    dataframe['month'] = dataframe['datetime'].dt.month
    dataframe['day'] = dataframe['datetime'].dt.day
    dataframe['weekday'] = dataframe['datetime'].dt.weekday

    dataframe['month_sin'] = np.sin((dataframe['month'] - 1) * (2. * np.pi / 12))
    dataframe['month_cos'] = np.cos((dataframe['month'] - 1) * (2. * np.pi / 12))
    dataframe['weekday_sin'] = np.sin((dataframe['weekday'] - 1) * (2. * np.pi / 7))
    dataframe['weekday_cos'] = np.cos((dataframe['weekday'] - 1) * (2. * np.pi / 7))

    return dataframe


def extract_time_features(dataframe):
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'], errors='coerce')
    dataframe = dataframe.dropna(subset=['datetime'])

    dataframe['hour'] = dataframe['datetime'].dt.hour
    dataframe['minute'] = dataframe['datetime'].dt.minute

    dataframe['hour_sin'] = np.sin(dataframe['hour'] * (2. * np.pi / 24))
    dataframe['hour_cos'] = np.cos(dataframe['hour'] * (2. * np.pi / 24))
    dataframe['minute_sin'] = np.sin(dataframe['minute'] * (2. * np.pi / 60))
    dataframe['minute_cos'] = np.cos(dataframe['minute'] * (2. * np.pi / 60))

    return dataframe


# -------------------------
# Rolling features (SAFE)
# -------------------------

def add_rolling_features(dataframe, windows):
    value_series = _get_value_series(dataframe)

    for window in windows:
        dataframe[f'value_rolling_mean_{window}'] = (
            value_series
            .rolling(window=window)
            .mean()
            .interpolate(method='linear')
            .ffill()
            .bfill()
        )

    return dataframe


# -------------------------
# Advanced features (SAFE)
# -------------------------

def add_advanced_features(dataframe, windows):
    value_series = _get_value_series(dataframe)

    dataframe['value_diff'] = value_series.diff().fillna(0)

    for window in windows:
        dataframe[f'value_ewm_{window}'] = (
            value_series.ewm(span=window, adjust=False).mean()
        )

        dataframe[f'value_volatility_{window}'] = (
            value_series
            .rolling(window=window)
            .std()
            .fillna(0)
        )

    baseline_volatility = (
        value_series.rolling(window=6).std().fillna(0)
    )

    threshold = baseline_volatility * 2
    dataframe['drift_flag'] = (value_series.diff().abs() > threshold).astype(int)

    return dataframe


def add_time_since_last_spike(dataframe, flag_column='drift_flag'):
    time_since = []
    counter = 0

    for flag in dataframe[flag_column]:
        if flag == 1:
            counter = 0
        else:
            counter += 1
        time_since.append(counter)

    dataframe['time_since_last_spike'] = time_since
    return dataframe


# -------------------------
# MAIN PREPROCESS PIPELINE
# -------------------------

def preprocess_data(data_file_path, timedate_column_name, value_column_name):
    if not os.path.exists(data_file_path) or os.path.getsize(data_file_path) <= 0:
        raise ValueError("Data file does not exist or it is empty!")

    dataframe = pd.read_csv(
        data_file_path,
        low_memory=False,
        dtype={timedate_column_name: str}
    )

    # 🔒 Remove duplicate columns early
    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]

    if timedate_column_name not in dataframe.columns and "datetime" not in dataframe.columns:
        raise ValueError(
            f"{data_file_path} does not contain '{timedate_column_name}' or 'datetime'."
        )

    functions = [
        lambda df: normalize_columns(df, timedate_column_name, value_column_name),
        extract_date_features,
        extract_time_features,
        lambda df: df.replace('Null', np.nan),
        lambda df: df.dropna(subset=['value']),
        lambda df: df.astype({'value': 'float'}),
        lambda df: add_rolling_features(df, windows=[3, 6, 12, 24]),
        lambda df: add_advanced_features(df, windows=[3, 6, 12, 24]),
        add_time_since_last_spike
    ]

    for function in functions:
        dataframe = function(dataframe)

    dataframe = dataframe[required_columns]
    dataframe.to_csv(data_file_path, index=False)

    logger.info(f"Updated file saved at: {data_file_path}")
