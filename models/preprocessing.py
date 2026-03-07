# models/preprocessing.py

import pandas as pd
import numpy as np

# -------------------------------
# Config / constants
# -------------------------------
SENSORS = ['ANKLE_LEFT','ANKLE_RIGHT','BELT','CHEST']
XYZ     = ['x','y','z']

# Raw 12D for LSTM
FEATURE_COLS_12D = [f"{s}_{axis}" for s in SENSORS for axis in XYZ]

# Tag mapping
TAG_MAP = {
    "010-000-024-033": "ANKLE_LEFT",
    "010-000-030-096": "ANKLE_RIGHT",
    "020-000-033-111": "CHEST",
    "020-000-032-221": "BELT"
}

# Activity mapping
ACTIVITY_MAP = {
    'walking': 'walking',
    'sitting down': 'sitting',
    'sitting': 'sitting',
    'sitting on the ground': 'sitting',
    'standing up from sitting': 'standing',
    'standing up from lying': 'standing',
    'standing up from sitting on the ground': 'standing',
    'lying': 'lying',
    'lying down': 'lying',
    'falling': 'falling',
    'on all fours': 'on_all_fours'
}


# -------------------------------
# Load and clean raw CSV
# -------------------------------
def load_and_clean(file_path):
    df = pd.read_csv(file_path, sep=",", header=None)
    df.columns = [
        "sequence_name", "tag_id", "timestamp",
        "date_time", "x", "y", "z", "activity"
    ]
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d.%m.%Y %H:%M:%S:%f")
    df[["x","y","z"]] = df[["x","y","z"]].astype(float)
    df = df.sort_values(['sequence_name', 'date_time']).reset_index(drop=True)
    df["body_position"] = df["tag_id"].map(TAG_MAP)
    df["label"] = df["activity"].apply(lambda x: 1 if x == "falling" else 0)
    df["activity_merged"] = df["activity"].map(ACTIVITY_MAP)
    return df


# -------------------------------
# Pivot to 12D (x,y,z per sensor)
# -------------------------------
def pivot_sensors_12D(df):
    for col in FEATURE_COLS_12D:
        df[col] = np.nan
    for s in SENSORS:
        mask = df['body_position'] == s
        for axis in XYZ:
            df.loc[mask, f"{s}_{axis}"] = df.loc[mask, axis]
    df[FEATURE_COLS_12D] = df.groupby('sequence_name')[FEATURE_COLS_12D].ffill().bfill()
    df = df.dropna(subset=FEATURE_COLS_12D)
    return df


# -------------------------------
# Build extra RF features (32 features)
# -------------------------------
def build_rf_features(df):
    df = df.copy()
    # Magnitude and jerk per sensor
    for s in SENSORS:
        axes = [f"{s}_{ax}" for ax in XYZ]
        df[f"{s}_mag"]  = np.sqrt(sum(df[c]**2 for c in axes))
        df[f"{s}_jerk"] = df.groupby('sequence_name')[f"{s}_mag"].diff().fillna(0)

    # Cross-sensor features
    mags = [f"{s}_mag" for s in SENSORS]
    df['mag_mean']  = df[mags].mean(axis=1)
    df['mag_std']   = df[mags].std(axis=1)
    df['mag_max']   = df[mags].max(axis=1)
    df['mag_range'] = df[mags].max(axis=1) - df[mags].min(axis=1)

    # Rolling window statistics (window=10 frames)
    W = 10
    for s in SENSORS:
        col = f"{s}_mag"
        df[f"{s}_roll_mean"] = df.groupby('sequence_name')[col].transform(
            lambda x: x.rolling(W, min_periods=1).mean()
        )
        df[f"{s}_roll_std"] = df.groupby('sequence_name')[col].transform(
            lambda x: x.rolling(W, min_periods=1).std().fillna(0)
        )

    # Final feature list for RF
    FEATURE_COLS_RF = (
        FEATURE_COLS_12D +                   # 12 raw
        [f"{s}_mag" for s in SENSORS] +     # 4 magnitudes
        [f"{s}_jerk" for s in SENSORS] +    # 4 jerks
        [f"{s}_roll_mean" for s in SENSORS] +  # 4 rolling means
        [f"{s}_roll_std"  for s in SENSORS] +  # 4 rolling stds
        ['mag_mean','mag_std','mag_max','mag_range']  # 4 cross-sensor
    )
    return df, FEATURE_COLS_RF


# -------------------------------
# Build sequences for LSTM
# -------------------------------
def build_sequences(df):
    X_sequences, y_sequences, groups = [], [], []
    for seq_name, group in df.groupby('sequence_name'):
        X_sequences.append(group[FEATURE_COLS_12D].values)
        y_sequences.append(group['activity_merged'].values)
        groups.append(seq_name)
    return np.array(X_sequences, dtype=object), np.array(y_sequences, dtype=object), np.array(groups)


# -------------------------------
# Full preprocessing pipeline
# -------------------------------
def preprocess_pipeline(file_path):
    """
    Returns:
        df: cleaned dataframe
        X_sequences: LSTM 12D sequences
        y_sequences: activity labels
        groups: sequence names
        feature_cols_rf: full 32-feature RF columns
    """
    df = load_and_clean(file_path)
    df = pivot_sensors_12D(df)
    X_sequences, y_sequences, groups = build_sequences(df)
    df, feature_cols_rf = build_rf_features(df)
    return df, X_sequences, y_sequences, groups, feature_cols_rf