# models/preprocessing.py

import pandas as pd
import numpy as np

# -------------------------------
# Config / constants
# -------------------------------

SENSORS = ['ANKLE_LEFT','ANKLE_RIGHT','BELT','CHEST']
XYZ = ['x','y','z']
FEATURE_COLS = [f"{s}_{axis}" for s in SENSORS for axis in XYZ]

TAG_MAP = {
    "010-000-024-033": "ANKLE_LEFT",
    "010-000-030-096": "ANKLE_RIGHT",
    "020-000-033-111": "CHEST",
    "020-000-032-221": "BELT"
}

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
# Preprocessing Functions
# -------------------------------

def load_and_clean(file_path):
    """
    Loads raw CSV and applies initial cleaning, mappings, and features.
    """

    df = pd.read_csv(file_path, sep=",", header=None)

    # Assign column names
    df.columns = [
        "sequence_name",  # A01–E05 (person/session)
        "tag_id",         # sensor ID
        "timestamp",      # numeric unique timestamp
        "date_time",      # formatted date
        "x",              # x coordinate
        "y",              # y coordinate
        "z",              # z coordinate
        "activity"        # activity label
    ]

    # Convert types
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d.%m.%Y %H:%M:%S:%f")
    df[["x","y","z"]] = df[["x","y","z"]].astype(float)

    # Sort
    df = df.sort_values(['sequence_name', 'date_time']).reset_index(drop=True)

    # Map sensors
    df["body_position"] = df["tag_id"].map(TAG_MAP)

    # Binary fall label
    df["label"] = df["activity"].apply(lambda x: 1 if x == "falling" else 0)

    # Magnitude of acceleration
    df["acc_mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

    # Merge activity labels
    df["activity_merged"] = df["activity"].map(ACTIVITY_MAP)

    return df


def pivot_sensors_12D(df):
    """
    Converts long-format sensor data to 12D (x,y,z for 4 sensors)
    """
    # Initialize columns
    for col in FEATURE_COLS:
        df[col] = np.nan

    for s in SENSORS:
        mask = df['body_position'] == s
        for axis in XYZ:
            df.loc[mask, f"{s}_{axis}"] = df.loc[mask, axis]

    # Forward/backward fill
    df[FEATURE_COLS] = df.groupby('sequence_name')[FEATURE_COLS].ffill().bfill()

    # Drop sequences with missing sensors
    df = df.dropna(subset=FEATURE_COLS)

    return df


def build_sequences(df):
    """
    Build sequences per class for LSTM/HMM.
    Returns:
        X_sequences: list of (T, 12) arrays
        y_sequences: list of (T,) arrays
        groups: list of sequence names
    """
    X_sequences, y_sequences, groups = [], [], []

    for seq_name, group in df.groupby('sequence_name'):
        X_sequences.append(group[FEATURE_COLS].values)
        y_sequences.append(group['activity_merged'].values)
        groups.append(seq_name)

    X_sequences = np.array(X_sequences, dtype=object)
    y_sequences = np.array(y_sequences, dtype=object)
    groups = np.array(groups)

    return X_sequences, y_sequences, groups


# -------------------------------
# Full preprocessing pipeline
# -------------------------------

def preprocess_pipeline(file_path):
    """
    Runs full preprocessing: load -> pivot -> build sequences
    """
    df = load_and_clean(file_path)
    df = pivot_sensors_12D(df)
    X_sequences, y_sequences, groups = build_sequences(df)
    return df, X_sequences, y_sequences, groups, FEATURE_COLS