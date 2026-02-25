# models/hmm/train.py

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

from models.preprocessing import preprocess_pipeline



# -----------------------------
# Configuration
# -----------------------------

STATE_MAP = {
    'falling': 3,
    'walking': 4,
    'standing': 3,
    'sitting': 3,
    'lying': 2,
    'on_all_fours': 3
}


# -----------------------------
# Build activity chunks
# -----------------------------

def build_chunks(X_sequences, y_sequences):
    """
    Splits sequences into activity-homogeneous chunks
    required for per-class HMM training.
    """

    train_class_sequences = {}

    for seq, labels in zip(X_sequences, y_sequences):

        prev = labels[0]
        start = 0

        for t in range(1, len(labels)):
            if labels[t] != prev:
                chunk = seq[start:t]
                train_class_sequences.setdefault(prev, []).append(chunk)
                start = t
                prev = labels[t]

        # last chunk
        chunk = seq[start:]
        train_class_sequences.setdefault(prev, []).append(chunk)

    return train_class_sequences


# -----------------------------
# Training Function
# -----------------------------

def train_hmm(X_sequences, y_sequences):
    """
    Train one GaussianHMM per activity class.
    Returns:
        models: dict of activity -> trained HMM
        scaler: fitted StandardScaler
    """

    # ------------------------
    # Build activity chunks
    # ------------------------
    train_class_sequences = build_chunks(X_sequences, y_sequences)

    all_chunks = [
        chunk
        for seqs in train_class_sequences.values()
        for chunk in seqs
    ]

    # ------------------------
    # Scale features
    # ------------------------
    scaler = StandardScaler()
    scaler.fit(np.vstack(all_chunks))

    # Scale per class
    for act in train_class_sequences:
        train_class_sequences[act] = [
            scaler.transform(chunk)
            for chunk in train_class_sequences[act]
        ]

    # ------------------------
    # Train one HMM per activity
    # ------------------------
    models = {}

    for act, seqs in train_class_sequences.items():

        X_concat = np.vstack(seqs)
        lengths = [len(s) for s in seqs]

        n_states = STATE_MAP.get(act, 3)

        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=200,
            random_state=42
        )

        model.fit(X_concat, lengths)
        models[act] = model

        print(f"Trained HMM for class '{act}' with {n_states} states.")

    return models, scaler


# -----------------------------
# Save Artifacts
# -----------------------------

def save_artifacts(models, scaler,
                   save_dir="models/hmm/artifacts"):
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(models, os.path.join(save_dir, "models.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    print("✅ HMM artifacts saved successfully.")


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":

    # Load and preprocess using shared pipeline
    file_path = "data/ConfLongDemo_JSI.txt"
    df, X_sequences, y_sequences, groups, feature_cols = preprocess_pipeline(file_path)

    # Train
    models, scaler = train_hmm(X_sequences, y_sequences)

    # Save
    save_artifacts(models, scaler)