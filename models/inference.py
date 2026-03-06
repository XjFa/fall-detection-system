# models/inference.py

import torch
import joblib
import numpy as np
from models.lstm.model import FallLSTM

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load LSTM artifacts
# -----------------------------
lstm_checkpoint = torch.load(
    "models/lstm/artifacts/model.pt",
    map_location=device,
    weights_only=False
)

lstm_model = FallLSTM(
    input_dim=lstm_checkpoint["input_dim"],
    hidden_dim=lstm_checkpoint["hidden_dim"],
    num_layers=lstm_checkpoint["num_layers"],
    dropout=lstm_checkpoint["dropout"]
).to(device)

lstm_model.load_state_dict(lstm_checkpoint["model_state_dict"])
lstm_model.eval()

lstm_scaler = joblib.load("models/lstm/artifacts/scaler.pkl")
lstm_threshold = lstm_checkpoint["threshold"]


# -----------------------------
# Load HMM artifacts
# -----------------------------
hmm_models = joblib.load("models/hmm/artifacts/models.pkl")
hmm_scaler = joblib.load("models/hmm/artifacts/scaler.pkl")


# -----------------------------
# Load RF artifacts
# -----------------------------
rf_model = joblib.load("models/rf/artifacts/model.pkl")
rf_scaler = joblib.load("models/rf/artifacts/scaler.pkl")
rf_le = joblib.load("models/rf/artifacts/le.pkl")


# -----------------------------
# HMM helper
# -----------------------------
def predict_hmm_sequence(X_seq):
    """
    Predict a sequence using HMM models.
    Returns predicted activity and per-class probabilities.
    """
    X_scaled = hmm_scaler.transform(X_seq)

    scores = {
        act: model.score(X_scaled) / len(X_scaled)
        for act, model in hmm_models.items()
    }

    log_vals = np.array(list(scores.values()))
    probs = np.exp(log_vals - np.max(log_vals))
    probs /= probs.sum()

    prob_dict = dict(zip(scores.keys(), probs))
    pred_class = max(scores, key=scores.get)

    return pred_class, prob_dict


# -----------------------------
# RF helper
# -----------------------------
def predict_rf_sequence(X_seq):
    """
    Frame-level RF prediction aggregated over sequence
    """
    X_scaled = rf_scaler.transform(X_seq)
    probs_all_classes = rf_model.predict_proba(X_scaled)  # (T, n_classes)

    # Assume 'falling' class is labeled 1 in original encoding
    if "falling" in rf_le.classes_:
        fall_idx = np.where(rf_le.classes_ == "falling")[0][0]
        fall_probs = probs_all_classes[:, fall_idx]
    else:
        fall_probs = np.zeros(X_scaled.shape[0])

    mean_prob = float(fall_probs.mean())
    pred = "falling" if mean_prob > 0.5 else "non-fall"

    return pred, mean_prob


# -----------------------------
# Unified prediction
# -----------------------------
def predict_all(X_seq):
    """
    Predict using LSTM, HMM, and RF for a single sequence.

    Parameters:
        X_seq: numpy array of shape (T, 12) or full 32-feature array

    Returns:
        dict
    """
    if not isinstance(X_seq, np.ndarray):
        X_seq = np.array(X_seq)

    # -------- LSTM --------
    X_scaled_lstm = lstm_scaler.transform(X_seq)
    X_tensor = torch.tensor(X_scaled_lstm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = lstm_model(X_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    lstm_mean_prob = float(probs.mean())
    lstm_pred = "falling" if lstm_mean_prob > lstm_threshold else "non-fall"

    # -------- HMM --------
    hmm_pred, hmm_probs = predict_hmm_sequence(X_seq)

    # -------- RF --------
    rf_pred, rf_prob = predict_rf_sequence(X_seq)

    return {
        "lstm": {"prediction": lstm_pred, "fall_probability": lstm_mean_prob},
        "hmm": {"prediction": hmm_pred, "probabilities": hmm_probs},
        "rf": {"prediction": rf_pred, "fall_probability": rf_prob}
    }


# -----------------------------
# Optional standalone HMM
# -----------------------------
def predict_hmm(X_seq):
    pred_class, prob_dict = predict_hmm_sequence(X_seq)
    return {"prediction": pred_class, "probabilities": prob_dict}


# -----------------------------
# Optional standalone RF
# -----------------------------
def predict_rf(X_seq):
    pred, prob = predict_rf_sequence(X_seq)
    return {"prediction": pred, "fall_probability": prob}