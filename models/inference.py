# model/inference.py

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
    weights_only=False  # <-- FIX for PyTorch 2.6+
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

    # Softmax for normalized probabilities
    log_vals = np.array(list(scores.values()))
    probs = np.exp(log_vals - np.max(log_vals))
    probs /= probs.sum()

    prob_dict = dict(zip(scores.keys(), probs))
    pred_class = max(scores, key=scores.get)

    return pred_class, prob_dict


# -----------------------------
# Unified prediction
# -----------------------------
def predict_all(X_seq):
    """
    Predict using both LSTM and HMM for a single sequence.

    Parameters:
        X_seq: numpy array of shape (T, 12)

    Returns:
        dict
    """

    if not isinstance(X_seq, np.ndarray):
        X_seq = np.array(X_seq)

    # -------- LSTM --------
    X_scaled_lstm = lstm_scaler.transform(X_seq)
    X_tensor = torch.tensor(
        X_scaled_lstm,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = lstm_model(X_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    lstm_mean_prob = float(probs.mean())
    lstm_pred = "falling" if lstm_mean_prob > lstm_threshold else "non-fall"

    # -------- HMM --------
    hmm_pred, hmm_probs = predict_hmm_sequence(X_seq)

    return {
        "lstm": {
            "prediction": lstm_pred,
            "fall_probability": lstm_mean_prob
        },
        "hmm": {
            "prediction": hmm_pred,
            "probabilities": hmm_probs
        }
    }


# -----------------------------
# Optional standalone HMM
# -----------------------------
def predict_hmm(X_seq):
    pred_class, prob_dict = predict_hmm_sequence(X_seq)
    return {
        "prediction": pred_class,
        "probabilities": prob_dict
    }