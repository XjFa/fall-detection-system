# models/lstm/train.py

import numpy as np
import torch
import torch.nn as nn
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve

from models.lstm.model import FallLSTM
from models.preprocessing import preprocess_pipeline
import torch
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# LSTM training function
# ---------------------------
def train_lstm(X_sequences, y_sequences,
               hidden_dim=128,
               num_layers=2,
               dropout=0.2,
               epochs=10,
               lr=1e-3):
    """
    Train a bidirectional LSTM for fall detection.
    Returns:
        model: trained PyTorch model
        scaler: fitted StandardScaler
        best_thresh: optimized threshold for 'falling' class
    """

    # ------------------------
    # Scale sequences
    # ------------------------
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_sequences))
    X_scaled = [scaler.transform(x) for x in X_sequences]

    input_dim = X_scaled[0].shape[1]

    # ------------------------
    # Initialize model
    # ------------------------
    model = FallLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # ------------------------
    # Compute positive class weight
    # ------------------------
    y_all = np.concatenate([(np.array(y) == "falling").astype(float) for y in y_sequences])
    pos_weight = torch.tensor([len(y_all)/(np.sum(y_all)+1e-5)-1], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------
    # Training loop
    # ------------------------
    model.train()
    for epoch in range(epochs):
        for X_seq, y_seq in zip(X_scaled, y_sequences):
            X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(device)
            y_bin = (np.array(y_seq) == "falling").astype(float)
            y_tensor = torch.tensor(y_bin, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

            logits = model(X_tensor)
            loss = criterion(logits, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed.")

    # ------------------------
    # Threshold optimization
    # ------------------------
    model.eval()
    y_true_all, y_prob_all = [], []

    with torch.no_grad():
        for X_seq, y_seq in zip(X_scaled, y_sequences):
            X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            y_true_bin = (np.array(y_seq) == "falling").astype(int)
            y_true_all.extend(y_true_bin)
            y_prob_all.extend(probs)

    precision, recall, thresholds = precision_recall_curve(y_true_all, y_prob_all)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print("Optimized threshold for 'falling':", best_thresh)

    return model, scaler, best_thresh, input_dim


# ---------------------------
# Save artifacts
# ---------------------------
def save_artifacts(model, scaler, threshold, input_dim,
                   hidden_dim=128, num_layers=2, dropout=0.2,
                   save_dir="models/lstm/artifacts"):
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "threshold": threshold
    }, os.path.join(save_dir, "model.pt"))

    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    print("✅ LSTM artifacts saved successfully.")


# ---------------------------
# CLI / direct execution
# ---------------------------
if __name__ == "__main__":
    # Load and preprocess data
    file_path = "data/ConfLongDemo_JSI.txt"
    df, X_sequences, y_sequences, groups, feature_cols = preprocess_pipeline(file_path)

    # Train
    model, scaler, threshold, input_dim = train_lstm(X_sequences, y_sequences)

    # Save
    save_artifacts(model, scaler, threshold, input_dim)