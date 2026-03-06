# models/rf/train.py

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight

from models.preprocessing import preprocess_pipeline

# ---------------------------
# HMM smoother (Viterbi)
# ---------------------------
def hmm_smooth(posteriors, transition_matrix, n_classes):
    T = len(posteriors)
    log_trans = np.log(transition_matrix + 1e-10)
    log_emit  = np.log(posteriors + 1e-10)

    viterbi = np.full((T, n_classes), -np.inf)
    backptr = np.zeros((T, n_classes), dtype=int)

    viterbi[0] = log_emit[0] - np.log(n_classes)

    for t in range(1, T):
        for c in range(n_classes):
            candidates    = viterbi[t-1] + log_trans[:, c]
            best_prev     = np.argmax(candidates)
            viterbi[t, c] = candidates[best_prev] + log_emit[t, c]
            backptr[t, c] = best_prev

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[T-1] = np.argmax(viterbi[T-1])
    for t in range(T-2, -1, -1):
        path[t] = backptr[t+1, path[t+1]]

    return path

# ---------------------------
# RF training with LOSO
# ---------------------------
def train_rf(X_sequences, y_sequences, groups,
             n_estimators=200, max_depth=None, min_samples_leaf=2):
    le = LabelEncoder()
    all_labels = np.concatenate(y_sequences)
    le.fit(all_labels)
    n_classes = len(le.classes_)

    logo = LeaveOneGroupOut()
    all_y_true_rf, all_y_pred_rf = [], []
    all_y_true_hybrid, all_y_pred_hybrid = [], []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_sequences, groups=groups)):
        # Flatten sequences for frame-level RF training
        X_train = np.vstack([X_sequences[i] for i in train_idx])
        y_train = np.concatenate([y_sequences[i] for i in train_idx])
        y_train_enc = le.transform(y_train)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Build transition matrix for HMM smoothing
        y_train_seqs_enc = [le.transform(y_sequences[i]) for i in train_idx]
        trans_matrix = np.full((n_classes, n_classes), 1.0)  # Laplace smoothing
        for y_seq in y_train_seqs_enc:
            for t in range(len(y_seq)-1):
                trans_matrix[y_seq[t], y_seq[t+1]] += 1
        trans_matrix /= trans_matrix.sum(axis=1, keepdims=True)

        # Class weights for imbalance
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train_enc),
            y=y_train_enc
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight_dict,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train_scaled, y_train_enc)

        # Evaluate
        for idx in test_idx:
            X_test_seq = scaler.transform(X_sequences[idx])
            y_test_enc = le.transform(y_sequences[idx])

            rf_posteriors = rf.predict_proba(X_test_seq)
            rf_preds = np.argmax(rf_posteriors, axis=1)
            smoothed_preds = hmm_smooth(rf_posteriors, trans_matrix, n_classes)

            all_y_true_rf.extend(y_test_enc)
            all_y_pred_rf.extend(rf_preds)
            all_y_true_hybrid.extend(y_test_enc)
            all_y_pred_hybrid.extend(smoothed_preds)

        print(f"Completed fold {fold+1}/{len(groups)}")

    # Print results
    print("\n" + "="*55)
    print("RANDOM FOREST ONLY — Frame-Level Results")
    print("="*55)
    print(classification_report(
        le.inverse_transform(all_y_true_rf),
        le.inverse_transform(all_y_pred_rf),
        zero_division=0
    ))
    print("="*55)
    print("RF + HMM SMOOTHER — Frame-Level Results")
    print("="*55)
    print(classification_report(
        le.inverse_transform(all_y_true_hybrid),
        le.inverse_transform(all_y_pred_hybrid),
        zero_division=0
    ))

    acc_rf = accuracy_score(all_y_true_rf, all_y_pred_rf)
    acc_hybrid = accuracy_score(all_y_true_hybrid, all_y_pred_hybrid)
    print(f"RF Only Accuracy:      {acc_rf:.3f}")
    print(f"RF + HMM Accuracy:     {acc_hybrid:.3f}")
    print(f"Smoothing improvement: {acc_hybrid - acc_rf:+.3f}")

    return rf, scaler, le

# ---------------------------
# Save artifacts
# ---------------------------
def save_artifacts(model, scaler, le, save_dir="models/rf/artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(save_dir, "le.pkl"))
    print("✅ RF artifacts saved successfully.")

# ---------------------------
# CLI / direct execution
# ---------------------------
if __name__ == "__main__":
    file_path = "data/ConfLongDemo_JSI.txt"

    # Load preprocessing (32-feature RF)
    df, X_sequences, y_sequences, groups, feature_cols = preprocess_pipeline(file_path)

    # Train RF with LOSO
    rf_model, scaler, le = train_rf(X_sequences, y_sequences, groups)

    # Save artifacts
    save_artifacts(rf_model, scaler, le)