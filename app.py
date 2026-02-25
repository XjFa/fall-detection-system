# app.py
import streamlit as st
import numpy as np
import pandas as pd
from models.inference import predict_all
from models.preprocessing import preprocess_pipeline

st.set_page_config(page_title="Fall Detection Demo", layout="wide")
st.title("Fall Detection Demo (Sliding Window Detection)")

# ----------------------------
# Sliding window helper
# ----------------------------
def predict_snippets(X_seq, window_size=100, step_size=50):
    results = []

    for start in range(0, len(X_seq) - window_size + 1, step_size):
        end = start + window_size
        window = X_seq[start:end]

        pred = predict_all(window)

        results.append({
            "start": start,
            "end": end,
            "lstm_prob": pred["lstm"]["fall_probability"],
            "lstm_pred": pred["lstm"]["prediction"],
            "hmm_pred": pred["hmm"]["prediction"]
        })

    return pd.DataFrame(results)


# ----------------------------
# Load dataset
# ----------------------------
file_path = "data/ConfLongDemo_JSI.txt"
df, X_sequences, y_sequences, groups, feature_cols = preprocess_pipeline(file_path)

# Keep only sequences containing falling
fall_seq_indices = [
    i for i, y_seq in enumerate(y_sequences)
    if "falling" in y_seq
]

fall_sequences = {groups[i]: X_sequences[i] for i in fall_seq_indices}

# ----------------------------
# Sequence selection
# ----------------------------
seq_choice = st.selectbox(
    "Choose a falling sequence",
    ["none"] + list(fall_sequences.keys())
)

sequence_array = None

if seq_choice != "none":
    sequence_array = fall_sequences[seq_choice]
    st.success(f"Loaded '{seq_choice}' — shape {sequence_array.shape}")

# ----------------------------
# CSV upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Or upload your sensor CSV (12 columns: x,y,z for 4 sensors)", type="csv"
)

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    sequence_array = df_uploaded.values
    st.success(f"Uploaded CSV with shape {sequence_array.shape}")

# ----------------------------
# Window controls
# ----------------------------
if sequence_array is not None:

    st.subheader("Sliding Window Settings")

    col1, col2 = st.columns(2)

    with col1:
        window_size = st.slider("Window Size (timesteps)", 50, 300, 100)

    with col2:
        step_size = st.slider("Step Size", 10, 150, 50)

# ----------------------------
# Run detection
# ----------------------------
if st.button("Run Sliding Window Detection") and sequence_array is not None:

    snippet_df = predict_snippets(sequence_array, window_size, step_size)

    st.subheader("Fall Probability Over Time (LSTM)")
    st.line_chart(snippet_df["lstm_prob"])

    st.subheader("Detected Windows")
    st.dataframe(snippet_df)

    # Show high-risk windows
    high_risk = snippet_df[snippet_df["lstm_pred"] == "falling"]

    st.subheader("⚠ High Risk Windows (LSTM)")
    if len(high_risk) > 0:
        st.dataframe(high_risk)
    else:
        st.success("No fall detected in any window.")

    st.markdown("---")

    st.subheader("HMM Window Predictions")
    hmm_counts = snippet_df["hmm_pred"].value_counts()
    st.bar_chart(hmm_counts)

    st.markdown("### Demo Ground Truth")
    st.write("True label: falling (demo sequence contains fall)")