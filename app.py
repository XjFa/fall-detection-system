# app.py
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from models.inference import predict_all
from models.preprocessing import preprocess_pipeline

st.set_page_config(page_title="Fall Detection Demo", layout="wide")
st.title("📱 Fall Detection Demo (Sliding Window)")

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
            "rf_prob": pred["rf"]["fall_probability"],
            "rf_pred": pred["rf"]["prediction"],
            "hmm_pred": pred["hmm"]["prediction"]
        })
    return pd.DataFrame(results)

# ----------------------------
# Load dataset (cached for speed)
# ----------------------------
@st.cache_data
def load_demo(file_path):
    return preprocess_pipeline(file_path)

file_path = "data/ConfLongDemo_JSI.txt"
df, X_sequences, y_sequences, groups, feature_cols = load_demo(file_path)

# Keep sequences with falling
fall_seq_indices = [i for i, y_seq in enumerate(y_sequences) if "falling" in y_seq]
fall_sequences = {groups[i]: X_sequences[i] for i in fall_seq_indices}

# ----------------------------
# Sequence selection
# ----------------------------
seq_choice = st.selectbox(
    "Choose a demo sequence with fall",
    ["none"] + list(fall_sequences.keys())
)
sequence_array = None
y_sequence = None

if seq_choice != "none":
    idx = np.where(groups == seq_choice)[0][0]
    sequence_array = fall_sequences[seq_choice]
    y_sequence = y_sequences[idx]
    st.success(f"Loaded '{seq_choice}' — shape {sequence_array.shape}")

# ----------------------------
# Fixed sliding window
# ----------------------------
if sequence_array is not None:
    window_size = 100
    step_size = 50
    st.info(f"Sliding window: size={window_size}, step={step_size}")

# ----------------------------
# Run detection
# ----------------------------
if st.button("Run Detection") and sequence_array is not None:

    @st.cache_data
    def run_detection(X_seq, window_size, step_size):
        return predict_snippets(X_seq, window_size, step_size)

    snippet_df = run_detection(sequence_array, window_size, step_size)

    # ----------------------------
    # Sensor readings + vertical lines for ground truth fall periods
    # ----------------------------
    st.subheader("📊 Sensor Readings Over Time (with Ground Truth)")

    sensor_cols = [
        "ANKLE_LEFT_x","ANKLE_LEFT_y","ANKLE_LEFT_z",
        "ANKLE_RIGHT_x","ANKLE_RIGHT_y","ANKLE_RIGHT_z",
        "BELT_x","BELT_y","BELT_z",
        "CHEST_x","CHEST_y","CHEST_z"
    ]
    df_sensors = pd.DataFrame(sequence_array, columns=sensor_cols)
    df_sensors["timestep"] = df_sensors.index

    if y_sequence is not None:
        df_sensors["falling_gt"] = [1 if label=="falling" else 0 for label in y_sequence]

    # Melt sensor data for line chart
    df_melt = df_sensors.melt(
        id_vars=["timestep"],
        value_vars=sensor_cols,
        var_name="sensor",
        value_name="value"
    )

    sensor_chart = alt.Chart(df_melt).mark_line().encode(
        x="timestep",
        y="value",
        color="sensor"
    )

    # Compute vertical lines for fall periods
    if y_sequence is not None:
        df_sensors["fall_start"] = df_sensors["falling_gt"].diff().fillna(0) == 1
        df_sensors["fall_end"] = df_sensors["falling_gt"].diff().fillna(0) == -1

        fall_starts = df_sensors.index[df_sensors["fall_start"]].tolist()
        fall_ends = df_sensors.index[df_sensors["fall_end"]].tolist()
        if df_sensors["falling_gt"].iloc[0] == 1:
            fall_starts = [0] + fall_starts
        if df_sensors["falling_gt"].iloc[-1] == 1:
            fall_ends = fall_ends + [df_sensors.index[-1]]

        fall_intervals = pd.DataFrame({"start": fall_starts, "end": fall_ends})

        # Vertical lines
        start_lines = alt.Chart(fall_intervals).mark_rule(color="red", strokeWidth=2).encode(
            x="start"
        )
        end_lines = alt.Chart(fall_intervals).mark_rule(color="red", strokeWidth=2, strokeDash=[4,2]).encode(
            x="end"
        )

        final_chart = alt.layer(sensor_chart, start_lines, end_lines).interactive()
    else:
        final_chart = sensor_chart.interactive()

    st.altair_chart(final_chart, use_container_width=True)

    # ----------------------------
    # Model comparison dashboard
    # ----------------------------
    st.subheader("🧠 Model Comparison (Fall Probability)")
    col1, col2, col3 = st.columns(3)

    # ---- LSTM ----
    with col1:
        st.markdown("### LSTM")
        st.line_chart(snippet_df.set_index("start")["lstm_prob"])
        lstm_table = snippet_df[["start","end","lstm_prob","lstm_pred"]].copy()
        st.markdown("Window-level predictions (LSTM)")
        st.dataframe(lstm_table)
        high_risk_lstm = lstm_table[lstm_table["lstm_pred"]=="falling"]
        if len(high_risk_lstm) > 0:
            st.markdown("⚠ High Risk Windows")
            st.dataframe(high_risk_lstm)

    # ---- Random Forest ----
    with col2:
        st.markdown("### Random Forest + HMM Smoothing")
        st.line_chart(snippet_df.set_index("start")["rf_prob"])
        rf_table = snippet_df[["start","end","rf_prob","rf_pred"]].copy()
        st.markdown("Window-level predictions (RF)")
        st.dataframe(rf_table)
        high_risk_rf = rf_table[rf_table["rf_pred"]=="falling"]
        if len(high_risk_rf) > 0:
            st.markdown("⚠ High Risk Windows")
            st.dataframe(high_risk_rf)

    # ---- HMM ----
    with col3:
        st.markdown("### HMM")
        hmm_map = {"falling":1,"lying":2,"sitting":3,"standing":4,"walking":5,"on_all_fours":6}
        snippet_df["hmm_num"] = snippet_df["hmm_pred"].map(hmm_map)
        st.line_chart(snippet_df.set_index("start")["hmm_num"])
        hmm_table = snippet_df[["start","end","hmm_pred"]].copy()
        st.markdown("Window-level predictions (HMM)")
        st.dataframe(hmm_table)

    # ----------------------------
    # Combined fall alert
    # ----------------------------
    st.markdown("---")
    st.subheader("🚨 Combined Fall Alerts")
    snippet_df["combined_alert"] = np.where(
        (snippet_df["lstm_pred"]=="falling") | (snippet_df["rf_pred"]=="falling"),
        "falling",
        "non-fall"
    )
    combined_alert = snippet_df[snippet_df["combined_alert"]=="falling"]
    if len(combined_alert) > 0:
        st.markdown("Windows where at least one model predicts fall:")
        st.dataframe(combined_alert[["start","end","lstm_pred","rf_pred","hmm_pred"]])
    else:
        st.success("No fall detected by any model in this sequence.")

    # ----------------------------
    # Ground truth summary
    # ----------------------------
    st.markdown("---")
    st.subheader("Demo Ground Truth")
    st.write("True label: falling (demo sequence contains fall)")