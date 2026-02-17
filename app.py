# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import time
from animation.stick_figure import draw_stick_figure_3d

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="🧍 Stick-Figure Fall Detection", layout="centered")
st.title("🧍 Fall Detection Animation (3D)")

# -----------------------------
# Load trained model bundle
# -----------------------------
bundle = joblib.load("results/best_fall_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]
FALL_STATE = bundle["fall_state"]

# -----------------------------
# Sidebar controls: only x, y, z
# -----------------------------
st.sidebar.header("Person Position in Room")
x = st.sidebar.slider("X (Left/Right)", -5.0, 5.0, 0.0, 0.01)
y = st.sidebar.slider("Y (Forward/Back)", -5.0, 5.0, 0.0, 0.01)
z = st.sidebar.slider("Z (Height)", 0.0, 2.0, 1.0, 0.01)

# Start / Stop animation buttons
if 'animating' not in st.session_state:
    st.session_state['animating'] = False

start = st.sidebar.button("▶ Start Animation", key="start_btn")
stop  = st.sidebar.button("⏹ Stop Animation", key="stop_btn")

if start:
    st.session_state['animating'] = True
if stop:
    st.session_state['animating'] = False

# -----------------------------
# Placeholder and figure
# -----------------------------
placeholder = st.empty()
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Frame counter
if 't' not in st.session_state:
    st.session_state['t'] = 0

# Walking parameters
step_amp = 0.15   # how far legs move forward/back
step_speed = 0.3  # speed of leg oscillation
sway_amp = 0.05   # chest/head side sway

# -----------------------------
# Animation loop
# -----------------------------
while st.session_state['animating']:
    t = st.session_state['t']

    # -----------------------------
    # Animate legs (walking)
    # -----------------------------
    left_leg_z = -1.0 + step_amp * np.sin(step_speed * t)
    right_leg_z = -1.0 + step_amp * np.sin(step_speed * t + np.pi)  # opposite phase

    # Slight sway for chest/head
    sway = sway_amp * np.sin(step_speed * t)

    # Compute joint positions relative to x, y, z
    belt_pos    = np.array([x, y, z])
    chest_pos   = belt_pos + np.array([0.0 + sway, 0.0, 0.2])
    head_pos    = belt_pos + np.array([0.0 + sway, 0.0, 0.7])
    ankle_l_pos = belt_pos + np.array([-0.15, 0.0, left_leg_z])
    ankle_r_pos = belt_pos + np.array([ 0.15, 0.0, right_leg_z])

    # -----------------------------
    # Build model input (x, y, z only)
    # -----------------------------
    frame_dict = {"x": x, "y": y, "z": z}
    for col in feature_cols:
        if col not in ["x","y","z"]:
            frame_dict[col] = 0.0  # set remaining sensors to 0

    X_frame = pd.DataFrame([frame_dict])[feature_cols]

    # Scale positions
    if hasattr(scaler, "mean_"):
        X_frame[["x","y","z"]] = scaler.transform(X_frame[["x","y","z"]])

    # Compute fall probability
    _, posteriors = model.score_samples(X_frame.values)
    fall_prob = float(posteriors[:, FALL_STATE][0])

    # Draw stick figure
    draw_stick_figure_3d(ax, head_pos, chest_pos, belt_pos, ankle_l_pos, ankle_r_pos, fall_prob)
    placeholder.pyplot(fig)

    # Increment frame
    st.session_state['t'] += 1
    time.sleep(0.05)

    # Stop condition
    if not st.session_state['animating']:
        break
