# Smart Home Fall Detection for Elderly Safety (Data-Driven Simulation)

## Overview

Falls are a leading cause of injury among elderly individuals living independently.

This project implements a data-driven fall detection system using motion sensor data collected from a smart home environment.

Instead of relying on a single classification model, the system compares two different sequence modeling approaches:

**Hidden Markov Models (HMM)** – generative probabilistic model

**LSTM (Long Short-Term Memory network)** – deep learning sequence model

The frontend provides an interactive Streamlit dashboard that performs sliding-window fall detection, visualizing fall probability over time.

---

## About the Dataset

**Context:**  
The dataset originates from a thesis focused on developing safer smart environments for independent living. It is collected from a **care-independent smart home** to detect falls among elderly people.

**Content:**  
- Each sample contains **three-dimensional sensor positions**: `X`, `Y`, `Z`.  
- Four sensors were attached to the person’s **chest, belt, and ankles**. Their activity is captured via **one-hot encoded columns**:  
  - `010-000-024-033`  
  - `010-000-030-096`  
  - `020-000-032-221`  
  - `020-000-033-111`  
- **Labels:**  
  - `0` → normal activity  
  - `1` → fall event  
- **Structure:**  
  - Each CSV file corresponds to **one person**.  
  - **Training set:** 20 persons (`data_1.csv` to `data_19.csv`)  
  - **Test set:** 5 persons (`data_20.csv` to `data_24.csv`)  
- **Note:** Timestamps were removed to avoid biased learning based on time sequences.

**Acknowledgements:**  
The dataset is a refactored version of the [UCI Localization Data for Person Activity dataset](https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity). All rights belong to the original authors:

> B. Kaluza, V. Mirchevska, E. Dovgan, M. Lustrek, M. Gams, *An Agent-based Approach to Care in Independent Living*, International Joint Conference on Ambient Intelligence (AmI-10), Malaga, Spain, In press.

---

## Project Structure
```
project_root/
│
├── data/
│   └── ConfLongDemo_JSI.txt
│
├── model/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── inference.py
│   │
│   ├── hmm/
│   │   ├── __init__.py
│   │   └── train.py
│   │
│   ├── lstm/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── train.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Backend Models

1. **LSTM Model:**  
   - Sequence-aware neural network

   - Binary classification: falling vs non-fall

   - Uses sigmoid probability output

   - Optimized decision threshold

   - Trained using full participant sequences

2. **HMM Model**  
   - One Gaussian HMM per activity

   - Diagonal covariance

   - Sequence likelihood comparison

   - Softmax normalization for probability comparison


---
## Detection Method
Instead of classifying entire sequences, the system uses: **Sliding Window Detection**

   - Window size (default: 100 timesteps)

   - Step size (default: 50 timesteps)

   - Each window is independently classified

  -  Produces probability curve over time

   This mimics real-world wearable fall detection systems.


---

## Running the Application

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```
2. **Models training (Run only once if not trained):**

- Train hmm model
```bash
python -m models.hmm.train
```

- Train lstm model - This might take a long time
```bash
python -m models.lstm.train
```

3. **Run the Streamlit app:**

```bash
streamlit run app.py
```

4. **Interact with the animation:**

   The dashboard allows:

      - Selecting participant sequences

      - Uploading custom CSV sensor data

      - Adjusting window size and step size

      - Running sliding window detection

   Visualizing:

      - LSTM fall probability over time

      - Detected fall windows

      - HMM activity predictions
