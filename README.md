# Smart Home Fall Detection for Elderly Safety (Data-Driven Simulation)

## Overview

Falls are a leading cause of injury among elderly people living independently.  
This project implements a **data-driven fall detection system** using **pre-recorded sensor data** from smart home experiments.  

The system simulates a **smart home monitoring setup**:

- **Input:** X/Y/Z motion data from pre-recorded sensors attached to chest, belt, and ankles of test participants.  
- **Backend:** Bayesian network model predicts the probability of a fall event, handling missing or noisy data.  
- **Frontend:** Interactive stick-figure animation visualizes motion and highlights fall events by collapsing the figure when the fall probability exceeds a threshold.  

**Goal:** Demonstrate a realistic fall detection system for elderly safety using pre-recorded sensor data, providing a foundation for smart home monitoring and early-warning systems.

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
fall_detection_project/
├── data/                     # CSV datasets
│   ├── train/                # Training data (data_1.csv – data_19.csv)
│   └── test/                 # Test data (data_20.csv – data_24.csv)
├── notebooks/                # Jupyter notebooks for data exploration and preprocessing
│   └── data_processing.ipynb
├── animation/                # Stick-figure animation code
│   ├── __init__.py
│   └── stick_figure.py
├── results/
│   └── fall_detection_model.pkl   # Trained model saved here
├── app.py                    # Streamlit frontend integrating animation and model
├── requirements.txt          # Python dependencies
└── README.md                 # Project README
```

---

## Features

1. **Data Preprocessing:**  
   - Load CSVs from train/test folders  
   - Explore X/Y/Z sensor data and anomalies  
   - Aggregate per-person observations for model training

2. **Bayesian Network Modeling:**  
   - Handles incomplete data via probabilistic inference  
   - Computes fall probabilities for each observation  
   - Trained on the 20-person training dataset, validated on 5-person test set

3. **Interactive Animation Frontend:**  
   - Stick-figure visual representation of the person using sensor data  
   - Head, chest, belt, and ankle joints displayed  
   - Dynamic animation of X/Y/Z positions  
   - Fall probability triggers a **collapse animation** (head and chest drop, color turns red)

4. **Integration:**  
   - Frontend takes live or pre-recorded X/Y/Z inputs  
   - Backend Bayesian model predicts fall probability  
   - Frontend animates motion and highlights fall events in real-time

---

## Usage

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run the Streamlit app:**

```bash
streamlit run app.py
```

3. **Interact with the animation:**
   - Use sidebar sliders to simulate X/Y/Z movements
   - Observe the stick figure collapse when the fall probability exceeds the threshold
   - Switch between example motions or manual input
