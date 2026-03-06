# Smart Home Fall Detection for Elderly Safety (Data-Driven Simulation)

## Overview

Falls are a leading cause of injury among elderly individuals living independently.  

This project implements a **data-driven fall detection system** using motion sensor data collected from a smart home environment.  

Instead of relying on a single classification model, the system compares multiple sequence modeling approaches:

| Model | Type | Purpose |
|-------|------|---------|
| **LSTM (Long Short-Term Memory)** | Deep learning sequence model(Baseline) | Sequence-aware fall detection |
| **Hidden Markov Model (HMM)** | Generative probabilistic | General activity modeling |
| **Random Forest + HMM Smoother** | Ensemble | Frame-level classification + temporal smoothing |
| **Hidden Semi-Markov Model (HSMM)** | Probabilistic with duration prior | Safety-critical fall detection |

The frontend provides an **interactive Streamlit dashboard** that performs sliding-window fall detection, visualizing fall probability over time.

---

## About the Dataset

### Context
This dataset originates from research on safer smart environments for independent living. It was collected to analyze human activity using **body-worn sensors**, with a focus on understanding movements and detecting falls in a controlled setting.

### Dataset Overview
- **Participants:** 5 individuals  
- **Repetitions:** Each participant performed the same scenario **five times**  
- **Sensors:** Four body-worn tags per participant:
  - `ANKLE_LEFT` → 010-000-024-033  
  - `ANKLE_RIGHT` → 010-000-030-096  
  - `CHEST` → 020-000-033-111  
  - `BELT` → 020-000-032-221  
- **Measurements:** 3D localization coordinates (`X`, `Y`, `Z`) for each sensor  
- **Activities:** walking, falling, lying down, sitting down, standing up, on all fours, and others (11 total)  
- **Instances:** 164,860  
- **Features:** 8 (sequence name, tag ID, timestamp, date, X, Y, Z, activity label)  
- **Missing values:** None  

### Data Structure
- Each row corresponds to a **single sensor reading** at a specific timestamp.  
- Sequence name identifies the participant and repetition: e.g., `A01` → Participant A, first scenario repetition.  
- Tag ID identifies the sensor.  
- `Activity` column is categorical with 11 possible activities.

### Variable Summary

| Column        | Description |
|---------------|-------------|
| Sequence Name | Participant & repetition (A01–E05) |
| Tag ID        | Sensor identifier (ANKLE_LEFT, ANKLE_RIGHT, CHEST, BELT) |
| Timestamp     | Unique numeric timestamp |
| Date          | Format: `dd.MM.yyyy HH:mm:ss:SSS` |
| X             | X-coordinate of the tag |
| Y             | Y-coordinate of the tag |
| Z             | Z-coordinate of the tag |
| Activity      | Human activity label (walking, falling, lying, etc.) |


### Acknowledgements
Refactored from the [UCI Localization Data for Person Activity dataset](https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity).  
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
│   ├── rf/
│   │   ├── __init__.py
│   │   └── train.py
├── app.py
├── requirements.txt
└── README.md
```
---
## Data Preprocessing

**General Preprocessing Steps For All Models:**

- Load dataset and assign descriptive column names  
- Convert `date_time` to pandas datetime and acceleration columns to float  
- Sort by `sequence_name` and `date_time` for chronological order  
- Check for missing values and verify data integrity  
- Map activity labels for binary or multi-class tasks  
- Apply **StandardScaler** for feature normalization  
- One-Hot encode categorical features (e.g., sensor ID)  
- Group by `sequence_name` for participant-level analysis  
- Compute derived features like **acceleration magnitude**

**Distinctive Preprocessing:**
- **Random Forest / HMM:** Sliding windows → statistical features per window (mean, std, min, max, skew, kurtosis)  
- **HMM / HSMM:** Sequence-level multivariate observations, optionally augmented with duration features for HSMM  

---

<img width="869" height="504" alt="image" src="https://github.com/user-attachments/assets/c102bc89-7eec-46d1-8e89-de826f1c9b49" />


---

## Backend Models

### 1. LSTM 
- Sequence-aware neural network
- Binary classification: falling vs non-fall
- Sigmoid probability output, optimized threshold
- Trained on full participant sequences

### 2. HMM
- One Gaussian HMM per activity
- Diagonal covariance
- Likelihood comparison and softmax normalization

### 3. Random Forest + HMM Smoother
- Stage 1: RF processes frame-level features → probability per activity  
- Stage 2: HMM smooths RF predictions via Viterbi decoder  
- Captures complex nonlinear boundaries + temporal coherence  

### 4. Hidden Semi-Markov Model (HSMM) - Not in Dashboard
- Explicit Poisson duration per activity  
- Segment-level Viterbi scoring  
- Prioritizes high recall on fall events

---
## Detection Method

- **Sliding Window Detection:**  
  - Default window size: 100 timesteps  
  - Step size: 50 timesteps  
- Classifies each window independently → produces probability curve over time  
- Mimics real-world wearable fall detection


---
## Exploratory Data Analysis
(Under Construction..)


---
## Model Results

| Model | Overall Accuracy | Falling Recall | Falling F1 | Notes |
|-------|----------------|----------------|------------|-------|
| 12D HMM | 0.72 | 0.40 | 0.35 | Baseline HMM |
| RF + HMM | 0.85 | 0.33 | 0.48 | Smooths RF predictions |
| HSMM | 0.44 | 0.93 | 0.78 | High recall for falls, low overall accuracy |

- RF+HMM achieves highest overall accuracy 
- HSMM maximizes safety-critical fall recall

## Interpretations
1. **Baseline Hidden Markov Model**


2. **Activity-Level 12D HMM Model**


3. **Random Forest (RF) + HMM Smoother**
<img width="604" height="382" alt="Screenshot 2026-03-03 080343" src="https://github.com/user-attachments/assets/3a785ecd-66a8-4905-b797-09d811c173db" />
<img width="630" height="450" alt="Screenshot 2026-03-03 080356" src="https://github.com/user-attachments/assets/bf71e785-f12a-46ea-bb5a-d7d8fed2d7eb" />

   - Achieved the highest overall accuracy of 0.85 and macro F1 of 0.74 across all tested models, outperforming the best previous pure HMM (12D model, 0.72 accuracy) by a meaningful margin.
   - The HMM smoothing layer contributed a +0.02 accuracy gain over RF alone (0.828 → 0.847) by penalising physically implausible frame-to-frame transitions
   - Confirms that temporal context adds value on top of frame-level classification
   - Walking and sitting are classified with high reliability (F1 0.93 and 0.92 respectively), reflecting that these activities have distinctive, consistent sensor signatures across subjects
   - Falling remains the weakest class at F1 0.48, with recall of only 0.33 — meaning the model misses roughly two-thirds of actual fall events at the frame level, which is a critical limitation for a fall detection system where false negatives carry high safety cost
   - The precision-recall imbalance for falling (precision 0.86, recall 0.33) suggests the model is conservative: when it predicts a fall it is usually correct, but it fails to flag the majority of actual falls, likely because falls occupy very few frames relative to the total dataset.

4. **Hidden Semi-Markov Model**
<img width="606" height="327" alt="Screenshot 2026-03-03 080529" src="https://github.com/user-attachments/assets/3e1893e1-6a02-4fa4-927f-065b62e8bc00" />

   - Despite having a low overall accuracy of 0.44, the HSMM achieved the highest falling recall of 0.93 and falling F1 of 0.78 across all tested models — outperforming the RF + HMM Smoother on the most safety-critical class by a substantial margin
   - This strength is directly attributable to the duration prior: the model penalises assigning "falling" to long, stable chunks, concentrating fall predictions on brief high-jerk segments where true falls occur
   - The model's poor performance on other classes, such as lying (F1 0.00) and sitting (F1 0.00), indicates that the duration distributions for static activities overlap heavily, causing the HSMM to misclassify these consistently — a known failure mode when activity durations are variable across subjects

---

## Deployment Strategy

- Use **HSMM** fall probability as **dedicated fall alert trigger**  
- Use **RF+HMM Smoother** for general activity state estimation  

---



## Running the Application

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```
**2. Train models (only if not already trained):**

- Train hmm model
```bash
python -m models.hmm.train
```

- Train lstm model - This might take a long time
```bash
python -m models.lstm.train
```

- Train RF+HMM model
```bash
python -m models.rf.train
```

**3. Run the Streamlit app**

```bash
streamlit run app.py
```

### Interact with the Fall Detection Dashboard

The dashboard provides an interactive interface for exploring participant sensor sequences and visualizing fall detection results.

**Features:**

- **Select participant sequences:**  
  Choose from sequences containing falls to analyze specific events.

- **Run sliding window detection:**  
  Execute LSTM, Random Forest, and HMM models on fixed-size sliding windows of sensor data.

**Visualizations:**

- **Sensor readings over time:**  
  View all accelerometer and gyroscope signals. Ground truth fall periods are marked with **red vertical lines**:  
  - Solid line → Fall start  
  - Dashed line → Fall end

- **LSTM & Random Forest fall probability:**  
  Window-level probabilities plotted as line charts. High-risk windows (predicted as falling) are highlighted in tables.

- **HMM activity predictions:**  
  Model-predicted activity classes over time are displayed as line plots.

- **Combined fall alerts:**  
  Identify windows where **any model predicts a fall**, summarized in a table for quick review.

This setup allows users to explore **ground truth vs. model predictions** interactively, making it easier to validate and analyze fall detection performance.