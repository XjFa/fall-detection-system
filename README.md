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

3. **Random Forest (RF) + HMM Smoother**
   - Random Forest directly learns the discriminative boundary P(activity | features)
     
   - Addresses the core limitation of pure generative HMM approaches of modelling P(features | activity)

4. **Hidden Semi-Markov Model (HSMM)**
   - Selected for fall detection
     
   - Incorporates explicit duration modelling

---
## Detection Method
Instead of classifying entire sequences, the system uses: **Sliding Window Detection**

   - Window size (default: 100 timesteps)

   - Step size (default: 50 timesteps)

   - Each window is independently classified

  -  Produces probability curve over time

   This mimics real-world wearable fall detection systems.


---
## Exploratory Data Analysis




---
## Model Results & Interpretations
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
- Across all models tested, falling consistently had the lowest or near-lowest F1, which reflects the class's inherent difficulty: it is the rarest activity, has the shortest duration, and its sensor signature partially overlaps with transitional activities like standing up from lying
- The RF + HMM and HSMM models represent a fundamental tradeoff: the prior optimises overall accuracy and is well-suited for general activity recognition, while the HSMM optimises fall-specific recall and is better suited as a safety-critical alerting system where missing a fall is more costly than a false alarm
- We believe it's best to combine both for practical deployment: **use the HSMM's falling probability as a dedicated fall alert trigger while using the RF + HMM Smoother for overall activity state estimation**



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
