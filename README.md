# Kaggle - Predicting F1 Pit Stops

A binary classification model that predicts whether an F1 driver will pit on the next lap (`PitNextLap`), built for the Kaggle Playground Series S6E5 competition.

---

## The Problem

Each row in the dataset represents a single lap driven by a driver in a race. The goal is to predict whether that driver will make a pit stop on the very next lap (1) or not (0). This is a binary classification problem with roughly a 20% positive rate — pit stops are relatively rare events.

---

## Pipeline

### 1. Data Loading and Feature Engineering — `dataset.py`

The raw CSV data contains a mix of categorical and numeric columns. Since XGBoost requires all numeric inputs, categorical columns are label encoded — each unique string value is mapped to an integer.

**Categorical columns encoded:**
- `Driver` — 887 unique driver codes (e.g. `VER`, `D109`)
- `Compound` — 5 tyre types: HARD, MEDIUM, SOFT, INTERMEDIATE, WET
- `Race` — 26 unique race names (e.g. `British Grand Prix`)

The encoding is built from the training data and then applied to the test set. Any value in the test set not seen during training gets mapped to `-1`.

**Numeric columns passed through directly:**
| Column | Description |
|---|---|
| `Year` | Season year |
| `PitStop` | Whether a pit stop occurred this lap |
| `LapNumber` | Current lap number in the race |
| `Stint` | Which stint the driver is on |
| `TyreLife` | How many laps the current tyres have done |
| `Position` | Current race position |
| `LapTime (s)` | Lap time in seconds |
| `LapTime_Delta` | Difference from the driver's average lap time |
| `Cumulative_Degradation` | Accumulated tyre degradation over the stint |
| `RaceProgress` | Fraction of the race completed (0–1) |
| `Position_Change` | Position gained/lost this lap |

This gives **14 features** per lap.

---

### 2. Training — `train.py`

Training uses **5-fold stratified cross-validation**. Stratified means each fold preserves the ~20% positive rate, giving a fair evaluation across all folds.

For each fold:
1. The data is split into 80% train / 20% validation
2. A fresh XGBoost model is trained
3. Training and validation accuracy are printed

After all folds, mean and standard deviation are shown so you can see how consistent the model is across different slices of the data.

You are then asked whether to train a final model on the full dataset and save it. If yes, it saves as `model01.json`, `model02.json`, etc. — auto-incrementing so previous runs are never overwritten.

---

### 3. Submission — `main.py`

Lists all saved models and lets you pick one (defaults to the latest). Generates predictions on the test set and saves a Kaggle-ready CSV as `submission01.csv`, `submission02.csv`, etc., with the number matching the model that was used.

---

## The Model — XGBoost Classifier

XGBoost (Extreme Gradient Boosting) is an ensemble of decision trees trained sequentially, where each new tree learns to correct the mistakes of the previous ones. It is well suited to this problem because:

- The features are a mix of numeric and encoded categorical values — trees handle this naturally without needing normalisation
- The relationships between features (e.g. tyre life + race progress + degradation) are non-linear and interaction-heavy — gradient boosting captures these automatically
- It is fast on tabular data at this scale (~440k rows, 14 features)

**Key hyperparameters and what they do:**

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 5000 | Maximum number of trees — early stopping will cut this short |
| `max_depth` | 4 | How deep each tree can grow — shallower trees reduce overfitting |
| `learning_rate` | 0.01 | How much each tree contributes — lower = more trees needed, but more stable |
| `early_stopping_rounds` | 50 | Stops training if validation loss hasn't improved in 50 rounds |
| `subsample` | 0.8 | Each tree only sees 80% of the training rows — adds randomness to prevent overfitting |
| `colsample_bytree` | 0.8 | Each tree only sees 80% of the features — similar effect |
| `min_child_weight` | 3 | Minimum data in a leaf — prevents splits on tiny groups |
| `gamma` | 0.1 | Minimum loss reduction required to make a split — conservative splitting |
| `reg_alpha` | 0.1 | L1 regularisation — pushes less important feature weights toward zero |
| `reg_lambda` | 1.0 | L2 regularisation — smooths weights to prevent large values |
| `tree_method` | hist | Histogram-based tree building — much faster on large datasets |

The objective is `binary:logistic` (outputs a probability, thresholded at 0.5 for the final prediction) and the evaluation metric during training is `logloss`, which penalises confident wrong predictions more heavily than accuracy alone would.

---

## Usage

```bash
# Set up environment
source venv/bin/activate
pip install -r requirements.txt

# Train (runs 5-fold CV, then asks if you want to save)
python3 train.py

# Generate submission (picks a saved model, writes matching submission CSV)
python3 main.py
```
