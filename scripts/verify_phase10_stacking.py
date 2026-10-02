from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from app.ml.ensemble.stacking import StackingEnsemble


ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------
# Load existing Athena data
# ------------------------------------------------------------------

X_train = pd.read_csv(ROOT / "data/processed/train_test/X_train.csv")
y_train = pd.read_csv(ROOT / "data/processed/train_test/y_train.csv").squeeze("columns")

X_test = pd.read_csv(ROOT / "data/processed/train_test/X_test.csv")
y_test = pd.read_csv(ROOT / "data/processed/train_test/y_test.csv").squeeze("columns")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ------------------------------------------------------------------
# Load existing trained base models
# ------------------------------------------------------------------

models = {
    "random_forest": joblib.load(ROOT / "saved_models/random_forest_model.pkl"),
    "xgboost": joblib.load(ROOT / "saved_models/xgboost_model.pkl"),
    "lightgbm": joblib.load(ROOT / "saved_models/lightgbm_model.pkl"),
}

print("MODELS:", list(models))

# ------------------------------------------------------------------
# Create a validation set from TRAINING DATA ONLY.
#
# The saved base models are already trained, so this validation split
# is used only to verify the stacking implementation safely.
# ------------------------------------------------------------------

X_meta_train, X_validation, y_meta_train, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42,
)

print("META TRAIN:", X_meta_train.shape)
print("VALIDATION:", X_validation.shape)

# ------------------------------------------------------------------
# Fit stacking meta-model on validation data.
# ------------------------------------------------------------------

stacking = StackingEnsemble(
    models=models,
    threshold=0.5,
)

stacking.fit(X_validation, y_validation)

# ------------------------------------------------------------------
# Evaluate on untouched TEST data.
# ------------------------------------------------------------------

stack_probability = stacking.predict_proba(X_test)
stack_prediction = stacking.predict(X_test)

print("STACKING PROBABILITY SHAPE:", stack_probability.shape)
print("STACKING PREDICTION SHAPE:", stack_prediction.shape)

precision = precision_score(y_test, stack_prediction, zero_division=0)
recall = recall_score(y_test, stack_prediction, zero_division=0)
f1 = f1_score(y_test, stack_prediction, zero_division=0)
pr_auc = average_precision_score(y_test, stack_probability)

print()
print("--- STACKING ENSEMBLE ---")
print("Precision=", precision)
print("Recall=", recall)
print("F1=", f1)
print("PR-AUC=", pr_auc)

# ------------------------------------------------------------------
# Basic verification
# ------------------------------------------------------------------

assert stack_probability.shape == (len(X_test),)
assert stack_prediction.shape == (len(X_test),)
assert np.isfinite(stack_probability).all()
assert set(np.unique(stack_prediction)).issubset({0, 1})

print()
print("PHASE 10 STACKING: PASS")
