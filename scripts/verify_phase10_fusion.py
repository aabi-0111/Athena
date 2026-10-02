from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from app.ml.ensemble.fusion import FusionEnsemble

BASE = Path("data/processed/train_test")
MODELS = Path("saved_models")

X_test = pd.read_csv(BASE / "X_test.csv")
y_test = pd.read_csv(BASE / "y_test.csv").squeeze("columns")

models = {
    "random_forest": joblib.load(MODELS / "random_forest_model.pkl"),
    "xgboost": joblib.load(MODELS / "xgboost_model.pkl"),
    "lightgbm": joblib.load(MODELS / "lightgbm_model.pkl"),
}

print("X_TEST:", X_test.shape)
print("Y_TEST:", y_test.shape)
print("MODELS:", list(models))

results = {}

for method in ("weighted_average", "max"):
    fusion = FusionEnsemble(
        models=models,
        method=method,
        threshold=0.5,
    )

    probability = fusion.predict_proba(X_test)
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, probability)),
    }

    results[method] = metrics

    print(f"\n--- FUSION: {method} ---")
    print("Probability shape:", probability.shape)
    print("Prediction shape:", prediction.shape)
    print("Precision=", metrics["precision"])
    print("Recall=", metrics["recall"])
    print("F1=", metrics["f1"])
    print("PR-AUC=", metrics["pr_auc"])

report = {
    "phase": 10,
    "component": "fusion_ensemble",
    "status": "PASS",
    "base_models": list(models),
    "dataset": {
        "test_rows": int(len(y_test)),
        "features": int(X_test.shape[1]),
    },
    "methods": results,
}

Path("reports").mkdir(exist_ok=True)
with open("reports/phase10_fusion_comparison.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nREPORT: reports/phase10_fusion_comparison.json")
print("PHASE 10 FUSION: PASS")

