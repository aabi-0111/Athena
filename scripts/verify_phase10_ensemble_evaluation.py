from pathlib import Path
from datetime import datetime, timezone

import sys
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.ensemble.voting import VotingEnsemble
from app.ml.ensemble.stacking import StackingEnsemble
from app.ml.ensemble.fusion import FusionEnsemble


BASE = Path("data/processed/train_test")
MODELS = Path("saved_models")
REPORT = Path("reports/phase10_ensemble_evaluation.json")


# ---------------------------------------------------------
# 1. Load test data
# ---------------------------------------------------------

X_test = pd.read_csv(BASE / "X_test.csv")
y_test = pd.read_csv(BASE / "y_test.csv").squeeze("columns")

print("X_TEST:", X_test.shape)
print("Y_TEST:", y_test.shape)


# ---------------------------------------------------------
# 2. Load candidate models
# ---------------------------------------------------------

models = {
    "random_forest": joblib.load(
        MODELS / "random_forest_model.pkl"
    ),
    "xgboost": joblib.load(
        MODELS / "xgboost_model.pkl"
    ),
    "lightgbm": joblib.load(
        MODELS / "lightgbm_model.pkl"
    ),
}

print("MODELS:", list(models))


# ---------------------------------------------------------
# 3. Evaluation helper
# ---------------------------------------------------------

def evaluate(name, probability, prediction):
    metrics = {
        "precision": float(
            precision_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                prediction,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_test,
                probability,
            )
        ),
    }

    print(f"\n--- {name} ---")
    print("Precision =", metrics["precision"])
    print("Recall    =", metrics["recall"])
    print("F1        =", metrics["f1"])
    print("PR-AUC    =", metrics["pr_auc"])

    return metrics


results = {}


# ---------------------------------------------------------
# 4. Individual model baselines
# ---------------------------------------------------------

for name, model in models.items():
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    results[name] = evaluate(
        name,
        probability,
        prediction,
    )


# ---------------------------------------------------------
# 5. Voting ensemble
# ---------------------------------------------------------

voting = VotingEnsemble(
    models=models,
    threshold=0.5,
)

voting_probability = voting.predict_proba(X_test)
voting_prediction = voting.predict(X_test)

results["voting"] = evaluate(
    "Voting Ensemble",
    voting_probability,
    voting_prediction,
)


# ---------------------------------------------------------
# 6. Stacking ensemble
# ---------------------------------------------------------

X_train = pd.read_csv(BASE / "X_train.csv")
y_train = pd.read_csv(BASE / "y_train.csv").squeeze("columns")

from sklearn.model_selection import train_test_split

X_meta_train, X_validation, y_meta_train, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train,
)

stacking = StackingEnsemble(
    models=models,
    threshold=0.5,
)

stacking.fit(
    X_validation,
    y_validation,
)

stacking_probability = stacking.predict_proba(X_test)
stacking_prediction = stacking.predict(X_test)

results["stacking"] = evaluate(
    "Stacking Ensemble",
    stacking_probability,
    stacking_prediction,
)


# ---------------------------------------------------------
# 7. Fusion - weighted average
# ---------------------------------------------------------

fusion_average = FusionEnsemble(
    models=models,
    threshold=0.5,
    method="weighted_average",
)

fusion_average_probability = fusion_average.predict_proba(X_test)
fusion_average_prediction = fusion_average.predict(X_test)

results["fusion_weighted_average"] = evaluate(
    "Fusion - Weighted Average",
    fusion_average_probability,
    fusion_average_prediction,
)


# ---------------------------------------------------------
# 8. Fusion - max
# ---------------------------------------------------------

fusion_max = FusionEnsemble(
    models=models,
    threshold=0.5,
    method="max",
)

fusion_max_probability = fusion_max.predict_proba(X_test)
fusion_max_prediction = fusion_max.predict(X_test)

results["fusion_max"] = evaluate(
    "Fusion - Max",
    fusion_max_probability,
    fusion_max_prediction,
)


# ---------------------------------------------------------
# 9. Save final report
# ---------------------------------------------------------

timestamp = datetime.now(timezone.utc)
timestamp_iso = timestamp.isoformat()
timestamp_file = timestamp.strftime("%Y%m%d_%H%M%S")

report = {
    "phase": 10,
    "component": "ensemble_evaluation",
    "status": "PASS",
    "timestamp_utc": timestamp_iso,
    "dataset": {
        "test_rows": int(len(y_test)),
        "features": int(X_test.shape[1]),
    },
    "base_models": list(models),
    "evaluated_components": [
        "random_forest",
        "xgboost",
        "lightgbm",
        "voting",
        "stacking",
        "fusion_weighted_average",
        "fusion_max",
    ],
    "metrics": results,
}

# Latest report
REPORT.parent.mkdir(exist_ok=True)

with open(REPORT, "w") as f:
    json.dump(report, f, indent=2)


# Historical run
history_dir = Path("reports/phase10/history")
history_dir.mkdir(parents=True, exist_ok=True)

historical_report = (
    history_dir
    / f"phase10_ensemble_evaluation_{timestamp_file}.json"
)

with open(historical_report, "w") as f:
    json.dump(report, f, indent=2)


# Append-only history index for plotting
history_file = Path("reports/phase10/phase10_ensemble_history.json")

if history_file.exists():
    with open(history_file, "r") as f:
        history = json.load(f)
else:
    history = []

history.append(report)

with open(history_file, "w") as f:
    json.dump(history, f, indent=2)


print("\nLATEST REPORT:", REPORT)
print("HISTORICAL REPORT:", historical_report)
print("HISTORY INDEX:", history_file)
print("TIMESTAMP UTC:", timestamp_iso)
print("PHASE 10 ENSEMBLE EVALUATION: PASS")
