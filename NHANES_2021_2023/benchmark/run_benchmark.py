
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    f1_score
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.model_registry import get_models


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = ROOT
RESULTS_DIR = ROOT / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ["T0", "T1"]
SPLITS = ["train", "validation", "test"]

TARGET = "CVD"
IDENTIFIERS = ["SEQN", TARGET]

MIN_SENSITIVITY = 0.85
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_split(tier, split):
    file_path = DATA_DIR / f"NHANES_{tier}_{split}_processed.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    return pd.read_csv(file_path)


def prepare_xy(df):
    X = df.drop(columns=IDENTIFIERS)
    y = df[TARGET].astype(int)

    return X, y


# ============================================================
# SELECT THRESHOLD ON VALIDATION
# ============================================================

def select_threshold(y_true, probabilities, min_sensitivity=0.85):
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)

    candidates = np.where(tpr >= min_sensitivity)[0]

    if len(candidates) == 0:
        # Fallback: choose the threshold with highest sensitivity.
        best_idx = np.argmax(tpr)
    else:
        # Among thresholds satisfying sensitivity requirement,
        # choose the one with the highest specificity.
        best_idx = candidates[np.argmin(fpr[candidates])]

    return float(thresholds[best_idx])


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    return {
        "AUROC": roc_auc_score(y_true, probabilities),
        "AUPRC": average_precision_score(y_true, probabilities),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "Threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():
    all_results = []
    all_predictions = []

    for tier in TIERS:
        print("\n" + "=" * 60)
        print(f"BENCHMARK: {tier}")
        print("=" * 60)

        train_df = load_split(tier, "train")
        val_df = load_split(tier, "validation")
        test_df = load_split(tier, "test")

        X_train, y_train = prepare_xy(train_df)
        X_val, y_val = prepare_xy(val_df)
        X_test, y_test = prepare_xy(test_df)

        # Ensure identical feature columns across splits.
        if not (
            list(X_train.columns)
            == list(X_val.columns)
            == list(X_test.columns)
        ):
            raise ValueError(f"Feature mismatch in {tier}")

        models = get_models(
            random_state=RANDOM_STATE,
            feature_names=X_train.columns.tolist()
        )

        print(f"Train: {len(X_train)}")
        print(f"Validation: {len(X_val)}")
        print(f"Test: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Models: {len(models)}")

        for model_name, model in models.items():
            print(f"\n--- Training {model_name} ({tier}) ---")

            start_time = time.time()

            try:
                model.fit(X_train, y_train)

                # Select threshold using validation data only.
                val_probabilities = model.predict_proba(X_val)[:, 1]

                threshold = select_threshold(
                    y_val,
                    val_probabilities,
                    MIN_SENSITIVITY
                )

                # Final evaluation on the untouched test split.
                test_probabilities = model.predict_proba(X_test)[:, 1]

                metrics = evaluate_model(
                    y_test,
                    test_probabilities,
                    threshold
                )

                elapsed = time.time() - start_time

                result = {
                    "Tier": tier,
                    "Model": model_name,
                    "Train_Size": len(X_train),
                    "Validation_Size": len(X_val),
                    "Test_Size": len(X_test),
                    "N_Features": X_train.shape[1],
                    "Train_Time_Seconds": round(elapsed, 2),
                    **metrics
                }

                all_results.append(result)

                for i, probability in enumerate(test_probabilities):
                    all_predictions.append({
                        "Tier": tier,
                        "Model": model_name,
                        "SEQN": test_df.iloc[i]["SEQN"],
                        "CVD_True": int(y_test.iloc[i]),
                        "Probability": float(probability),
                        "Prediction": int(
                            probability >= threshold
                        ),
                        "Threshold": threshold
                    })

                print(
                    f"AUROC: {metrics['AUROC']:.4f} | "
                    f"AUPRC: {metrics['AUPRC']:.4f} | "
                    f"Sensitivity: {metrics['Sensitivity']:.4f} | "
                    f"Specificity: {metrics['Specificity']:.4f}"
                )

            except Exception as error:
                print(f"[ERROR] {model_name} - {tier}: {error}")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)

    results_path = RESULTS_DIR / "NHANES_benchmark_results.csv"
    predictions_path = RESULTS_DIR / "NHANES_test_predictions.csv"

    results_df.to_csv(results_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)

    if not results_df.empty:
        print(
            results_df[
                ["Tier", "Model", "AUROC", "AUPRC",
                 "Sensitivity", "Specificity"]
            ].to_string(index=False)
        )

    print(f"\nResults saved: {results_path}")
    print(f"Predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()
