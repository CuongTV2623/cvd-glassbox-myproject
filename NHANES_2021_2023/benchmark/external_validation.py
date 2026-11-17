
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

# ============================================================
# PATHS AND IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))

from benchmark.model_registry import get_models

EXTERNAL_DIR = (
    PROJECT_ROOT / "NHANES_2017_2018" / "processed"
)

RESULTS_DIR = ROOT / "benchmark" / "external_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIERS = ["T0", "T1"]

TARGET = "CVD"
IDENTIFIERS = ["SEQN", TARGET]

MIN_SENSITIVITY = 0.85
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data(tier, split):
    file_path = ROOT / f"NHANES_{tier}_{split}_processed.csv"

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    return pd.read_csv(file_path)


def load_external_data(tier):
    file_path = (
        EXTERNAL_DIR /
        f"NHANES_{tier}_external_processed.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    return pd.read_csv(file_path)


def prepare_xy(df):
    X = df.drop(columns=IDENTIFIERS)
    y = df[TARGET].astype(int)

    return X, y


# ============================================================
# SELECT THRESHOLD ON 2021-2023 VALIDATION
# ============================================================

def select_threshold(
    y_true,
    probabilities,
    min_sensitivity=0.85
):
    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities
    )

    candidates = np.where(
        tpr >= min_sensitivity
    )[0]

    if len(candidates) == 0:
        best_idx = np.argmax(tpr)
    else:
        best_idx = candidates[
            np.argmin(fpr[candidates])
        ]

    return float(thresholds[best_idx])


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn) if (tp + fn) else 0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) else 0
    )

    return {
        "AUROC": roc_auc_score(
            y_true, probabilities
        ),
        "AUPRC": average_precision_score(
            y_true, probabilities
        ),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# ============================================================
# EXTERNAL VALIDATION
# ============================================================

def main():

    all_results = []
    all_predictions = []

    for tier in TIERS:

        print("\n" + "=" * 65)
        print(f"EXTERNAL VALIDATION: {tier}")
        print("=" * 65)

        # Load 2021-2023 training and validation data
        train_df = load_training_data(
            tier, "train"
        )

        val_df = load_training_data(
            tier, "validation"
        )

        # Load independent 2017-2018 external data
        external_df = load_external_data(tier)

        X_train, y_train = prepare_xy(train_df)
        X_val, y_val = prepare_xy(val_df)
        X_external, y_external = prepare_xy(
            external_df
        )

        # Check feature consistency
        if not (
            list(X_train.columns)
            == list(X_val.columns)
            == list(X_external.columns)
        ):
            raise ValueError(
                f"Feature mismatch in {tier}"
            )

        print("Training rows:", len(X_train))
        print("Validation rows:", len(X_val))
        print("External rows:", len(X_external))
        print("Number of features:", X_train.shape[1])

        print("\nExternal CVD distribution:")
        print(y_external.value_counts().sort_index())

        # Initialize models
        models = get_models(
            random_state=RANDOM_STATE,
            feature_names=X_train.columns.tolist()
        )

        print("Number of models:", len(models))

        for model_name, model in models.items():

            print(f"\n--- {model_name} ({tier}) ---")

            start_time = time.time()

            try:
                # 1. Train only on NHANES 2021-2023
                model.fit(X_train, y_train)

                # 2. Select threshold on 2021-2023 validation
                val_probabilities = (
                    model.predict_proba(X_val)[:, 1]
                )

                threshold = select_threshold(
                    y_val,
                    val_probabilities,
                    MIN_SENSITIVITY
                )

                # 3. Evaluate on NHANES 2017-2018
                external_probabilities = (
                    model.predict_proba(X_external)[:, 1]
                )

                metrics = evaluate_model(
                    y_external,
                    external_probabilities,
                    threshold
                )

                elapsed = time.time() - start_time

                result = {
                    "Tier": tier,
                    "Model": model_name,
                    "Train_Size": len(X_train),
                    "Validation_Size": len(X_val),
                    "External_Size": len(X_external),
                    "N_Features": X_train.shape[1],
                    "Train_Time_Seconds": round(
                        elapsed, 2
                    ),
                    **metrics
                }

                all_results.append(result)

                for i, probability in enumerate(
                    external_probabilities
                ):
                    all_predictions.append({
                        "Tier": tier,
                        "Model": model_name,
                        "SEQN": external_df.iloc[i]["SEQN"],
                        "CVD_True": int(
                            y_external.iloc[i]
                        ),
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
                print(
                    f"[ERROR] {model_name} - {tier}: {error}"
                )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)

    results_path = (
        RESULTS_DIR / "NHANES_external_results.csv"
    )

    predictions_path = (
        RESULTS_DIR / "NHANES_external_predictions.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print("\n" + "=" * 65)
    print("EXTERNAL VALIDATION COMPLETED")
    print("=" * 65)

    if not results_df.empty:
        print(
            results_df[
                [
                    "Tier",
                    "Model",
                    "AUROC",
                    "AUPRC",
                    "Sensitivity",
                    "Specificity"
                ]
            ].to_string(index=False)
        )

    print("\nResults saved:", results_path)
    print("Predictions saved:", predictions_path)


if __name__ == "__main__":
    main()
