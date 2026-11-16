
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    f1_score
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import get_models

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "benchmark" / "multi_seed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
TIERS = ["T0", "T1"]
MIN_SENSITIVITY = 0.85

NUMERIC_FEATURES = [
    "RIDAGEYR",
    "INDFMPIR",
    "BMXBMI",
    "BMXWAIST",
    "PAD680"
]

CATEGORICAL_T0 = [
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "SMQ020"
]

CATEGORICAL_T1 = CATEGORICAL_T0 + [
    "BPQ020",
    "BPQ080",
    "DIQ010"
]


# ============================================================
# CREATE STRATIFIED SPLIT
# ============================================================

def create_split(df, seed):

    train_ids, temp_ids = train_test_split(
        df["SEQN"],
        test_size=0.4,
        random_state=seed,
        stratify=df["CVD"]
    )

    temp = df[df["SEQN"].isin(temp_ids)]

    val_ids, test_ids = train_test_split(
        temp["SEQN"],
        test_size=0.5,
        random_state=seed,
        stratify=temp["CVD"]
    )

    split = pd.DataFrame({
        "SEQN": df["SEQN"],
        "split": "train"
    })

    split.loc[
        split["SEQN"].isin(val_ids), "split"
    ] = "validation"

    split.loc[
        split["SEQN"].isin(test_ids), "split"
    ] = "test"

    return split


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_tier(df, split, tier):

    df = df.merge(
        split,
        on="SEQN",
        how="left",
        validate="one_to_one"
    )

    categorical_features = (
        CATEGORICAL_T0 if tier == "T0"
        else CATEGORICAL_T1
    )

    feature_columns = NUMERIC_FEATURES + categorical_features

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, categorical_features)
        ],
        sparse_threshold=0
    )

    X_train = train[feature_columns]
    X_val = val[feature_columns]
    X_test = test[feature_columns]

    # Fit only on training data.
    X_train_processed = preprocessor.fit_transform(X_train)

    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    def make_processed_dataframe(part, X_processed):

        result = pd.DataFrame(
            X_processed,
            columns=feature_names
        )

        result.insert(0, "CVD", part["CVD"].values)
        result.insert(0, "SEQN", part["SEQN"].values)

        return result

    train_processed = make_processed_dataframe(
        train, X_train_processed
    )

    val_processed = make_processed_dataframe(
        val, X_val_processed
    )

    test_processed = make_processed_dataframe(
        test, X_test_processed
    )

    return train_processed, val_processed, test_processed


# ============================================================
# SELECT THRESHOLD ON VALIDATION
# ============================================================

def select_threshold(y_true, probabilities):

    fpr, tpr, thresholds = roc_curve(
        y_true, probabilities
    )

    candidates = np.where(
        tpr >= MIN_SENSITIVITY
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

    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0

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
# MAIN
# ============================================================

def main():

    all_results = []
    all_predictions = []
    all_errors = []

    # Load the two tier datasets.
    datasets = {
        tier: pd.read_csv(ROOT / f"NHANES_{tier}.csv")
        for tier in TIERS
    }

    # Confirm T0 and T1 contain the same participants and labels.
    reference = datasets["T0"][["SEQN", "CVD"]].sort_values("SEQN")
    comparison = datasets["T1"][["SEQN", "CVD"]].sort_values("SEQN")

    if not reference.reset_index(drop=True).equals(
        comparison.reset_index(drop=True)
    ):
        raise ValueError("T0 and T1 do not have identical participants/labels.")

    # Run each seed.
    for seed in SEEDS:

        print("\n" + "#" * 65)
        print(f"RANDOM SEED: {seed}")
        print("#" * 65)

        # One common split for T0 and T1.
        split = create_split(datasets["T0"], seed)

        split.to_csv(
            OUTPUT_DIR / f"split_seed_{seed}.csv",
            index=False
        )

        for tier in TIERS:

            print("\n" + "=" * 60)
            print(f"BENCHMARK: {tier} | SEED {seed}")
            print("=" * 60)

            train_df, val_df, test_df = preprocess_tier(
                datasets[tier],
                split,
                tier
            )

            X_train = train_df.drop(columns=["SEQN", "CVD"])
            y_train = train_df["CVD"].astype(int)

            X_val = val_df.drop(columns=["SEQN", "CVD"])
            y_val = val_df["CVD"].astype(int)

            X_test = test_df.drop(columns=["SEQN", "CVD"])
            y_test = test_df["CVD"].astype(int)

            print("Train:", len(X_train))
            print("Validation:", len(X_val))
            print("Test:", len(X_test))
            print("Features:", X_train.shape[1])

            models = get_models(
                random_state=seed,
                feature_names=X_train.columns.tolist()
            )

            for model_name, model in models.items():

                print(f"\n--- Training {model_name} ({tier}, seed={seed}) ---")

                start_time = time.time()

                try:
                    model.fit(X_train, y_train)

                    val_probabilities = model.predict_proba(
                        X_val
                    )[:, 1]

                    threshold = select_threshold(
                        y_val,
                        val_probabilities
                    )

                    test_probabilities = model.predict_proba(
                        X_test
                    )[:, 1]

                    metrics = evaluate_model(
                        y_test,
                        test_probabilities,
                        threshold
                    )

                    elapsed = time.time() - start_time

                    result = {
                        "Seed": seed,
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
                            "Seed": seed,
                            "Tier": tier,
                            "Model": model_name,
                            "SEQN": test_df.iloc[i]["SEQN"],
                            "CVD_True": int(y_test.iloc[i]),
                            "Probability": float(probability),
                            "Threshold": threshold
                        })

                    print(
                        f"AUROC: {metrics['AUROC']:.4f} | "
                        f"AUPRC: {metrics['AUPRC']:.4f} | "
                        f"Sensitivity: {metrics['Sensitivity']:.4f} | "
                        f"Specificity: {metrics['Specificity']:.4f} | "
                        f"Time: {elapsed:.1f}s"
                    )

                except Exception as error:

                    print(f"[ERROR] {model_name}: {error}")

                    all_errors.append({
                        "Seed": seed,
                        "Tier": tier,
                        "Model": model_name,
                        "Error": str(error)
                    })

        # Save progress after each seed.
        pd.DataFrame(all_results).to_csv(
            OUTPUT_DIR / "multiseed_results_partial.csv",
            index=False
        )

        pd.DataFrame(all_predictions).to_csv(
            OUTPUT_DIR / "multiseed_predictions_partial.csv",
            index=False
        )

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)
    errors_df = pd.DataFrame(all_errors)

    results_df.to_csv(
        OUTPUT_DIR / "multiseed_results.csv",
        index=False
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "multiseed_predictions.csv",
        index=False
    )

    errors_df.to_csv(
        OUTPUT_DIR / "multiseed_errors.csv",
        index=False
    )

    if not results_df.empty:

        metrics = [
            "AUROC",
            "AUPRC",
            "Sensitivity",
            "Specificity"
        ]

        summary = results_df.groupby(
            ["Tier", "Model"]
        )[metrics].agg(["mean", "std"])

        summary.to_csv(
            OUTPUT_DIR / "multiseed_summary.csv"
        )

        # Paired differences: same seed and model, T1 minus T0.
        paired = results_df.pivot(
            index=["Seed", "Model"],
            columns="Tier",
            values=["AUROC", "AUPRC"]
        )

        differences = pd.DataFrame({
            "Delta_AUROC": (
                paired[("AUROC", "T1")]
                - paired[("AUROC", "T0")]
            ),
            "Delta_AUPRC": (
                paired[("AUPRC", "T1")]
                - paired[("AUPRC", "T0")]
            )
        }).reset_index()

        differences.to_csv(
            OUTPUT_DIR / "multiseed_paired_differences.csv",
            index=False
        )

        print("\n===== MULTI-SEED SUMMARY =====")
        print(summary.to_string())

        print("\n===== PAIRED DIFFERENCES (T1 - T0) =====")
        print(
            differences.groupby("Model")[
                ["Delta_AUROC", "Delta_AUPRC"]
            ].agg(["mean", "std"]).to_string()
        )

    print("\n" + "=" * 60)
    print("MULTI-SEED BENCHMARK COMPLETED")
    print("=" * 60)

    print("Results:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
