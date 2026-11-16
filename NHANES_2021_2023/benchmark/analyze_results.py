
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark" / "results"

INPUT_FILE = RESULTS_DIR / "NHANES_test_predictions.csv"
OUTPUT_FILE = RESULTS_DIR / "NHANES_paired_bootstrap.csv"

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)


def calculate_metrics(y, probabilities):
    return (
        roc_auc_score(y, probabilities),
        average_precision_score(y, probabilities)
    )


def analyze_model(data, model_name):
    t0 = data[data["Tier"] == "T0"].copy()
    t1 = data[data["Tier"] == "T1"].copy()

    # Match the same participants between T0 and T1.
    paired = t0[
        ["SEQN", "CVD_True", "Probability"]
    ].merge(
        t1[["SEQN", "CVD_True", "Probability"]],
        on="SEQN",
        suffixes=("_T0", "_T1"),
        validate="one_to_one"
    )

    if not (paired["CVD_True_T0"] == paired["CVD_True_T1"]).all():
        raise ValueError(f"Labels differ between tiers: {model_name}")

    y = paired["CVD_True_T0"].to_numpy(dtype=int)
    p0 = paired["Probability_T0"].to_numpy(dtype=float)
    p1 = paired["Probability_T1"].to_numpy(dtype=float)

    auc0, ap0 = calculate_metrics(y, p0)
    auc1, ap1 = calculate_metrics(y, p1)

    delta_auc = auc1 - auc0
    delta_ap = ap1 - ap0

    bootstrap_auc = []
    bootstrap_ap = []

    n = len(y)

    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)

        y_sample = y[indices]
        p0_sample = p0[indices]
        p1_sample = p1[indices]

        # Skip samples containing only one class.
        if len(np.unique(y_sample)) < 2:
            continue

        auc0_sample, ap0_sample = calculate_metrics(
            y_sample, p0_sample
        )
        auc1_sample, ap1_sample = calculate_metrics(
            y_sample, p1_sample
        )

        bootstrap_auc.append(auc1_sample - auc0_sample)
        bootstrap_ap.append(ap1_sample - ap0_sample)

    auc_ci = np.percentile(bootstrap_auc, [2.5, 97.5])
    ap_ci = np.percentile(bootstrap_ap, [2.5, 97.5])

    return {
        "Model": model_name,
        "N_Paired": n,
        "AUROC_T0": auc0,
        "AUROC_T1": auc1,
        "Delta_AUROC": delta_auc,
        "AUROC_CI_Lower": auc_ci[0],
        "AUROC_CI_Upper": auc_ci[1],
        "AUPRC_T0": ap0,
        "AUPRC_T1": ap1,
        "Delta_AUPRC": delta_ap,
        "AUPRC_CI_Lower": ap_ci[0],
        "AUPRC_CI_Upper": ap_ci[1],
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    predictions = pd.read_csv(INPUT_FILE)

    required = {
        "Tier", "Model", "SEQN",
        "CVD_True", "Probability"
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []

    for model_name in sorted(predictions["Model"].unique()):
        print(f"Analyzing {model_name}...")

        model_data = predictions[
            predictions["Model"] == model_name
        ]

        result = analyze_model(model_data, model_name)
        results.append(result)

    result_df = pd.DataFrame(results)

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== PAIRED BOOTSTRAP RESULTS =====")

    columns = [
        "Model",
        "Delta_AUROC",
        "AUROC_CI_Lower",
        "AUROC_CI_Upper",
        "Delta_AUPRC",
        "AUPRC_CI_Lower",
        "AUPRC_CI_Upper"
    ]

    print(result_df[columns].to_string(index=False))

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
