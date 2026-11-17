
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "benchmark" / "external_results"
    / "NHANES_external_predictions.csv"
)

OUTPUT_FILE = (
    ROOT / "benchmark" / "external_results"
    / "paired_bootstrap_T0_T1.csv"
)

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)

df = pd.read_csv(INPUT_FILE)

results = []

for model in sorted(df["Model"].unique()):

    t0 = (
        df[(df["Tier"] == "T0") & (df["Model"] == model)]
        .sort_values("SEQN")
        .reset_index(drop=True)
    )

    t1 = (
        df[(df["Tier"] == "T1") & (df["Model"] == model)]
        .sort_values("SEQN")
        .reset_index(drop=True)
    )

    # Dam bao hai tang co cung doi tuong va cung nhan
    if not np.array_equal(t0["SEQN"].to_numpy(), t1["SEQN"].to_numpy()):
        raise ValueError(f"SEQN mismatch: {model}")

    if not np.array_equal(
        t0["CVD_True"].to_numpy(),
        t1["CVD_True"].to_numpy()
    ):
        raise ValueError(f"CVD label mismatch: {model}")

    y = t0["CVD_True"].to_numpy().astype(int)
    p0 = t0["Probability"].to_numpy()
    p1 = t1["Probability"].to_numpy()

    delta_auroc = []
    delta_auprc = []

    n = len(y)

    for _ in range(N_BOOTSTRAP):

        # Dung chung chi so lay mau cho T0 va T1
        indices = rng.integers(0, n, size=n)

        y_boot = y[indices]

        if len(np.unique(y_boot)) < 2:
            continue

        p0_boot = p0[indices]
        p1_boot = p1[indices]

        auc0 = roc_auc_score(y_boot, p0_boot)
        auc1 = roc_auc_score(y_boot, p1_boot)

        ap0 = average_precision_score(y_boot, p0_boot)
        ap1 = average_precision_score(y_boot, p1_boot)

        delta_auroc.append(auc1 - auc0)
        delta_auprc.append(ap1 - ap0)

    delta_auroc = np.array(delta_auroc)
    delta_auprc = np.array(delta_auprc)

    # Chenh lech tren toan bo tap external
    observed_auc = (
        roc_auc_score(y, p1) - roc_auc_score(y, p0)
    )

    observed_ap = (
        average_precision_score(y, p1)
        - average_precision_score(y, p0)
    )

    auc_ci = np.percentile(delta_auroc, [2.5, 97.5])
    ap_ci = np.percentile(delta_auprc, [2.5, 97.5])

    results.append({
        "Model": model,
        "N": n,
        "AUROC_Difference_T1_T0": observed_auc,
        "AUROC_CI_Lower": auc_ci[0],
        "AUROC_CI_Upper": auc_ci[1],
        "AUROC_CI_Excludes_Zero": (
            auc_ci[0] > 0 or auc_ci[1] < 0
        ),
        "AUPRC_Difference_T1_T0": observed_ap,
        "AUPRC_CI_Lower": ap_ci[0],
        "AUPRC_CI_Upper": ap_ci[1],
        "AUPRC_CI_Excludes_Zero": (
            ap_ci[0] > 0 or ap_ci[1] < 0
        ),
        "Valid_Bootstrap": len(delta_auroc)
    })

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_FILE, index=False)

print("===== PAIRED BOOTSTRAP: T1 - T0 =====")
print(result_df.to_string(index=False))

print("\nSaved:", OUTPUT_FILE)
