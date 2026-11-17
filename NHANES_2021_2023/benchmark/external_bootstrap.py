
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# PATHS
# =========================
ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "benchmark" / "external_results"
    / "NHANES_external_predictions.csv"
)

OUTPUT_DIR = ROOT / "benchmark" / "external_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "NHANES_external_bootstrap_CI.csv"

# =========================
# CONFIGURATION
# =========================
N_BOOTSTRAP = 2000
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95

rng = np.random.default_rng(RANDOM_STATE)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

required_columns = [
    "Tier", "Model", "CVD_True", "Probability"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

print("===== EXTERNAL BOOTSTRAP =====")
print("Rows:", len(df))
print("Bootstrap repetitions:", N_BOOTSTRAP)

results = []

# =========================
# BOOTSTRAP BY MODEL
# =========================
for (tier, model), group in df.groupby(["Tier", "Model"]):

    y_true = group["CVD_True"].to_numpy()
    y_prob = group["Probability"].to_numpy()

    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]

    auroc_values = []
    auprc_values = []

    n = len(y_true)

    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)

        y_boot = y_true[indices]
        p_boot = y_prob[indices]

        # Bỏ mẫu bootstrap chỉ có một lớp
        if len(np.unique(y_boot)) < 2:
            continue

        auroc_values.append(
            roc_auc_score(y_boot, p_boot)
        )

        auprc_values.append(
            average_precision_score(y_boot, p_boot)
        )

    alpha = 1 - CONFIDENCE_LEVEL

    auroc_ci = np.percentile(
        auroc_values,
        [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    auprc_ci = np.percentile(
        auprc_values,
        [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    results.append({
        "Tier": tier,
        "Model": model,
        "N": n,
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUROC_CI_Lower": auroc_ci[0],
        "AUROC_CI_Upper": auroc_ci[1],
        "AUPRC": average_precision_score(y_true, y_prob),
        "AUPRC_CI_Lower": auprc_ci[0],
        "AUPRC_CI_Upper": auprc_ci[1],
        "Valid_Bootstrap": len(auroc_values)
    })

    print(
        f"{tier} | {model} | "
        f"AUROC={roc_auc_score(y_true, y_prob):.4f} "
        f"CI=({auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}) | "
        f"AUPRC={average_precision_score(y_true, y_prob):.4f} "
        f"CI=({auprc_ci[0]:.4f}, {auprc_ci[1]:.4f})"
    )

# =========================
# SAVE RESULTS
# =========================
result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_FILE, index=False)

print("\n===== COMPLETED =====")
print(result_df.to_string(index=False))
print("\nSaved:", OUTPUT_FILE)
