
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ==========================================
# 1. PATH
# ==========================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "multi_seed"
OUTPUT_DIR = DATA_DIR / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "multiseed_results.csv"


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("===== DATA CHECK =====")
print("Total results:", len(df))
print("Number of models:", df["Model"].nunique())
print("Number of tiers:", df["Tier"].nunique())
print("Number of seeds:", df["Seed"].nunique())

assert set(df["Tier"]) == {"T0", "T1"}, \
    "Missing T0 or T1 results"

assert df.groupby(["Tier", "Model", "Seed"]).size().max() == 1, \
    "Duplicate experiment results"


# ==========================================
# 3. GENERALIZATION STABILITY
# ==========================================

metrics = ["AUROC", "AUPRC", "Sensitivity", "Specificity"]

summary_rows = []

for (tier, model), group in df.groupby(["Tier", "Model"]):

    for metric in metrics:

        values = group[metric].dropna().to_numpy()

        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        if n > 1:
            se = stats.sem(values)
            critical = stats.t.ppf(0.975, n - 1)
            ci_lower = mean - critical * se
            ci_upper = mean + critical * se
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        summary_rows.append({
            "Tier": tier,
            "Model": model,
            "Metric": metric,
            "Mean": mean,
            "SD": std,
            "Min": np.min(values),
            "Max": np.max(values),
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper,
            "N": n
        })


stability = pd.DataFrame(summary_rows)

stability.to_csv(
    OUTPUT_DIR / "generalization_stability.csv",
    index=False
)


# ==========================================
# 4. COMPARE T0 AND T1
# ==========================================

comparison_rows = []

for metric in ["AUROC", "AUPRC"]:

    pivot = df.pivot(
        index=["Seed", "Model"],
        columns="Tier",
        values=metric
    )

    pivot["Difference"] = pivot["T1"] - pivot["T0"]

    for model, group in pivot.groupby("Model"):

        values = group["Difference"].dropna().to_numpy()

        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        se = stats.sem(values)
        critical = stats.t.ppf(0.975, n - 1)

        comparison_rows.append({
            "Metric": metric,
            "Model": model,
            "Mean_Difference": mean,
            "SD": std,
            "CI_Lower": mean - critical * se,
            "CI_Upper": mean + critical * se,
            "Positive_Seeds": int(np.sum(values > 0)),
            "Negative_Seeds": int(np.sum(values < 0))
        })


comparison = pd.DataFrame(comparison_rows)

comparison.to_csv(
    OUTPUT_DIR / "generalization_T0_T1_comparison.csv",
    index=False
)


# ==========================================
# 5. VISUALIZATION
# ==========================================

for metric in ["AUROC", "AUPRC"]:

    plot_df = stability[
        stability["Metric"] == metric
    ]

    models = sorted(df["Model"].unique())

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(12, 6))

    for i, tier in enumerate(["T0", "T1"]):

        subset = (
            plot_df[plot_df["Tier"] == tier]
            .set_index("Model")
            .reindex(models)
        )

        plt.bar(
            x + (i - 0.5) * width,
            subset["Mean"],
            width,
            yerr=subset["SD"],
            capsize=4,
            label=tier
        )

    plt.xticks(x, models, rotation=30)
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(f"{metric}: Generalization Performance")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"generalization_{metric.lower()}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 6. PRINT RESULTS
# ==========================================

print("\n===== GENERALIZATION STABILITY =====")

print(
    stability[
        stability["Metric"] == "AUROC"
    ].round(4).to_string(index=False)
)

print("\n===== T0 VS T1 COMPARISON =====")

print(
    comparison.round(4).to_string(index=False)
)

print("\n===== COMPLETED =====")
print("Output:", OUTPUT_DIR)
