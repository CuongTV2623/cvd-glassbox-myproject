
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "multi_seed"
OUTPUT_DIR = DATA_DIR / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "multiseed_paired_differences.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "Seed",
    "Model",
    "Delta_AUROC",
    "Delta_AUPRC"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("===== DATA CHECK =====")
print("Number of rows:", len(df))
print("Number of seeds:", df["Seed"].nunique())
print("Number of models:", df["Model"].nunique())

if df.duplicated(["Seed", "Model"]).any():
    raise ValueError("Duplicate Seed-Model pairs detected.")

expected_rows = df["Seed"].nunique() * df["Model"].nunique()

if len(df) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} rows, but found {len(df)}."
    )


# ============================================================
# 3. STATISTICAL ANALYSIS
# ============================================================

results = []

for model, group in df.groupby("Model"):

    row = {"Model": model}

    for metric in ["Delta_AUROC", "Delta_AUPRC"]:

        values = group[metric].dropna().to_numpy()
        n = len(values)

        mean_value = np.mean(values)
        std_value = np.std(values, ddof=1)

        # 95% confidence interval using t distribution
        se = stats.sem(values)

        if n > 1:
            t_critical = stats.t.ppf(0.975, df=n - 1)
            ci_lower = mean_value - t_critical * se
            ci_upper = mean_value + t_critical * se
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        # One-sample paired t-test against zero
        if n > 1 and std_value > 0:
            t_stat, t_pvalue = stats.ttest_1samp(
                values, popmean=0
            )
        else:
            t_stat, t_pvalue = np.nan, np.nan

        # Wilcoxon signed-rank test against zero
        try:
            if np.allclose(values, 0):
                w_stat, w_pvalue = 0.0, 1.0
            else:
                w_stat, w_pvalue = stats.wilcoxon(
                    values,
                    alternative="two-sided"
                )
        except ValueError:
            w_stat, w_pvalue = np.nan, np.nan

        prefix = metric.replace("Delta_", "")

        row[f"{prefix}_Mean"] = mean_value
        row[f"{prefix}_SD"] = std_value
        row[f"{prefix}_CI_Lower"] = ci_lower
        row[f"{prefix}_CI_Upper"] = ci_upper
        row[f"{prefix}_TTest_P"] = t_pvalue
        row[f"{prefix}_Wilcoxon_P"] = w_pvalue

    results.append(row)


stats_df = pd.DataFrame(results)

stats_df.to_csv(
    OUTPUT_DIR / "multiseed_statistical_tests.csv",
    index=False
)


# ============================================================
# 4. PRINT RESULTS
# ============================================================

print("\n===== STATISTICAL RESULTS =====")

columns_to_show = [
    "Model",
    "AUROC_Mean",
    "AUROC_SD",
    "AUROC_CI_Lower",
    "AUROC_CI_Upper",
    "AUROC_TTest_P",
    "AUROC_Wilcoxon_P",
    "AUPRC_Mean",
    "AUPRC_SD",
    "AUPRC_CI_Lower",
    "AUPRC_CI_Upper",
    "AUPRC_TTest_P",
    "AUPRC_Wilcoxon_P"
]

print(
    stats_df[columns_to_show].round(4).to_string(index=False)
)


# ============================================================
# 5. PLOT MEAN IMPROVEMENT
# ============================================================

plot_df = stats_df.sort_values("AUROC_Mean")

for metric in ["AUROC", "AUPRC"]:

    means = plot_df[f"{metric}_Mean"].to_numpy()
    stds = plot_df[f"{metric}_SD"].to_numpy()
    models = plot_df["Model"].to_numpy()

    plt.figure(figsize=(10, 6))

    plt.barh(
        models,
        means,
        xerr=stds,
        capsize=4
    )

    plt.axvline(
        x=0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(f"Mean improvement in {metric} (T1 - T0)")
    plt.ylabel("Model")
    plt.title(
        f"Multi-seed {metric} Improvement "
        "(Mean and Standard Deviation)"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"multiseed_delta_{metric.lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 6. PLOT IMPROVEMENT ACROSS SEEDS
# ============================================================

for metric in ["Delta_AUROC", "Delta_AUPRC"]:

    pivot = df.pivot(
        index="Seed",
        columns="Model",
        values=metric
    )

    plt.figure(figsize=(11, 6))

    for model in pivot.columns:
        plt.plot(
            pivot.index,
            pivot[model],
            marker="o",
            label=model
        )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Random seed")
    plt.ylabel(metric.replace("Delta_", "Improvement in "))
    plt.title(f"{metric} Across Five Random Seeds")
    plt.xticks(sorted(df["Seed"].unique()))
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    filename = metric.lower() + "_by_seed.png"

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print("\n================================================")
print("MULTI-SEED STATISTICAL ANALYSIS COMPLETED")
print("================================================")
print("Output directory:", OUTPUT_DIR)
