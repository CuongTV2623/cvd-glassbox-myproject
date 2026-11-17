
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark" / "results"
EXTERNAL_DIR = ROOT / "benchmark" / "external_results"

internal_file = RESULTS_DIR / "NHANES_benchmark_results.csv"
external_file = EXTERNAL_DIR / "NHANES_external_results.csv"

OUTPUT_DIR = EXTERNAL_DIR / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD RESULTS
# ==========================================

internal = pd.read_csv(internal_file)
external = pd.read_csv(external_file)

print("Internal results:", internal.shape)
print("External results:", external.shape)

# ==========================================
# SELECT METRICS
# ==========================================

metrics = [
    "AUROC",
    "AUPRC",
    "Sensitivity",
    "Specificity"
]

internal = internal[
    ["Tier", "Model"] + metrics
].copy()

external = external[
    ["Tier", "Model"] + metrics
].copy()

# Rename columns
internal = internal.rename(
    columns={
        metric: f"{metric}_Internal"
        for metric in metrics
    }
)

external = external.rename(
    columns={
        metric: f"{metric}_External"
        for metric in metrics
    }
)

# ==========================================
# MERGE
# ==========================================

comparison = pd.merge(
    internal,
    external,
    on=["Tier", "Model"],
    how="inner",
    validate="one_to_one"
)

if len(comparison) != 16:
    raise ValueError(
        f"Expected 16 matched results, got {len(comparison)}"
    )

# ==========================================
# CALCULATE DIFFERENCES
# ==========================================

for metric in metrics:
    comparison[f"{metric}_Difference"] = (
        comparison[f"{metric}_External"]
        - comparison[f"{metric}_Internal"]
    )

# ==========================================
# SAVE CSV
# ==========================================

comparison.to_csv(
    OUTPUT_DIR / "NHANES_internal_external_comparison.csv",
    index=False
)

print("\n===== INTERNAL VS EXTERNAL COMPARISON =====")

print(
    comparison[
        [
            "Tier",
            "Model",
            "AUROC_Internal",
            "AUROC_External",
            "AUROC_Difference",
            "AUPRC_Internal",
            "AUPRC_External",
            "AUPRC_Difference"
        ]
    ].round(4).to_string(index=False)
)

# ==========================================
# SUMMARY BY TIER
# ==========================================

summary = comparison.groupby("Tier")[
    [
        "AUROC_Internal",
        "AUROC_External",
        "AUROC_Difference",
        "AUPRC_Internal",
        "AUPRC_External",
        "AUPRC_Difference"
    ]
].mean().reset_index()

summary.to_csv(
    OUTPUT_DIR / "NHANES_generalization_summary.csv",
    index=False
)

print("\n===== SUMMARY BY TIER =====")
print(summary.round(4).to_string(index=False))

# ==========================================
# PLOT 1: AUROC COMPARISON
# ==========================================

for tier in ["T0", "T1"]:

    subset = comparison[
        comparison["Tier"] == tier
    ].copy()

    x = range(len(subset))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - width / 2 for i in x],
        subset["AUROC_Internal"],
        width,
        label="Internal 2021-2023"
    )

    plt.bar(
        [i + width / 2 for i in x],
        subset["AUROC_External"],
        width,
        label="External 2017-2018"
    )

    plt.xticks(
        list(x),
        subset["Model"],
        rotation=30
    )

    plt.ylabel("AUROC")
    plt.xlabel("Model")
    plt.title(f"Internal vs External AUROC - {tier}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"AUROC_comparison_{tier}.png",
        dpi=300
    )

    plt.close()

# ==========================================
# PLOT 2: AUROC DIFFERENCE
# ==========================================

for tier in ["T0", "T1"]:

    subset = comparison[
        comparison["Tier"] == tier
    ].copy()

    plt.figure(figsize=(12, 6))

    plt.bar(
        subset["Model"],
        subset["AUROC_Difference"]
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.ylabel("External AUROC - Internal AUROC")
    plt.xlabel("Model")
    plt.title(f"AUROC Generalization Difference - {tier}")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"AUROC_difference_{tier}.png",
        dpi=300
    )

    plt.close()

print("\nComparison completed.")
print("Output directory:", OUTPUT_DIR)
