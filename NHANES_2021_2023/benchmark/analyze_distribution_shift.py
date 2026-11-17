
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INTERNAL_FILE = ROOT / "NHANES_T1.csv"

EXTERNAL_FILE = (
    ROOT.parent
    / "NHANES_2017_2018"
    / "processed"
    / "NHANES_adult_merged.csv"
)

OUTPUT_DIR = (
    ROOT / "benchmark"
    / "external_results"
    / "distribution_shift"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# FEATURES
# ==========================================

NUMERIC_FEATURES = [
    "RIDAGEYR",
    "INDFMPIR",
    "BMXBMI",
    "BMXWAIST",
    "PAD680"
]

CATEGORICAL_FEATURES = [
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "SMQ020",
    "BPQ020",
    "BPQ080",
    "DIQ010"
]

# ==========================================
# LOAD DATA
# ==========================================

internal = pd.read_csv(INTERNAL_FILE)
external = pd.read_csv(EXTERNAL_FILE)


# Xu ly ma dac biet cua PAD680
for df in [internal, external]:
    df["PAD680"] = pd.to_numeric(df["PAD680"], errors="coerce")
    df["PAD680"] = df["PAD680"].replace([7777, 9999], float("nan"))

datasets = {
    "Internal_2021_2023": internal,
    "External_2017_2018": external
}

print("===== DATA INFORMATION =====")

for name, df in datasets.items():
    print(f"\n{name}")
    print("Rows:", len(df))

# ==========================================
# NUMERIC DISTRIBUTION
# ==========================================

numeric_results = []

for feature in NUMERIC_FEATURES:

    for dataset_name, df in datasets.items():

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).dropna()

        numeric_results.append({
            "Feature": feature,
            "Dataset": dataset_name,
            "Count": len(values),
            "Missing_Rate": df[feature].isna().mean(),
            "Mean": values.mean(),
            "Std": values.std(),
            "Median": values.median(),
            "Q1": values.quantile(0.25),
            "Q3": values.quantile(0.75),
            "Min": values.min(),
            "Max": values.max()
        })

numeric_df = pd.DataFrame(numeric_results)

numeric_df.to_csv(
    OUTPUT_DIR / "numeric_distribution.csv",
    index=False
)

# ==========================================
# NUMERIC DIFFERENCE
# ==========================================

numeric_comparison = []

for feature in NUMERIC_FEATURES:

    internal_values = pd.to_numeric(
        internal[feature],
        errors="coerce"
    ).dropna()

    external_values = pd.to_numeric(
        external[feature],
        errors="coerce"
    ).dropna()

    internal_mean = internal_values.mean()
    external_mean = external_values.mean()

    internal_std = internal_values.std()
    external_std = external_values.std()

    pooled_std = np.sqrt(
        (internal_std ** 2 + external_std ** 2) / 2
    )

    standardized_difference = (
        (external_mean - internal_mean) / pooled_std
        if pooled_std > 0 else np.nan
    )

    numeric_comparison.append({
        "Feature": feature,
        "Internal_Mean": internal_mean,
        "External_Mean": external_mean,
        "Mean_Difference": external_mean - internal_mean,
        "Internal_Std": internal_std,
        "External_Std": external_std,
        "Standardized_Mean_Difference": standardized_difference
    })

comparison_df = pd.DataFrame(numeric_comparison)

comparison_df.to_csv(
    OUTPUT_DIR / "numeric_shift_comparison.csv",
    index=False
)

# ==========================================
# CATEGORICAL DISTRIBUTION
# ==========================================

categorical_results = []

for feature in CATEGORICAL_FEATURES:

    for dataset_name, df in datasets.items():

        proportions = (
            df[feature]
            .value_counts(normalize=True, dropna=False)
        )

        for category, proportion in proportions.items():

            categorical_results.append({
                "Feature": feature,
                "Dataset": dataset_name,
                "Category": str(category),
                "Proportion": proportion
            })

categorical_df = pd.DataFrame(categorical_results)

categorical_df.to_csv(
    OUTPUT_DIR / "categorical_distribution.csv",
    index=False
)

# ==========================================
# NUMERIC DISTRIBUTION PLOTS
# ==========================================

for feature in NUMERIC_FEATURES:

    plt.figure(figsize=(9, 5))

    plt.hist(
        pd.to_numeric(
            internal[feature],
            errors="coerce"
        ).dropna(),
        bins=30,
        alpha=0.5,
        density=True,
        label="Internal 2021-2023"
    )

    plt.hist(
        pd.to_numeric(
            external[feature],
            errors="coerce"
        ).dropna(),
        bins=30,
        alpha=0.5,
        density=True,
        label="External 2017-2018"
    )

    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(f"Distribution Shift - {feature}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"distribution_{feature}.png",
        dpi=300
    )

    plt.close()

# ==========================================
# SUMMARY
# ==========================================

print("\n===== NUMERIC SHIFT SUMMARY =====")

print(
    comparison_df.round(4).to_string(index=False)
)

print("\n===== COMPLETED =====")
print("Output directory:", OUTPUT_DIR)
