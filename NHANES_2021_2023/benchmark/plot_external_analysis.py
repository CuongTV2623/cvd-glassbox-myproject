
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "external_results"
COMPARISON_DIR = RESULTS_DIR / "comparison"
SHIFT_DIR = RESULTS_DIR / "distribution_shift"

COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "LOGREG",
    "LOGREG_SPLINE",
    "RF",
    "MLP",
    "EBM",
    "XGBOOST",
    "LIGHTGBM",
    "CATBOOST"
]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9
})


# ==================================================
# 2. AUPRC COMPARISON
# ==================================================

def plot_auprc_comparison():

    file = COMPARISON_DIR / "NHANES_internal_external_comparison.csv"
    df = pd.read_csv(file)

    for tier in ["T0", "T1"]:

        data = (
            df[df["Tier"] == tier]
            .set_index("Model")
            .reindex(MODELS)
            .reset_index()
        )

        x = np.arange(len(data))
        width = 0.36

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(
            x - width / 2,
            data["AUPRC_Internal"],
            width,
            label="Internal (2021-2023)"
        )

        ax.bar(
            x + width / 2,
            data["AUPRC_External"],
            width,
            label="External (2017-2018)"
        )

        ax.set_title(f"AUPRC Comparison - {tier}")
        ax.set_xlabel("Model")
        ax.set_ylabel("AUPRC")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=25, ha="right")
        ax.set_ylim(0, 0.55)

        ax.grid(axis="y", alpha=0.25)
        ax.legend()

        fig.tight_layout()

        output = COMPARISON_DIR / f"AUPRC_comparison_{tier}.png"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        print(f"Created: {output.name}")


# ==================================================
# 3. CATEGORICAL DISTRIBUTION
# ==================================================


def plot_categorical_distribution():

    file = SHIFT_DIR / "categorical_distribution.csv"
    df = pd.read_csv(file)

    # Ý nghĩa các mã trả lời theo NHANES
    category_labels = {
        "RIAGENDR": {
            "1": "Male",
            "2": "Female"
        },
        "RIDRETH3": {
            "1": "Mexican American",
            "2": "Other Hispanic",
            "3": "Non-Hispanic White",
            "4": "Non-Hispanic Black",
            "6": "Non-Hispanic Asian",
            "7": "Other / Multiracial"
        },
        "DMDEDUC2": {
            "1": "Below 9th grade",
            "2": "9-11th grade",
            "3": "High school / GED",
            "4": "Some college / AA",
            "5": "College graduate",
            "7": "Refused",
            "9": "Don't know"
        },
        "SMQ020": {
            "1": "Yes",
            "2": "No"
        },
        "BPQ020": {
            "1": "Yes",
            "2": "No",
            "7": "Refused",
            "9": "Don't know"
        },
        "BPQ080": {
            "1": "Yes",
            "2": "No",
            "7": "Refused",
            "9": "Don't know"
        },
        "DIQ010": {
            "1": "Yes",
            "2": "No",
            "3": "Borderline",
            "7": "Refused",
            "9": "Don't know"
        }
    }

    features = df["Feature"].drop_duplicates().tolist()

    fig, axes = plt.subplots(
        4, 2,
        figsize=(16, 17)
    )

    axes = axes.flatten()

    for ax, feature in zip(axes, features):

        data = df[df["Feature"] == feature].copy()

        # Chuyển mã thành chuỗi; giữ riêng giá trị thiếu
        data["Category_key"] = data["Category"].apply(
            lambda x: "Missing" if pd.isna(x) else str(int(x))
        )

        # Lấy tất cả nhóm xuất hiện ở cả hai chu kỳ
        categories = sorted(
            data["Category_key"].unique(),
            key=lambda x: (x == "Missing", int(x) if x != "Missing" else 999)
        )

        internal = (
            data[data["Dataset"] == "Internal_2021_2023"]
            .set_index("Category_key")["Proportion"]
            .to_dict()
        )

        external = (
            data[data["Dataset"] == "External_2017_2018"]
            .set_index("Category_key")["Proportion"]
            .to_dict()
        )

        x = np.arange(len(categories))
        width = 0.36

        internal_values = [
            internal.get(c, 0) * 100 for c in categories
        ]

        external_values = [
            external.get(c, 0) * 100 for c in categories
        ]

        ax.bar(
            x - width / 2,
            internal_values,
            width,
            label="Internal (2021-2023)"
        )

        ax.bar(
            x + width / 2,
            external_values,
            width,
            label="External (2017-2018)"
        )

        # Hiển thị tên nhóm thay cho mã số
        labels = [
            "Missing" if c == "Missing"
            else category_labels.get(feature, {}).get(c, c)
            for c in categories
        ]

        ax.set_title(feature, fontweight="bold")
        ax.set_ylabel("Proportion (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=25,
            ha="right",
            fontsize=8
        )

        ax.grid(axis="y", alpha=0.25)

    # Tắt ô trống cuối cùng
    axes[-1].axis("off")

    # Chú thích chung
    handles, _ = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        ["Internal (2021-2023)", "External (2017-2018)"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.975)
    )

    fig.suptitle(
        "Categorical Feature Distribution Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output = SHIFT_DIR / "categorical_distribution_comparison.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Created: {output}")


# ==================================================
# 4. MAIN
# ==================================================

if __name__ == "__main__":

    print("GENERATING EXTERNAL ANALYSIS FIGURES")

    plot_auprc_comparison()
    plot_categorical_distribution()

    print("\nALL FIGURES GENERATED SUCCESSFULLY!")
