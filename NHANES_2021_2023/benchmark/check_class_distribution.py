
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

internal_file = ROOT / "NHANES_T0_test_processed.csv"
external_file = (
    ROOT.parent
    / "NHANES_2017_2018"
    / "processed"
    / "NHANES_T0_external_processed.csv"
)

datasets = {
    "Internal Test 2021-2023": pd.read_csv(internal_file),
    "External 2017-2018": pd.read_csv(external_file)
}

for name, df in datasets.items():
    counts = df["CVD"].value_counts().sort_index()
    total = len(df)

    print(f"\n===== {name} =====")
    print("Total:", total)
    print("CVD = 0:", counts.get(0, 0))
    print("CVD = 1:", counts.get(1, 0))
    print(
        "CVD prevalence:",
        round(counts.get(1, 0) / total * 100, 2),
        "%"
    )
