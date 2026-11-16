
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent
df = pd.read_csv(folder / "NHANES_adult_clean.csv")

tier_vars = [
    "BPQ020",
    "BPQ080",
    "DIQ010"
]

for col in tier_vars:
    print(f"\n===== {col} =====")
    print(df[col].value_counts(dropna=False).sort_index())

print("\n===== PHAN BO CVD =====")
print(df["CVD"].value_counts(dropna=False))
