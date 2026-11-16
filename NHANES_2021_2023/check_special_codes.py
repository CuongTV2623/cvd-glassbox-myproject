
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent
df = pd.read_csv(folder / "NHANES_adult_merged.csv")

check_vars = [
    "INDFMPIR",
    "ALQ121",
    "ALQ142",
    "ALQ170",
    "PAD790Q",
    "PAD810Q"
]

for col in check_vars:
    print(f"\n===== {col} =====")
    print(df[col].value_counts(dropna=False).head(20).to_string())
    print("Gia tri nho nhat:", df[col].min())
