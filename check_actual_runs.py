
from pathlib import Path
import pandas as pd

path = Path("results/raw_results.csv")

if not path.exists():
    print("Khong tim thay results/raw_results.csv")
    raise SystemExit

df = pd.read_csv(path)

print("===== CAC COT =====")
print(df.columns.tolist())

print("\n===== CAC RUN BENCHMARK THUC TE =====")

runs = df[df["stage"] == "benchmark"]

if runs.empty:
    print("Chua co benchmark nao duoc ghi nhan.")
else:
    summary = runs[
        ["dataset", "tier", "model", "seed"]
    ].drop_duplicates()

    print(summary.sort_values(
        ["dataset", "tier", "model", "seed"]
    ).to_string(index=False))

    print("\n===== TONG HOP =====")
    print("Models:", sorted(runs["model"].unique()))
    print("Seeds:", sorted(runs["seed"].unique()))
    print("Tiers:", sorted(runs["tier"].unique()))
