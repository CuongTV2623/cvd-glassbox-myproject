
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent

target = pd.read_csv(folder / "NHANES_CVD_target.csv")

demo = pd.read_sas(
    folder / "DEMO_L.xpt",
    format="xport"
)

mcq = pd.read_sas(
    folder / "MCQ_L.xpt",
    format="xport"
)

df = target.merge(
    demo[["SEQN", "RIDAGEYR"]],
    on="SEQN"
)

# Chi lay nguoi tu 20 tuoi tro len chua co nhan
missing = df[
    (df["RIDAGEYR"] >= 20) &
    (df["CVD"].isna())
]

missing = missing.merge(
    mcq[["SEQN", "MCQ160B", "MCQ160C",
         "MCQ160D", "MCQ160E", "MCQ160F"]],
    on="SEQN"
)

variables = [
    "MCQ160B", "MCQ160C", "MCQ160D",
    "MCQ160E", "MCQ160F"
]

print("===== 45 TRUONG HOP CHUA CO NHAN =====")
print("So nguoi:", len(missing))

for col in variables:
    print(f"\n{col}:")
    print(missing[col].value_counts(dropna=False))

print("\n===== CAC DONG DU LIEU =====")
print(missing[["SEQN"] + variables].to_string(index=False))

missing.to_csv(
    folder / "unlabeled_adults.csv",
    index=False
)

print("\nDa luu unlabeled_adults.csv")
