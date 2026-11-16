
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent

# Doc nhan CVD
target = pd.read_csv(folder / "NHANES_CVD_target.csv")

# Doc thong tin tuoi
demo = pd.read_sas(
    folder / "DEMO_L.xpt",
    format="xport"
)

# Ghep theo SEQN
df = target.merge(
    demo[["SEQN", "RIDAGEYR"]],
    on="SEQN",
    how="left",
    validate="one_to_one"
)

# Thong ke theo nhom tuoi
df["AGE_GROUP"] = pd.cut(
    df["RIDAGEYR"],
    bins=[0, 19, 39, 59, 79, 120],
    labels=["Under 20", "20-39", "40-59", "60-79", "80+"]
)

print("===== NHAN CVD THEO NHOM TUOI =====")
print(pd.crosstab(
    df["AGE_GROUP"],
    df["CVD"],
    margins=True,
    dropna=False
))

print("\n===== NHOM CHUA CO NHAN =====")
missing = df[df["CVD"].isna()]

print("Tong so:", len(missing))
print("Duoi 20 tuoi:", (missing["RIDAGEYR"] < 20).sum())
print("Tu 20 tuoi tro len:", (missing["RIDAGEYR"] >= 20).sum())

print("\n===== HOAN THANH =====")
