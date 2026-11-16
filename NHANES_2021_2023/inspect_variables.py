
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent

files = [
    "MCQ_L.xpt",
    "DEMO_L.xpt",
    "BMX_L.xpt",
    "BPQ_L.xpt",
    "DIQ_L.xpt",
    "SMQ_L.xpt",
    "ALQ_L.xpt",
    "PAQ_L.xpt"
]

for filename in files:
    df = pd.read_sas(folder / filename, format="xport")

    print(f"\n===== {filename} =====")
    print(", ".join(df.columns))

# Kiem tra nhan benh tim mach
mcq = pd.read_sas(folder / "MCQ_L.xpt", format="xport")

target_vars = [
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F"
]

print("\n===== KIEM TRA NHAN TIM MACH =====")

for col in target_vars:
    if col in mcq.columns:
        print(f"\n{col}")
        print(mcq[col].value_counts(dropna=False))
