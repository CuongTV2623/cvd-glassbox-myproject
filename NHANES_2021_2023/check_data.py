
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

print("===== KIEM TRA DU LIEU NHANES =====")

for filename in files:
    path = folder / filename

    if not path.exists():
        print(f"\n[THIEU FILE] {filename}")
        continue

    df = pd.read_sas(path, format="xport")

    print(f"\n--- {filename} ---")
    print("So dong:", len(df))
    print("So cot:", len(df.columns))
    print("Co SEQN:", "SEQN" in df.columns)

    if "SEQN" in df.columns:
        print("SEQN trung lap:", df["SEQN"].duplicated().sum())

print("\n===== HOAN THANH =====")
