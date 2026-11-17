
from pathlib import Path
import pandas as pd

# ==========================================
# 1. PATH
# ==========================================

folder = Path(__file__).resolve().parent
processed = folder / "processed"

df = pd.read_csv(
    processed / "NHANES_adult_labeled.csv"
)

print("So dong ban dau:", len(df))


# ==========================================
# 2. MERGE FEATURE TABLES
# ==========================================

files = [
    "BMX_J.xpt",
    "BPQ_J.xpt",
    "DIQ_J.xpt",
    "SMQ_J.xpt",
    "ALQ_J.xpt",
    "PAQ_J.xpt"
]

for filename in files:

    data = pd.read_sas(
        folder / filename,
        format="xport"
    )

    df = df.merge(
        data,
        on="SEQN",
        how="left",
        validate="one_to_one"
    )

    print(
        f"{filename}: {len(df)} rows, "
        f"{len(data.columns)} columns"
    )


# ==========================================
# 3. CHECK DATA
# ==========================================

print("\n===== MERGE RESULTS =====")

print("So dong:", len(df))
print("So cot:", len(df.columns))
print("SEQN trung lap:", df["SEQN"].duplicated().sum())

print("\n===== MISSING VALUES =====")

missing = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(missing.head(15).round(2))


# ==========================================
# 4. SAVE
# ==========================================

output = processed / "NHANES_adult_merged.csv"

df.to_csv(output, index=False)

print("\nDa luu:", output)
