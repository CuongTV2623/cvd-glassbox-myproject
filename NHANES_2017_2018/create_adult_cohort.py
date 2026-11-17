
from pathlib import Path
import pandas as pd

# ==========================================
# 1. PATH
# ==========================================

ROOT = Path(__file__).resolve().parent

DEMO_FILE = ROOT / "DEMO_J.xpt"
MCQ_FILE = ROOT / "MCQ_J.xpt"

OUTPUT_DIR = ROOT / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("===== LOADING DATA =====")

demo = pd.read_sas(DEMO_FILE, format="xport")
mcq = pd.read_sas(MCQ_FILE, format="xport")

print("DEMO:", demo.shape)
print("MCQ:", mcq.shape)


# ==========================================
# 3. MERGE DEMO AND MCQ
# ==========================================

df = demo.merge(
    mcq,
    on="SEQN",
    how="inner",
    validate="one_to_one"
)

print("Merged data:", df.shape)


# ==========================================
# 4. SELECT ADULTS
# ==========================================

df = df[df["RIDAGEYR"] >= 20].copy()

print("Adults:", len(df))


# ==========================================
# 5. CREATE CVD LABEL
# ==========================================

cvd_columns = [
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F"
]

print("\n===== CHECK CVD VARIABLES =====")

for col in cvd_columns:
    print(col, ":", col in df.columns)

# Convert to numeric
df[cvd_columns] = df[cvd_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# CVD = 1 if any answer is YES (1)
has_cvd = df[cvd_columns].eq(1).any(axis=1)

# CVD = 0 only if all five answers are NO (2)
no_cvd = df[cvd_columns].eq(2).all(axis=1)

df["CVD"] = pd.NA

df.loc[has_cvd, "CVD"] = 1
df.loc[no_cvd, "CVD"] = 0

df["CVD"] = df["CVD"].astype("Int64")


# ==========================================
# 6. SPLIT LABELED / UNLABELED
# ==========================================

labeled = df[df["CVD"].notna()].copy()
unlabeled = df[df["CVD"].isna()].copy()


# ==========================================
# 7. SAVE RESULTS
# ==========================================

df.to_csv(
    OUTPUT_DIR / "NHANES_adult_all.csv",
    index=False
)

labeled.to_csv(
    OUTPUT_DIR / "NHANES_adult_labeled.csv",
    index=False
)

unlabeled.to_csv(
    OUTPUT_DIR / "NHANES_adult_unlabeled.csv",
    index=False
)


# ==========================================
# 8. SUMMARY
# ==========================================

print("\n===== COHORT RESULTS =====")

print("Total adults:", len(df))
print("Labeled:", len(labeled))
print("Unlabeled:", len(unlabeled))

print("\nCVD distribution:")
print(labeled["CVD"].value_counts().sort_index())

print("\nCVD percentage:")
print(
    labeled["CVD"].value_counts(normalize=True).sort_index()
    .round(4)
)

print("\n===== COMPLETED =====")
print("Output directory:", OUTPUT_DIR)
