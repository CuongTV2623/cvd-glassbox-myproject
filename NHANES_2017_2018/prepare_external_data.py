
from pathlib import Path
import pandas as pd
import joblib

# =========================
# 1. PATHS
# =========================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "processed"
PREPROCESSOR_DIR = ROOT.parent / "NHANES_2021_2023" / "preprocessors"

INPUT_FILE = DATA_DIR / "NHANES_adult_merged.csv"

# =========================
# 2. FEATURE DEFINITIONS
# =========================
NUMERIC_FEATURES = [
    "RIDAGEYR",
    "INDFMPIR",
    "BMXBMI",
    "BMXWAIST",
    "PAD680"
]

CATEGORICAL_T0 = [
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "SMQ020"
]

CATEGORICAL_T1 = CATEGORICAL_T0 + [
    "BPQ020",
    "BPQ080",
    "DIQ010"
]

# =========================
# 3. LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

# Xu ly ma dac biet cua PAD680
df["PAD680"] = pd.to_numeric(df["PAD680"], errors="coerce")
df["PAD680"] = df["PAD680"].replace([7777, 9999], float("nan"))

print("===== EXTERNAL DATA CHECK =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("CVD distribution:")
print(df["CVD"].value_counts(dropna=False))

# =========================
# 4. PREPARE T0 AND T1
# =========================
for tier, categorical_features in [
    ("T0", CATEGORICAL_T0),
    ("T1", CATEGORICAL_T1)
]:
    print(f"\n===== PREPARING {tier} =====")

    features = NUMERIC_FEATURES + categorical_features

    missing_columns = [
        col for col in features + ["SEQN", "CVD"]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for {tier}: {missing_columns}"
        )

    # Giữ đúng thứ tự đặc trưng khi huấn luyện
    X = df[features].copy()
    y = df["CVD"].copy()

    preprocessor_path = (
        PREPROCESSOR_DIR / f"preprocessor_{tier}.joblib"
    )

    preprocessor = joblib.load(preprocessor_path)

    # Chỉ transform, tuyệt đối không fit lại
    X_processed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()

    X_processed = pd.DataFrame(
        X_processed,
        columns=feature_names
    )

    output = pd.concat(
        [
            df[["SEQN", "CVD"]].reset_index(drop=True),
            X_processed.reset_index(drop=True)
        ],
        axis=1
    )

    output_file = DATA_DIR / f"NHANES_{tier}_external_processed.csv"
    output.to_csv(output_file, index=False)

    print("Features:", X_processed.shape[1])
    print("Output:", output_file)
    print("Shape:", output.shape)

print("\nDa chuan bi xong du lieu external validation.")
