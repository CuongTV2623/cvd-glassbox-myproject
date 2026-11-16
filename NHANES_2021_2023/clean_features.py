
from pathlib import Path
import pandas as pd
import numpy as np

folder = Path(__file__).parent
df = pd.read_csv(folder / "NHANES_adult_merged.csv")

# Các biến có giá trị 0 hợp lệ nhưng đang được biểu diễn
# bằng số rất nhỏ khi đọc file XPT
zero_valid_vars = [
    "INDFMPIR",
    "ALQ121", "ALQ142", "ALQ170",
    "PAD790Q", "PAD810Q", "PAD680"
]

for col in zero_valid_vars:
    if col in df.columns:
        df[col] = df[col].mask(
            df[col].abs() < 1e-70, 0
        )

# Mã từ chối trả lời / không biết
special_codes = {
    "DMDEDUC2": [7, 9],
    "BPQ020": [7, 9],
    "BPQ080": [7, 9],
    "DIQ010": [7, 9],
    "SMQ020": [7, 9],
    "ALQ121": [77, 99],
    "ALQ142": [77, 99],
    "ALQ130": [777, 999],
    "ALQ170": [777, 999],
    "PAD790Q": [7777, 9999],
    "PAD810Q": [7777, 9999],
    "PAD680": [7777, 9999]
}

for col, codes in special_codes.items():
    if col in df.columns:
        df[col] = df[col].replace(codes, np.nan)

df.to_csv(folder / "NHANES_adult_clean.csv", index=False)

print("Da luu NHANES_adult_clean.csv")
print("So dong:", len(df))

print("\n===== TY LE THIEU SAU LAM SACH =====")
check_vars = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3",
    "DMDEDUC2", "INDFMPIR",
    "BMXBMI", "BMXWAIST",
    "BPQ020", "BPQ080", "DIQ010",
    "SMQ020", "ALQ121", "ALQ142",
    "PAD790Q", "PAD810Q", "PAD680"
]

print(
    (df[check_vars].isna().mean() * 100)
    .round(2)
    .sort_values(ascending=False)
    .to_string()
)
