
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent
df = pd.read_csv(folder / "NHANES_adult_clean.csv")

# Nhóm T0: thông tin nhân khẩu học,
# nhân trắc học và hành vi cơ bản
T0 = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "INDFMPIR",
    "BMXBMI",
    "BMXWAIST",
    "SMQ020",
    "PAD680"
]

# Nhóm T1: bổ sung thông tin bệnh lý đã được chẩn đoán
T1 = T0 + [
    "BPQ020",
    "BPQ080",
    "DIQ010"
]

# Giữ cùng các bản ghi, không xóa người chỉ vì thiếu đặc trưng
base_cols = ["SEQN", "CVD"]

df_T0 = df[base_cols + T0].copy()
df_T1 = df[base_cols + T1].copy()

df_T0.to_csv(folder / "NHANES_T0.csv", index=False)
df_T1.to_csv(folder / "NHANES_T1.csv", index=False)

print("===== KET QUA TAO DATASET =====")
print("T0:", df_T0.shape)
print("T1:", df_T1.shape)

print("\nSo dong T0:", len(df_T0))
print("So dong T1:", len(df_T1))

print("\nSo nhan CVD:")
print(df_T0["CVD"].value_counts().sort_index())

print("\nTy le thieu T0 (%):")
print((df_T0[T0].isna().mean() * 100).round(2))

print("\nTy le thieu T1 (%):")
print((df_T1[T1].isna().mean() * 100).round(2))
