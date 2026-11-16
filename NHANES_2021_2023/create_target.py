
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent

# Doc du lieu benh ly
mcq = pd.read_sas(
    folder / "MCQ_L.xpt",
    format="xport"
)

# 5 bien benh tim mach
cvd_vars = [
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F"
]

# Tao nhan tong hop
answers = mcq[cvd_vars]

# Co it nhat mot benh
has_cvd = answers.eq(1).any(axis=1)

# Khong mac ca 5 benh
no_cvd = answers.eq(2).all(axis=1)

mcq["CVD"] = pd.NA
mcq.loc[has_cvd, "CVD"] = 1
mcq.loc[no_cvd, "CVD"] = 0

# Thong ke
print("===== KET QUA TAO NHAN CVD =====")
print(mcq["CVD"].value_counts(dropna=False))

print("\n===== TY LE NHAN =====")
print(mcq["CVD"].value_counts(normalize=True, dropna=False))

print("\nTong so nguoi:", len(mcq))
print("So nguoi co nhan:", mcq["CVD"].notna().sum())

# Luu ket qua
mcq[["SEQN", "CVD"]].to_csv(
    folder / "NHANES_CVD_target.csv",
    index=False
)

print("\nDa luu NHANES_CVD_target.csv")
