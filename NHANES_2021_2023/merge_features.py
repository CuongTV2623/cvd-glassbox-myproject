
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent

# Cohort người trưởng thành đã có nhãn
df = pd.read_csv(folder / "NHANES_adult_labeled.csv")

print("So dong ban dau:", len(df))

# Các bảng đặc trưng cần ghép
files = [
    "BMX_L.xpt",
    "BPQ_L.xpt",
    "DIQ_L.xpt",
    "SMQ_L.xpt",
    "ALQ_L.xpt",
    "PAQ_L.xpt"
]

for filename in files:
    data = pd.read_sas(folder / filename, format="xport")

    # Ghép theo mã người tham gia
    df = df.merge(
        data,
        on="SEQN",
        how="left",
        validate="one_to_one"
    )

    print(f"{filename}: {len(df)} rows, {len(data.columns)} columns")

# Kiểm tra kết quả
print("\n===== KET QUA GHEP DU LIEU =====")
print("So dong:", len(df))
print("So cot:", len(df.columns))
print("So SEQN trung lap:", df["SEQN"].duplicated().sum())

# Thống kê tỷ lệ thiếu của từng cột
missing = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n===== 15 COT THIEU DU LIEU NHIEU NHAT =====")
print(missing.head(15).round(2))

# Lưu dữ liệu
output = folder / "NHANES_adult_merged.csv"
df.to_csv(output, index=False)

print("\nDa luu:", output.name)
