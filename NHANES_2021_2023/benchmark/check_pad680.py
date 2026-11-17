
import pandas as pd

files = {
    "2021-2023": "NHANES_2021_2023/NHANES_T1.csv",
    "2017-2018": "NHANES_2017_2018/processed/NHANES_adult_merged.csv"
}

for name, file in files.items():
    df = pd.read_csv(file)

    print(f"\n===== {name} =====")

    print(df["PAD680"].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    ))

    print("\nGiá trị lớn hơn 1440:")
    print(df.loc[df["PAD680"] > 1440, "PAD680"].value_counts())

    print("\nGiá trị đặc biệt:")
    print(df["PAD680"].value_counts().head(15))
