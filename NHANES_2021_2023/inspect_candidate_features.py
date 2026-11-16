
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent
df = pd.read_csv(folder / "NHANES_adult_merged.csv")

candidate_features = [
    # Demographics
    "RIDAGEYR", "RIAGENDR", "RIDRETH3",
    "DMDEDUC2", "INDFMPIR",

    # Body measurements
    "BMXBMI", "BMXWAIST", "BMXWT", "BMXHT",

    # Blood pressure and cholesterol
    "BPQ020", "BPQ030", "BPQ080",

    # Diabetes
    "DIQ010",

    # Smoking
    "SMQ020", "SMQ040",

    # Alcohol
    "ALQ111", "ALQ121", "ALQ130",
    "ALQ142", "ALQ151", "ALQ170",

    # Physical activity
    "PAD790Q", "PAD800", "PAD810Q",
    "PAD820", "PAD680"
]

features = [col for col in candidate_features if col in df.columns]

print("===== KIEM TRA BIEN DAU VAO =====")
print("So bien:", len(features))

print("\n===== TY LE THIEU DU LIEU =====")
missing = df[features].isna().mean().mul(100).sort_values(ascending=False)
print(missing.round(2).to_string())

print("\n===== GIA TRI CUA CAC BIEN =====")
for col in features:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head(10).to_string())
