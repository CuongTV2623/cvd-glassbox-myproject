
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

folder = Path(__file__).parent

df = pd.read_csv(folder / "NHANES_T0.csv")

# Chia train 60%, phần còn lại 40%
train_ids, temp_ids = train_test_split(
    df["SEQN"],
    test_size=0.4,
    random_state=42,
    stratify=df["CVD"]
)

# Chia phần còn lại thành validation 20% và test 20%
temp = df[df["SEQN"].isin(temp_ids)]

val_ids, test_ids = train_test_split(
    temp["SEQN"],
    test_size=0.5,
    random_state=42,
    stratify=temp["CVD"]
)

split = pd.DataFrame({
    "SEQN": df["SEQN"],
    "split": "train"
})

split.loc[split["SEQN"].isin(val_ids), "split"] = "validation"
split.loc[split["SEQN"].isin(test_ids), "split"] = "test"

split.to_csv(folder / "NHANES_split.csv", index=False)

print("===== KET QUA CHIA DU LIEU =====")

for name in ["train", "validation", "test"]:
    ids = split.loc[split["split"] == name, "SEQN"]
    subset = df[df["SEQN"].isin(ids)]

    print(f"\n{name.upper()}:")
    print("So dong:", len(subset))
    print(subset["CVD"].value_counts().sort_index())
