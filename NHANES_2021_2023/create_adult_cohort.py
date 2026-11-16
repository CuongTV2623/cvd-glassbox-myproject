
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent

# Doc du lieu
mcq = pd.read_sas(folder / "MCQ_L.xpt", format="xport")
demo = pd.read_sas(folder / "DEMO_L.xpt", format="xport")

# Tao nhan CVD
cvd_vars = [
    "MCQ160B", "MCQ160C", "MCQ160D",
    "MCQ160E", "MCQ160F"
]

answers = mcq[cvd_vars]

has_cvd = answers.eq(1).any(axis=1)
no_cvd = answers.eq(2).all(axis=1)

mcq["CVD"] = pd.NA
mcq.loc[has_cvd, "CVD"] = 1
mcq.loc[no_cvd, "CVD"] = 0

# Ghep thong tin nhan khau hoc
df = mcq.merge(
    demo,
    on="SEQN",
    how="inner",
    validate="one_to_one"
)

# Chi lay nguoi tu 20 tuoi tro len
adults = df[df["RIDAGEYR"] >= 20].copy()

# Tach nhom co nhan va thieu nhan
labeled = adults[adults["CVD"].notna()].copy()
unlabeled = adults[adults["CVD"].isna()].copy()

# Luu file
adults.to_csv(folder / "NHANES_adult_all.csv", index=False)
labeled.to_csv(folder / "NHANES_adult_labeled.csv", index=False)
unlabeled.to_csv(folder / "NHANES_adult_unlabeled.csv", index=False)

print("===== KET QUA COHORT =====")
print("Tong nguoi truong thanh:", len(adults))
print("Nguoi co nhan:", len(labeled))
print("Nguoi thieu nhan:", len(unlabeled))
print("\nPhan bo CVD:")
print(labeled["CVD"].value_counts().sort_index())

print("\nDa luu 3 file cohort.")
