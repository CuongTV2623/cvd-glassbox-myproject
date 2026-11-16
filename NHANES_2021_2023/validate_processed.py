
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent

for split in ["train", "validation", "test"]:

    print(f"\n===== KIEM TRA {split.upper()} =====")

    t0 = pd.read_csv(
        folder / f"NHANES_T0_{split}_processed.csv"
    )

    t1 = pd.read_csv(
        folder / f"NHANES_T1_{split}_processed.csv"
    )

    print("T0 shape:", t0.shape)
    print("T1 shape:", t1.shape)

    print("T0 missing:", t0.isna().sum().sum())
    print("T1 missing:", t1.isna().sum().sum())

    print("T0 SEQN duplicates:", t0["SEQN"].duplicated().sum())
    print("T1 SEQN duplicates:", t1["SEQN"].duplicated().sum())

    print("Same SEQN:", set(t0["SEQN"]) == set(t1["SEQN"]))
    print("Same CVD:", (
        t0.set_index("SEQN")["CVD"].sort_index().equals(
            t1.set_index("SEQN")["CVD"].sort_index()
        )
    ))

    print("T0 label distribution:")
    print(t0["CVD"].value_counts().sort_index().to_dict())

    print("T1 label distribution:")
    print(t1["CVD"].value_counts().sort_index().to_dict())
