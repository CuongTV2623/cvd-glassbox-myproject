
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.model_registry import get_models

DATA_DIR = ROOT 

print("ROOT:", ROOT)
print("DATA_DIR:", DATA_DIR)
print("FILE EXISTS:", (DATA_DIR / "NHANES_T0_train_processed.csv").exists())

train = pd.read_csv(
    DATA_DIR / "NHANES_T0_train_processed.csv"
)

X = train.drop(columns=["SEQN", "CVD"])
feature_names = X.columns.tolist()

models = get_models(
    random_state=42,
    feature_names=feature_names
)

print("\n===== NHANES MODEL REGISTRY =====")

for name, model in models.items():
    print(name, ":", type(model).__name__)

print("\nTotal models:", len(models))
