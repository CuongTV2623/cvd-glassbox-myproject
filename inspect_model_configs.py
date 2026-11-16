
from pathlib import Path
import yaml

models = [
    "ebm",
    "logreg",
    "logreg_spline",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp"
]

folder = Path("configs/models")

for model in models:
    path = folder / f"{model}.yaml"

    print(f"\n===== {model.upper()} =====")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    print(yaml.dump(
        config.get("defaults", {}),
        allow_unicode=True,
        sort_keys=False
    ))
