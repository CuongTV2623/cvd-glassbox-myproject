
from pathlib import Path
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

folder = Path(__file__).parent

# Doc thong tin chia tap
split = pd.read_csv(folder / "NHANES_split.csv")

# Khai bao bien so
numeric_features = [
    "RIDAGEYR",
    "INDFMPIR",
    "BMXBMI",
    "BMXWAIST",
    "PAD680"
]

# Khai bao bien phan loai T0
categorical_T0 = [
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "SMQ020"
]

# T1 bo sung cac bien benh ly
categorical_T1 = categorical_T0 + [
    "BPQ020",
    "BPQ080",
    "DIQ010"
]

for tier in ["T0", "T1"]:

    print(f"\n===== XU LY {tier} =====")

    df = pd.read_csv(folder / f"NHANES_{tier}.csv")

    # Gan nhan chia tap theo SEQN
    df = df.merge(
        split,
        on="SEQN",
        how="left",
        validate="one_to_one"
    )

    categorical_features = (
        categorical_T0 if tier == "T0"
        else categorical_T1
    )

    # Pipeline cho bien so
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline cho bien phan loai
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Ket hop hai pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        sparse_threshold=0
    )

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    X_train = train[numeric_features + categorical_features]
    X_val = val[numeric_features + categorical_features]
    X_test = test[numeric_features + categorical_features]

    # Chi fit tren TRAIN
    X_train_processed = preprocessor.fit_transform(X_train)

    # Validation va Test chi transform
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Tao DataFrame da xu ly
    for name, part, X_processed in [
        ("train", train, X_train_processed),
        ("validation", val, X_val_processed),
        ("test", test, X_test_processed)
    ]:
        result = pd.DataFrame(
            X_processed,
            columns=feature_names
        )

        result.insert(0, "CVD", part["CVD"].values)
        result.insert(0, "SEQN", part["SEQN"].values)

        result.to_csv(
            folder / f"NHANES_{tier}_{name}_processed.csv",
            index=False
        )

    print("So bien sau One-Hot:", len(feature_names))
    print("Train:", X_train_processed.shape)
    print("Validation:", X_val_processed.shape)
    print("Test:", X_test_processed.shape)

print("\nDa xu ly xong ca T0 va T1.")
