
"""
Model registry for NHANES 2021-2023 benchmark.

Input:
    - X: pandas DataFrame containing already-preprocessed numeric features
    - y: binary target (0/1)

Models:
    1. Logistic Regression
    2. Logistic Regression + Splines
    3. Random Forest
    4. MLP
    5. Explainable Boosting Machine (EBM)
    6. XGBoost
    7. LightGBM
    8. CatBoost
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_models(random_state=42, feature_names=None):
    """
    Return available CPU models.

    Parameters
    ----------
    random_state : int
        Random seed.
    feature_names : list
        Names of preprocessed feature columns.
    """

    models = {}

    # 1. Logistic Regression
    models["LOGREG"] = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs"
    )

    # 2. Logistic Regression + Splines
    # Apply splines only to continuous numeric features.
    if feature_names is not None:
        numeric_features = [
            col for col in feature_names
            if col.startswith("num__")
        ]

        categorical_features = [
            col for col in feature_names
            if col.startswith("cat__")
        ]

        if numeric_features:
            spline_preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "spline",
                        SplineTransformer(
                            n_knots=5,
                            degree=3,
                            include_bias=False
                        ),
                        numeric_features
                    ),
                    (
                        "categorical",
                        "passthrough",
                        categorical_features
                    )
                ],
                remainder="drop"
            )

            models["LOGREG_SPLINE"] = Pipeline([
                ("preprocessor", spline_preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state
                    )
                )
            ])

    # 3. Random Forest
    models["RF"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    # 4. Multi-Layer Perceptron
    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        random_state=random_state
    )

    # 5. Explainable Boosting Machine
    try:
        from interpret.glassbox import ExplainableBoostingClassifier

        models["EBM"] = ExplainableBoostingClassifier(
            outer_bags=14,
            interactions=10,
            max_bins=256,
            learning_rate=0.01,
            random_state=random_state,
            n_jobs=-1
        )
    except ImportError:
        print("[WARNING] interpret not installed. Skipping EBM.")

    # 6. XGBoost
    try:
        from xgboost import XGBClassifier

        models["XGBOOST"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=1.0,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist"
        )
    except ImportError:
        print("[WARNING] xgboost not installed. Skipping XGBoost.")

    # 7. LightGBM
    try:
        from lightgbm import LGBMClassifier

        models["LIGHTGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
    except ImportError:
        print("[WARNING] lightgbm not installed. Skipping LightGBM.")

    # 8. CatBoost
    try:
        from catboost import CatBoostClassifier

        models["CATBOOST"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False
        )
    except ImportError:
        print("[WARNING] catboost not installed. Skipping CatBoost.")

    return models
