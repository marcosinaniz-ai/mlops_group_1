"""
Educational Goal:
- Why this module exists in an MLOps system: Feature preprocessing must be consistent between training and inference to avoid skew.
- Responsibility (separation of concerns): Define the feature transformation recipe (not fitting it) using ColumnTransformer.
- Pipeline contract (inputs and outputs): Input is column-name configuration; output is an unfitted preprocessor object.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import List, Optional




from typing import Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(config: Dict) -> ColumnTransformer:
    """
    Build an UNFITTED preprocessing recipe (blueprint).
    Reads feature lists from config.yaml.
    """

    features_cfg = config.get("features", {})
    num_cols: List[str] = features_cfg.get("numerical", []) or []
    cat_cols: List[str] = features_cfg.get("categorical", []) or []

    if not num_cols and not cat_cols:
        raise ValueError("Feature lists are empty in config['features'].")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    transformers = []

    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))

    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor