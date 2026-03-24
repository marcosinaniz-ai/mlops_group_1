"""
Module: Features / Preprocessing
--------------------------------
Role: Define a consistent preprocessing recipe for training & inference.

This version matches src.main usage:
  preprocessor = get_feature_preprocessor(numeric_cols=[...], categorical_cols=[...])

Returns: an UNFITTED sklearn ColumnTransformer.
"""

from typing import List, Sequence

import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def get_feature_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ColumnTransformer:
    """
    Build an UNFITTED preprocessing recipe.

    - Numeric: median impute + standardize
    - Categorical: most_frequent impute + one-hot
    """

    logger.info("Building feature preprocessor")

    num_cols = list(numeric_cols) if numeric_cols is not None else []
    cat_cols = list(categorical_cols) if categorical_cols is not None else []

    if not num_cols and not cat_cols:
        raise ValueError("numeric_cols and categorical_cols are both empty; cannot build preprocessor.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # sklearn compatibility: sparse_output introduced in newer versions
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

    logger.info("Feature preprocessor built")

    return ColumnTransformer(transformers=transformers, remainder="drop")
