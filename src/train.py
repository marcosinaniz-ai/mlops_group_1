"""
Module: Model Training
----------------------
Notebook logic:
- Pipeline(preprocess, LinearRegression)
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
):
    print("[train.train_model] Training LinearRegression inside a Pipeline")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    return model