"""
Module: Model Training
----------------------
Notebook logic:
- Pipeline(preprocess, LinearRegression)
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
):
    logger.info("Training LinearRegression inside a Pipeline")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    return model
