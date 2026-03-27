"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""

import pandas as pd
import numpy as np

import logging

from src.utils import safe_exp

logger = logging.getLogger(__name__)


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: Fitted scikit-learn Pipeline.
    - X_infer: Inference features DataFrame (same raw columns as training).
    Outputs:
    - df_pred: DataFrame with a SINGLE column named 'prediction' preserving
    input index.
    Why this contract matters for reliable ML delivery:
    - A stable prediction schema simplifies integrations (batch jobs, APIs)
    and reduces downstream breaking changes.
    """

    logger.info("Running inference and returning prediction.")

    preds = model.predict(X_infer)
    df_pred = pd.DataFrame({"prediction": safe_exp(preds)}, index=X_infer.index)

    return df_pred
