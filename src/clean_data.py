"""
Module: Data Cleaning
---------------------
Role: Dataset-specific cleaning for the insurance dataset.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Clean).

Supports both:
- Training mode: raw data includes "charges" and target_column is provided
- Inference mode: raw data does not include "charges" and target_column may be None
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe(
    df_raw: pd.DataFrame,
    target_column: str | None,
    required_columns: set,
    categorical_columns: set,
    numeric_columns: set,
) -> pd.DataFrame:
    """
    Clean raw insurance data into a stable, model-ready dataset.

    Behavior:
    - Always validates required feature columns
    - Normalizes categorical columns
    - Coerces numeric columns
    - If "charges" exists and target_column is provided, creates target_column = log(charges)
    - If "charges" does not exist, returns cleaned feature-only dataframe for inference
    """

    logger.info("Cleaning dataframe")

    if df_raw is None:
        raise ValueError("df_raw cannot be None")

    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("df_raw must be a pandas DataFrame")

    if df_raw.empty:
        raise ValueError("df_raw is empty")

    df = df_raw.copy()

    # 1) Standardize column names
    df.columns = [str(c).strip() for c in df.columns]

    # 2) Validate required raw schema
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # 3) Drop duplicates
    df = df.drop_duplicates()
    logger.info("Dropped duplicates, %s rows remain", len(df))

    # 4) Normalize categoricals
    for c in categorical_columns:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    if "smoker" in df.columns:
        df["smoker"] = df["smoker"].replace({"y": "yes", "n": "no"})

    # 5) Coerce numeric columns
    for c in numeric_columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise")

    # 6) Training mode only: create target from charges
    if "charges" in df.columns:
        if not target_column or not isinstance(target_column, str):
            raise ValueError(
                "target_column must be a non-empty string when 'charges' is present"
            )

        df["charges"] = pd.to_numeric(df["charges"], errors="raise")

        if (df["charges"] <= 0).any():
            bad_n = int((df["charges"] <= 0).sum())
            raise ValueError(
                f"'charges' must be positive to compute log; found {bad_n} non-positive rows."
            )

        df[target_column] = np.log(df["charges"]).astype(float)
        df = df.drop(columns=["charges"])
        logger.info("Created target column '%s' from charges", target_column)

    logger.info(
        "Cleaned dataframe with %s rows and %s columns",
        len(df),
        len(df.columns),
    )

    return df
