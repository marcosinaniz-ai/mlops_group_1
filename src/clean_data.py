"""
Module: Data Cleaning
---------------------
Role: Dataset-specific cleaning for the insurance dataset.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Clean).

This version matches src.main expectations:
- Adds target_column (e.g. "log_charges") computed from "charges"
- Drops "charges" afterward
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
CATEGORICAL_COLUMNS = {"sex", "smoker", "region"}
NUMERIC_COLUMNS = {"age", "bmi", "children", "charges"}


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean raw insurance data into a stable, model-ready dataset.

    Contract aligned with main.py:
      - target_column is the name of the *output* target column (e.g. "log_charges")
      - "charges" must exist in raw input
      - output contains target_column and does NOT contain "charges"
    """
    if df_raw is None:
        raise ValueError("df_raw cannot be None")
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("df_raw must be a pandas DataFrame")
    if df_raw.empty:
        raise ValueError("df_raw is empty")
    if not target_column or not isinstance(target_column, str):
        raise ValueError("target_column must be a non-empty string")

    df = df_raw.copy()

    # 1) Standardize column names
    df.columns = [str(c).strip() for c in df.columns]

    # 2) Validate required raw schema (must include charges)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # 3) Drop duplicates
    df = df.drop_duplicates()

    # 4) Normalize categoricals
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # Normalize common variants lightly
    if "smoker" in df.columns:
        df["smoker"] = df["smoker"].replace({"y": "yes", "n": "no"})

    # 5) Coerce numeric columns
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # 6) Create target column from charges
    # Use log(charges) exactly like notebook expectation; require positive charges
    if (df["charges"] <= 0).any():
        bad_n = int((df["charges"] <= 0).sum())
        raise ValueError(f"'charges' must be positive to compute log; found {bad_n} non-positive rows.")

    df[target_column] = np.log(df["charges"]).astype(float)

    # 7) Drop raw target
    df = df.drop(columns=["charges"])

    return df