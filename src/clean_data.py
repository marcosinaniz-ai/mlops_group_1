"""
Module: Data Cleaning
---------------------
Role: Dataset-specific cleaning for the insurance dataset.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Clean).
"""

import pandas as pd


REQUIRED_COLUMNS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
CATEGORICAL_COLUMNS = {"sex", "smoker", "region"}
NUMERIC_COLUMNS = {"age", "bmi", "children", "charges"}


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean raw insurance data into a stable, model-ready dataset.

    Cleaning rules:
      - Standardize column names (strip)
      - Validate required columns exist
      - Drop duplicate rows
      - Coerce numeric columns to numeric (raise if impossible)
      - Normalize categorical strings (lower/strip)
      - Enforce target column exists and is numeric
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

    # 2) Validate schema
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # 3) Target validation
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"
        )

    # 4) Drop duplicates
    df = df.drop_duplicates()

    # 5) Normalize categoricals
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # Optional: normalize common values
    # sex: male/female, smoker: yes/no
    # (keep it light so you don't break unexpected variants)
    if "smoker" in df.columns:
        df["smoker"] = df["smoker"].replace({"y": "yes", "n": "no"})

    # 6) Coerce numerics
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="raise")

    return df