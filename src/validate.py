"""
Module: Data Validation
-----------------------
Role: Check data quality (schema) before training.
Input: pandas.DataFrame + required columns + target column name.
Output: True if valid, otherwise raises ValueError.

This version matches src.main usage:
  validate_dataframe(df_clean, required_columns=[...], target_column=...)
"""

from __future__ import annotations
from typing import Iterable, List, Optional
import pandas as pd

    # --------------------------------
    # START STUDENT CODE
    # --------------------------------
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    target_column: Optional[str] = None,
) -> bool:
    """
    Fail-fast validation:
      - df is non-empty
      - all required_columns exist
      - target_column exists (if provided)
    """
    print("[validate.validate_dataframe] Validating dataframe (fail fast for empty/missing columns)")

    if df is None or df.empty:
        raise ValueError("Validation failed: DataFrame is empty. Check data ingestion and cleaning steps.")

    required_columns_list: List[str] = list(required_columns) if required_columns is not None else []
    if not required_columns_list:
        raise ValueError("Validation failed: required_columns is missing or empty.")

    missing = [c for c in required_columns_list if c not in df.columns]
    if missing:
        raise ValueError(
            f"Validation failed: Missing required columns: {missing}. Present columns: {list(df.columns)}"
        )

    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Validation failed: Target column '{target_column}' is missing.")
        # Basic sanity: target should be numeric for this regression pipeline
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValueError(f"Validation failed: Target column '{target_column}' must be numeric.")

    return True
    # ----------------------------
    # END STUDENT CODE
    # ----------------------------