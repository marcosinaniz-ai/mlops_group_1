"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""
"""
Educational Goal:
- Why this module exists in an MLOps system: Validation catches obvious data contract breaks early (cheap) before training (expensive).
- Responsibility (separation of concerns): Fail fast on empty data and missing required columns; keep checks minimal and readable.
- Pipeline contract (inputs and outputs): Input df + required column list; output True if valid, otherwise raise.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, config: dict) -> bool:
    """
    Inputs:
    - df: DataFrame to validate.
    - required_columns: List of columns that must exist in df.
    Outputs:
    - is_valid: True if valid; raises ValueError otherwise.
    Why this contract matters for reliable ML delivery:
    - Simple, explicit validation prevents silent training on wrong schemas and reduces costly downstream debugging.
    """
    print("[validate.validate_dataframe] Validating dataframe (fail fast for empty/missing columns)")  # TODO: replace with logging later

    if df is None or df.empty:
        raise ValueError("Validation failed: DataFrame is empty. Check data ingestion and cleaning steps.")


    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------

    schema_cfg = config.get("schema", {})
    required_columns = schema_cfg.get("required_columns", [])
    target = schema_cfg.get("target")

    if not required_columns:
        raise ValueError("Validation failed: config['schema']['required_columns'] is missing or empty.")

    # Required columns must exist
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Validation failed: Missing required columns: {missing}. Present columns: {list(df.columns)}"
        )

    # Target must exist (if defined)
    if target and target not in df.columns:
        raise ValueError(f"Validation failed: Target column '{target}' is missing.")

    # Missing values must be explicit (fail fast here)
    missing_counts = df[required_columns].isna().sum()
    if (missing_counts > 0).any():
        bad = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Validation failed: Missing values detected: {bad}")

    # Domain checks
    domain = schema_cfg.get("domain", {})

    for col in domain.get("non_negative", []):
        if col in df.columns and (df[col] < 0).any():
            raise ValueError(f"Validation failed: column '{col}' has negative values.")

    for col in domain.get("positive", []):
        if col in df.columns and (df[col] <= 0).any():
            raise ValueError(f"Validation failed: column '{col}' has non-positive values.")

    allowed_values = domain.get("allowed_values", {})
    for col, allowed in allowed_values.items():
        if col in df.columns:
            allowed_norm = {str(x).lower() for x in allowed}
            observed_norm = set(df[col].dropna().astype(str).str.lower().unique())
            bad_vals = observed_norm - allowed_norm
            if bad_vals:
                raise ValueError(f"Validation failed: column '{col}' has invalid values: {sorted(bad_vals)}")

    return True
    

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

 