"""
Module: Data Loader
-------------------
Role: Ingest raw data from a CSV file.
Input: Path to CSV.
Output: pandas.DataFrame (Raw).
"""

from pathlib import Path
import pandas as pd

from src.utils import load_csv


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Load raw insurance data from disk.

    Contract:
      - Input: Path to a CSV file
      - Output: Non-empty pandas.DataFrame
      - Fail-fast: raises clear exceptions if path is invalid
    """
    if raw_data_path is None:
        raise ValueError("raw_data_path cannot be None")

    raw_data_path = Path(raw_data_path)

    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {raw_data_path}. "
            "Place the insurance CSV there or update config.yaml."
        )

    if not raw_data_path.is_file():
        raise IsADirectoryError(f"raw_data_path must be a file, got: {raw_data_path}")

    df_raw = load_csv(raw_data_path)

    if df_raw is None or df_raw.empty:
        raise ValueError(f"Loaded dataframe is empty from: {raw_data_path}")

    return df_raw