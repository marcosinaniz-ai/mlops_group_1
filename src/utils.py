"""
Module: Utilities
-----------------
Shared helpers for IO and small safe transforms.

This version matches main.py usage and evaluate.py dependencies:
- save_csv(..., index=bool)  <-- main passes index=True for predictions
- save_json(...)
- safe_exp(...)              <-- evaluate uses for dollar-space metrics
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Union

import logging

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame with error handling.

    Why this exists:
    - Centralizes CSV loading logic and error handling.
    - Provides clear, actionable errors for common issues (missing file, empty data).
    - Ensures consistent logging for data loading across the project.
    """

    logger.info(f"Loading CSV from: {filepath}")

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"CSV not found at: {filepath}. Check data ingestion or file paths."
        )
    if not filepath.is_file():
        raise IsADirectoryError(
            f"Expected a file path, got directory: {filepath}"
        )

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(f"CSV loaded but contains 0 rows: {filepath}")

    return df


def save_csv(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    """
    Save DataFrame as CSV.

    Matches main.py usage:
      save_csv(df_clean, clean_path)
      save_csv(df_pred, pred_path, index=True)
    """

    logger.info(f"Saving CSV to: {filepath}")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=index)


def save_model(model, filepath: Path) -> None:
    """
    Save a scikit-learn model (or Pipeline) to disk using joblib.
    """

    logger.info(f"Saving model to: {filepath}")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Load a scikit-learn model (or Pipeline) from disk using joblib.
    """
    
    logger.info(f"Loading model from: {filepath}")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Model file not found: {filepath}. Train and save the model first (run `python -m src.main`)."
        )

    return joblib.load(filepath)


def save_json(obj: Any, filepath: Path) -> None:
    """
    Save a JSON-serializable object to disk.
    """
    
    logger.info(f"Saving JSON to: {filepath}")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_exp(x: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
    """
    Exponentiate values safely.
    For this project, target is log(charges), so exp brings it back to
    dollar space.

    Handles:
      - pandas Series
      - numpy arrays
      - scalars
    """
    if isinstance(x, pd.Series):
        return x.apply(lambda v: math.exp(v))
    if isinstance(x, np.ndarray):
        return np.exp(x)
    return math.exp(float(x))
