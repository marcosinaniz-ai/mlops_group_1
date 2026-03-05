"""
TODO: Replace print statements with standard library logging in a later session
TODO: Any temp or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

from pathlib import Path
import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to a CSV file.
    Outputs:
    - df: Loaded pandas DataFrame.
    Why this contract matters for reliable ML delivery:
    - Centralizing CSV reads makes pipelines more reproducible and easier to
    debug when file formats change.
    """

    # TODO: replace with logging later
    print(f"[utils.load_csv] Loading CSV from: {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {filepath}."
            "Check data ingestion or file paths."
        )

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(
            f"CSV loaded but contains 0 rows: {filepath}"
        )

    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save.
    - filepath: Path where the CSV should be written.
    Outputs:
    - None (writes file as a side-effect).
    Why this contract matters for reliable ML delivery:
    - Standardizing output locations makes downstream automation and reviews
    predictable (CI, reports, deployments).
    """

    # TODO: replace with logging later
    print(f"[utils.save_csv] Saving CSV to: {filepath}")

    # Ensure the directory exists before writing
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: Any fitted scikit-learn compatible object (here: Pipeline).
    - filepath: Path to write the serialized model artifact.
    Outputs:
    - None (writes file as a side-effect).
    Why this contract matters for reliable ML delivery:
    - Persisting the exact trained Pipeline enables consistent inference and
    supports promotion across environments.
    """
    # TODO: replace with logging later
    print(f"[utils.save_model] Saving model to: {filepath}")

    # Ensure the directory exists before writing
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to a serialized model artifact.
    Outputs:
    - model: Loaded model object.
    Why this contract matters for reliable ML delivery:
    - Inference systems must load the same artifact format across training and
    serving to prevent drift in behavior.
    """
    # TODO: replace with logging later
    print(f"[utils.load_model] Loading model from: {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Model file not found: {filepath}. "
            "Train and save the model first (run `python -m src.main`)."
        )

    return joblib.load(filepath)
