"""
Module: Main Pipeline
---------------------
Orchestrates:
Load -> Clean -> Validate -> Split -> Features -> Train -> Evaluate -> Infer

Run from repo root:
  python -m src.main
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

from sklearn.model_selection import train_test_split
import yaml

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.logger import configure_logging
from src.train import train_model
from src.utils import load_csv, save_csv, save_model
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)

# -----------------------------
# Config loading and validation
# -----------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    '''Load YAML config file and validate it loads into a dictionary'''

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must load into a dictionary")

    return config


def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Enforce a required top-level config section

    Why this exists
    - This produces an actionable error tied to config.yaml structure
    """
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ValueError(
            f"config.yaml must contain a top-level '{section}' mapping")
    return value


def main() -> None:

    # -----------------------------
    # Load and validate config.yaml
    # -----------------------------

    config = load_config(Path("config.yaml"))

    paths_config = require_section(config, "paths")
    train_config = require_section(config, "train")
    schema_config = require_section(config, "schema")
    features_config = require_section(config, "features")
    logging_config = require_section(config, "logging")

    configure_logging(
        log_level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("file", "reports/pipeline.log")
    )

    logger.info("Starting end-to-end insurance prediction pipeline")

    # Ensure standard dirs
    for d in ["data/raw", "data/processed", "data/inference", "models", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    raw_path = Path(paths_config["raw"])
    df_raw = load_raw_data(raw_path)

    # Step 2: Clean (adds log_charges and drops charges)
    target_column = schema_config["target"]
    required_cols = set(schema_config["required_columns"])
    categorical_cols = set(features_config["categorical"])
    numeric_cols = set(features_config["numerical"])

    df_clean = clean_dataframe(
        df_raw,
        target_column=target_column,
        required_columns=required_cols,
        categorical_columns=categorical_cols,
        numeric_columns=numeric_cols
    )

    # Step 3: Save clean
    clean_path = Path(paths_config["processed"])
    save_csv(df_clean, clean_path)

    # Step 4: Validate
    required_cols = (
        features_config["numerical"]
        + features_config["categorical"]
        + [schema_config["target"]])

    validate_dataframe(
        df_clean,
        required_columns=required_cols,
        target_column=target_column
    )

    # Step 5: Split
    logger.info("Splitting data into features (X) and prediction (y)")
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    # Simple split (not stratified; regression)
    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_config["test_size"],
        random_state=train_config["seed"],
    )

    # Step 6: Feature recipe
    preprocessor = get_feature_preprocessor(
        numeric_cols=features_config["numerical"],
        categorical_cols=features_config["categorical"],
    )

    # Step 7: Train
    model = train_model(X_train=X_train, y_train=y_train, preprocessor=preprocessor)

    # Step 8: Save model
    model_path = Path(paths_config["model"])
    save_model(model, model_path)

    # Step 9: Evaluate + save reports
    reports_dir = Path(paths_config["reports_dir"])
    _ = evaluate_model(model=model, X_test=X_test, y_test=y_test, reports_dir=reports_dir)

    # Step 10: Inference on inference data + save predictions
    infer_path = Path(paths_config["inference"])
    df_infer = load_csv(infer_path)

    df_pred = run_inference(model=model, X_infer=df_infer)
    pred_path = Path(paths_config["predictions"])
    save_csv(df_pred, pred_path, index=True)  # keep index to align with test rows

    logger.info("Pipeline complete")
    logger.info(f"Saved model: {model_path}")
    logger.info(f"Saved predictions: {pred_path}")
    logger.info(f"Reports directory: {reports_dir.resolve()}")


if __name__ == "__main__":
    main()
