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
from typing import Any, Dict
import logging

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import yaml
import wandb

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
    """Load YAML config file and validate it loads into a dictionary."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must load into a dictionary")

    return config


def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Require a top-level config section to exist and be a mapping."""
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"config.yaml must contain a top-level '{section}' mapping")
    return value


def optional_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Return an optional top-level config section as a mapping or {}."""
    value = cfg.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config.yaml section '{section}' must be a mapping")
    return value


# -----------------------------
# W&B helpers
# -----------------------------
def _wandb_is_enabled(wandb_cfg: Dict[str, Any]) -> bool:
    return bool(wandb_cfg.get("enabled", False))


def _wandb_get_str(wandb_cfg: Dict[str, Any], key: str, default: str = "") -> str:
    value = wandb_cfg.get(key, default)
    return str(value).strip() if value is not None else default


def _wandb_get_bool(wandb_cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(wandb_cfg.get(key, default))


def _wandb_get_int(wandb_cfg: Dict[str, Any], key: str, default: int = 0) -> int:
    value = wandb_cfg.get(key, default)
    try:
        return int(value)
    except Exception:
        return default


def _wandb_get_list(wandb_cfg: Dict[str, Any], key: str) -> list[str]:
    value = wandb_cfg.get(key, [])
    if not isinstance(value, list):
        return []

    out: list[str] = []
    for item in value:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _log_evaluation_to_wandb(eval_artifacts: Dict[str, Any]) -> None:
    """
    Log metrics and plots returned by evaluate_model().

    Expected structure:
    {
        "metrics": {...},
        "plots": {...}
    }
    """
    metrics = eval_artifacts.get("metrics", {})
    plots = eval_artifacts.get("plots", {})

    if isinstance(metrics, dict):
        metric_payload: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                metric_payload[f"metrics/test/{key}"] = float(value)
            except Exception:
                continue
        if metric_payload:
            wandb.log(metric_payload)

    if isinstance(plots, dict):
        image_payload: dict[str, Any] = {}
        for key, value in plots.items():
            try:
                path = Path(value)
                if path.exists():
                    image_payload[f"plots/{key}"] = wandb.Image(str(path))
            except Exception:
                continue
        if image_payload:
            wandb.log(image_payload)


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
    wandb_config = optional_section(config, "wandb")

    # Optional .env support (for WANDB_API_KEY, etc.)
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / ".env", override=False)

    # -----------------------------
    # Logging
    # -----------------------------
    configure_logging(
        log_level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("file", "reports/pipeline.log"),
    )

    logger.info("Starting end-to-end insurance prediction pipeline")

    # Ensure standard dirs
    for d in ["data/raw", "data/processed", "data/inference", "models", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    wandb_run = None

    # -----------------------------
    # Initialize W&B
    # -----------------------------
    if _wandb_is_enabled(wandb_config):
        wandb_project = _wandb_get_str(wandb_config, "project")
        if not wandb_project:
            raise ValueError(
                "config.yaml -> wandb.project must be a non-empty string when wandb.enabled is true."
            )

        wandb_name = _wandb_get_str(wandb_config, "name")
        wandb_job_type = _wandb_get_str(
            wandb_config, "job_type", default="training-pipeline"
        )
        wandb_group = _wandb_get_str(wandb_config, "group")
        wandb_notes = _wandb_get_str(wandb_config, "notes")
        wandb_tags = _wandb_get_list(wandb_config, "tags")

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name or None,
            job_type=wandb_job_type,
            group=wandb_group or None,
            notes=wandb_notes or None,
            tags=wandb_tags or None,
            config=config,
        )

        wandb_run.summary["entrypoint"] = "python -m src.main"
        wandb_run.summary["target_column"] = schema_config["target"]
        wandb_run.summary["model_artifact_path"] = paths_config["model"]

        logger.info(
            "Initialized W&B run: project=%s, name=%s, job_type=%s",
            wandb_project,
            wandb_run.name,
            wandb_job_type,
        )
    else:
        logger.info("W&B disabled, continuing without experiment tracking")

    try:
        # Step 1: Load
        raw_path = Path(paths_config["raw"])
        df_raw = load_raw_data(raw_path)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/raw_rows": int(df_raw.shape[0]),
                    "data/raw_cols": int(df_raw.shape[1]),
                }
            )

        # Step 2: Clean
        target_column = schema_config["target"]
        required_cols_set = set(schema_config["required_columns"])
        categorical_cols_set = set(features_config["categorical"])
        numeric_cols_set = set(features_config["numerical"])

        df_clean = clean_dataframe(
            df_raw,
            target_column=target_column,
            required_columns=required_cols_set,
            categorical_columns=categorical_cols_set,
            numeric_columns=numeric_cols_set,
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "data/clean_rows": int(df_clean.shape[0]),
                    "data/clean_cols": int(df_clean.shape[1]),
                }
            )

        # Step 3: Save clean
        clean_path = Path(paths_config["processed"])
        save_csv(df_clean, clean_path)

        # Step 4: Validate
        required_cols = (
            features_config["numerical"]
            + features_config["categorical"]
            + [target_column]
        )

        validate_dataframe(
            df_clean,
            required_columns=required_cols,
            target_column=target_column,
        )

        # Step 5: Split
        logger.info("Splitting data into features (X) and target (y)")
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=train_config["test_size"],
            random_state=train_config["seed"],
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "data/train_rows": int(X_train.shape[0]),
                    "data/test_rows": int(X_test.shape[0]),
                    "split/test_size": float(train_config["test_size"]),
                    "split/random_state": int(train_config["seed"]),
                }
            )

        # Step 6: Feature recipe
        preprocessor = get_feature_preprocessor(
            numeric_cols=features_config["numerical"],
            categorical_cols=features_config["categorical"],
        )

        # Step 7: Train
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
        )

        # Step 8: Save model
        model_path = Path(paths_config["model"])
        save_model(model, model_path)

        # Step 9: Evaluate + save reports
        reports_dir = Path(paths_config["reports_dir"])
        eval_artifacts = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            reports_dir=reports_dir,
        )

        if wandb_run is not None:
            metrics = eval_artifacts.get("metrics", {})
            if isinstance(metrics, dict):
                for summary_key in ["r2_log", "adj_r2_log", "mae_dollars", "rmse_dollars"]:
                    if summary_key in metrics:
                        try:
                            wandb_run.summary[summary_key] = float(metrics[summary_key])
                        except Exception:
                            pass

            _log_evaluation_to_wandb(eval_artifacts)

        # Log model artifact
        if wandb_run is not None:
            model_artifact_name = _wandb_get_str(
                wandb_config, "model_artifact_name", default="model"
            )

            model_artifact = wandb.Artifact(
                name=model_artifact_name,
                type="model",
                description="Trained scikit-learn insurance regression pipeline",
            )
            model_artifact.add_file(str(model_path))
            wandb.log_artifact(model_artifact)

            if _wandb_get_bool(wandb_config, "log_processed_data", default=False):
                data_artifact = wandb.Artifact(
                    name=f"{model_artifact_name}-processed-data",
                    type="dataset",
                    description="Processed training dataset written by the insurance pipeline",
                )
                data_artifact.add_file(str(clean_path))
                wandb.log_artifact(data_artifact)

        # Step 10: Inference on inference data + save predictions
        infer_path = Path(paths_config["inference"])
        df_infer = load_csv(infer_path)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/inference_rows": int(df_infer.shape[0]),
                    "data/inference_cols": int(df_infer.shape[1]),
                }
            )

        df_pred = run_inference(model=model, X_infer=df_infer)
        pred_path = Path(paths_config["predictions"])
        save_csv(df_pred, pred_path, index=True)

        if wandb_run is not None and _wandb_get_bool(
            wandb_config, "log_predictions_table", default=False
        ):
            n_rows = _wandb_get_int(wandb_config, "predictions_table_rows", default=50)
            preview_df = df_pred.head(n_rows)
            wandb.log({"tables/predictions_preview": wandb.Table(dataframe=preview_df)})

        if wandb_run is not None and _wandb_get_bool(
            wandb_config, "log_predictions", default=False
        ):
            model_artifact_name = _wandb_get_str(
                wandb_config, "model_artifact_name", default="model"
            )
            pred_artifact = wandb.Artifact(
                name=f"{model_artifact_name}-predictions",
                type="predictions",
                description="Inference outputs written by the insurance pipeline",
            )
            pred_artifact.add_file(str(pred_path))
            wandb.log_artifact(pred_artifact)

        logger.info("Pipeline complete")
        logger.info("Saved model: %s", model_path)
        logger.info("Saved predictions: %s", pred_path)
        logger.info("Reports directory: %s", reports_dir.resolve())

    except Exception:
        logger.exception("Pipeline failed")
        if wandb_run is not None:
            wandb.finish(exit_code=1)
        raise

    finally:
        if wandb_run is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()