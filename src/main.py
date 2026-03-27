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
from typing import Any, Dict, List, Optional, Tuple
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


def require_str(section: Dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.yaml: '{key}' must be a non-empty string")
    return value.strip()


def require_float(section: Dict[str, Any], key: str) -> float:
    value = section.get(key)
    try:
        return float(value)
    except Exception as e:
        raise ValueError(
            f"config.yaml: '{key}' must be a number. Got '{value}'") from e


def require_int(section: Dict[str, Any], key: str) -> int:
    value = section.get(key)
    try:
        return int(value)
    except Exception as e:
        raise ValueError(
            f"config.yaml: '{key}' must be an integer. Got '{value}'") from e


def require_list(section: Dict[str, Any], key: str) -> List[str]:
    value = section.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"config.yaml: '{key}' must be a list. Got type={type(value)}")
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def resolve_repo_path(project_root: Path, relative_path: str) -> Path:
    """
    Resolve a config path relative to the repo root

    This makes the repo reproducible across machines because we never rely on the current working directory
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("config.yaml: path values must be non-empty strings")
    return project_root / relative_path.strip()


# -----------------------------
# W&B helpers
# -----------------------------
def _wandb_is_enabled(cfg: Dict[str, Any]) -> bool:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return False
    return bool(wandb_cfg.get("enabled", False))


def _wandb_get_str(cfg: Dict[str, Any], key: str, default: str = "") -> str:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return str(value).strip() if value is not None else default


def _wandb_get_bool(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return bool(value)


def _wandb_get_int(cfg: Dict[str, Any], key: str, default: int = 0) -> int:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    try:
        return int(value)
    except Exception:
        return default


def _wandb_get_list(cfg: Dict[str, Any], key: str) -> List[str]:
    """Safely extract a list of strings, stripping whitespace and dropping empty values."""
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return []

    value = wandb_cfg.get(key, [])
    if not isinstance(value, list):
        return []

    out: List[str] = []
    for v in value:
        if v is None:
            continue
        s = str(v).strip()
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
    project_root = Path(__file__).resolve().parents[1]

    # -----------------------------
    # Load and validate config.yaml
    # -----------------------------
    config = load_config(project_root / "config.yaml")

    paths_config = require_section(config, "paths")
    train_config = require_section(config, "train")
    schema_config = require_section(config, "schema")
    features_config = require_section(config, "features")
    logging_config = require_section(config, "logging")

    log_file_path = resolve_repo_path(
        project_root, require_str(paths_config, "log_file"))
    log_level = require_str(logging_config, "level")

    # Optional .env support (for WANDB_API_KEY, etc.)
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / ".env", override=False)

    # -----------------------------
    # Logging
    # -----------------------------
    configure_logging(
        log_level=log_level,
        log_file=log_file_path,
    )

    # -----------------------------
    # Initialize W&B
    # -----------------------------
    wandb_run = None
    if _wandb_is_enabled(config):
        wandb_project = _wandb_get_str(config, "project")
        if not wandb_project:
            raise ValueError(
                "config.yaml -> wandb.project must be a non-empty string when wandb.enabled is true."
            )

        wandb_name = _wandb_get_str(config, "name")
        wandb_job_type = _wandb_get_str(
            config, "job_type", default="training-pipeline"
        )
        wandb_group = _wandb_get_str(config, "group")
        wandb_notes = _wandb_get_str(config, "notes")
        wandb_tags = _wandb_get_list(config, "tags")

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
        logger.info("Starting end-to-end insurance prediction pipeline")

        # Required targets and schemas
        target_column = require_str(schema_config, "target")
        required_cols_clean = require_list(schema_config, "required_columns")
        categorical_cols = require_list(features_config, "categorical")
        numeric_cols = require_list(features_config, "numerical")
        required_cols_val = (
            require_list(features_config, "numerical")
            + require_list(features_config, "categorical")
            + [target_column]
        )

        # Resolve paths
        raw_path = resolve_repo_path(
            project_root, require_str(paths_config, "raw")
        )
        clean_path = resolve_repo_path(
            project_root, require_str(paths_config, "processed")
        )
        model_path = resolve_repo_path(
            project_root, require_str(paths_config, "model")
        )
        reports_dir = resolve_repo_path(
            project_root, require_str(paths_config, "reports_dir")
        )
        infer_path = resolve_repo_path(
            project_root, require_str(paths_config, "inference")
        )
        pred_path = resolve_repo_path(
            project_root, require_str(paths_config, "predictions")
        )

        # Split settings
        test_size = require_float(train_config, "test_size")
        seed = require_int(train_config, "seed")

        # Step 1: Load
        df_raw = load_raw_data(raw_path)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/raw_rows": int(df_raw.shape[0]),
                    "data/raw_cols": int(df_raw.shape[1]),
                }
            )

        # Step 2: Clean
        df_clean = clean_dataframe(
            df_raw,
            target_column=target_column,
            required_columns=set(required_cols_clean),
            categorical_columns=set(categorical_cols),
            numeric_columns=set(numeric_cols),
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "data/clean_rows": int(df_clean.shape[0]),
                    "data/clean_cols": int(df_clean.shape[1]),
                }
            )

        # Step 3: Save clean data
        save_csv(df_clean, clean_path)

        # Step 4: Validate
        validate_dataframe(
            df_clean,
            required_columns=required_cols_val,
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
            test_size=test_size,
            random_state=seed
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
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )

        # Step 7: Train
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
        )

        # Step 8: Save model
        save_model(model, model_path)

        # Step 9: Evaluate + save reports
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
                config, "model_artifact_name", default="model"
            )

            model_artifact = wandb.Artifact(
                name=model_artifact_name,
                type="model",
                description="Trained scikit-learn insurance regression pipeline",
            )
            model_artifact.add_file(str(model_path))
            wandb.log_artifact(model_artifact)

            if _wandb_get_bool(config, "log_processed_data", default=False):
                data_artifact = wandb.Artifact(
                    name=f"{model_artifact_name}-processed-data",
                    type="dataset",
                    description="Processed training dataset written by the insurance pipeline",
                )
                data_artifact.add_file(str(clean_path))
                wandb.log_artifact(data_artifact)

        # Step 10: Inference on inference data + save predictions
        df_infer = load_csv(infer_path)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/inference_rows": int(df_infer.shape[0]),
                    "data/inference_cols": int(df_infer.shape[1]),
                }
            )

        df_pred = run_inference(model=model, X_infer=df_infer)
        save_csv(df_pred, pred_path, index=True)

        if wandb_run is not None and _wandb_get_bool(
            config, "log_predictions_table", default=False
        ):
            n_rows = _wandb_get_int(config, "predictions_table_rows", default=50)
            preview_df = df_pred.head(n_rows)
            wandb.log({"tables/predictions_preview": wandb.Table(dataframe=preview_df)})

        if wandb_run is not None and _wandb_get_bool(
            config, "log_predictions", default=False
        ):
            model_artifact_name = _wandb_get_str(
                config, "model_artifact_name", default="model"
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
