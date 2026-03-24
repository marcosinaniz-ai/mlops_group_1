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
from typing import Any

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

import wandb

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.utils import load_csv, save_csv, save_model
from src.validate import validate_dataframe

SETTINGS = {
    "paths": {
        "raw_data": "data/raw/insurance.csv",
        "processed_clean": "data/processed/clean.csv",
        "model": "models/linreg_insurance.joblib",
        "inference": "data/inference/insurance_inference.csv",
        "predictions": "reports/predictions.csv",
        "reports_dir": "reports",
    },
    # Notebook target is log(charges)
    "target_column": "log_charges",
    "test_size": 0.2,
    "random_state": 42,
    "features": {
        "numeric": ["age", "bmi", "children"],
        "categorical": ["sex", "smoker", "region"],
    },
    "wandb": {
        "enabled": True,
        "project": "insurance-regression",
        "name": "",
        "job_type": "training-pipeline",
        "group": "",
        "notes": "",
        "tags": ["insurance", "regression", "sklearn"],
        "log_processed_data": True,
        "log_predictions": True,
        "log_predictions_table": True,
        "predictions_table_rows": 50,
        "log_evaluation_plots": True,
        "model_artifact_name": "linreg-insurance-model",
    },
}


def _wandb_is_enabled(settings: dict[str, Any]) -> bool:
    wandb_cfg = settings.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return False
    return bool(wandb_cfg.get("enabled", False))


def _wandb_get_str(settings: dict[str, Any], key: str, default: str = "") -> str:
    wandb_cfg = settings.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return str(value).strip() if value is not None else default


def _wandb_get_bool(settings: dict[str, Any], key: str, default: bool = False) -> bool:
    wandb_cfg = settings.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    return bool(wandb_cfg.get(key, default))


def _wandb_get_int(settings: dict[str, Any], key: str, default: int = 0) -> int:
    wandb_cfg = settings.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    try:
        return int(value)
    except Exception:
        return default


def _wandb_get_list(settings: dict[str, Any], key: str) -> list[str]:
    wandb_cfg = settings.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return []

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


def _log_evaluation_to_wandb(eval_artifacts: dict[str, Any]) -> None:
    """
    Logs metrics and plots returned by evaluate_model().
    Expected structure:
    {
        "metrics": {...},
        "plots": {...}
    }
    """
    metrics = eval_artifacts.get("metrics", {})
    plots = eval_artifacts.get("plots", {})

    if isinstance(metrics, dict):
        payload: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                payload[f"metrics/test/{key}"] = float(value)
            except Exception:
                continue
        if payload:
            wandb.log(payload)

    if isinstance(plots, dict):
        image_payload = {}
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
    print("[main.main] Starting end-to-end insurance prediction pipeline")

    project_root = Path(__file__).resolve().parents[1]

    # Optional .env support (for WANDB_API_KEY, etc.)
    load_dotenv(dotenv_path=project_root / ".env", override=False)

    # Ensure standard dirs
    for d in ["data/raw", "data/processed", "data/inference", "models", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    wandb_run = None

    # -----------------------------
    # Initialize W&B
    # -----------------------------
    if _wandb_is_enabled(SETTINGS):
        wandb_project = _wandb_get_str(SETTINGS, "project")
        if not wandb_project:
            raise ValueError(
                "SETTINGS['wandb']['project'] must be a non-empty string when wandb is enabled."
            )

        wandb_name = _wandb_get_str(SETTINGS, "name")
        wandb_job_type = _wandb_get_str(
            SETTINGS, "job_type", default="training-pipeline"
        )
        wandb_group = _wandb_get_str(SETTINGS, "group")
        wandb_notes = _wandb_get_str(SETTINGS, "notes")
        wandb_tags = _wandb_get_list(SETTINGS, "tags")

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name if wandb_name else None,
            job_type=wandb_job_type,
            group=wandb_group if wandb_group else None,
            notes=wandb_notes if wandb_notes else None,
            tags=wandb_tags if wandb_tags else None,
            config=SETTINGS,
        )

        wandb_run.summary["entrypoint"] = "python -m src.main"
        wandb_run.summary["target_column"] = SETTINGS["target_column"]
        wandb_run.summary["model_artifact_path"] = SETTINGS["paths"]["model"]

        print(
            f"[main.main] Initialized W&B run: project={wandb_project}, "
            f"name={wandb_run.name}, job_type={wandb_job_type}"
        )
    else:
        print("[main.main] W&B disabled, continuing without experiment tracking")

    try:
        # Step 1: Load
        raw_path = Path(SETTINGS["paths"]["raw_data"])
        df_raw = load_raw_data(raw_path)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/raw_rows": int(df_raw.shape[0]),
                    "data/raw_cols": int(df_raw.shape[1]),
                }
            )

        # Step 2: Clean (adds log_charges and drops charges)
        target_column = SETTINGS["target_column"]
        df_clean = clean_dataframe(df_raw, target_column=target_column)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/clean_rows": int(df_clean.shape[0]),
                    "data/clean_cols": int(df_clean.shape[1]),
                }
            )

        # Step 3: Save clean
        clean_path = Path(SETTINGS["paths"]["processed_clean"])
        save_csv(df_clean, clean_path)

        # Step 4: Validate
        required_cols = (
            [target_column]
            + SETTINGS["features"]["numeric"]
            + SETTINGS["features"]["categorical"]
        )
        validate_dataframe(
            df_clean,
            required_columns=required_cols,
            target_column=target_column,
        )

        # Step 5: Split
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "data/train_rows": int(X_train.shape[0]),
                    "data/test_rows": int(X_test.shape[0]),
                    "split/test_size": float(SETTINGS["test_size"]),
                    "split/random_state": int(SETTINGS["random_state"]),
                }
            )

        # Step 6: Feature recipe
        preprocessor = get_feature_preprocessor(
            numeric_cols=SETTINGS["features"]["numeric"],
            categorical_cols=SETTINGS["features"]["categorical"],
        )

        # Step 7: Train
        from src.train import train_model

        model = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
        )

        # Step 8: Save model
        model_path = Path(SETTINGS["paths"]["model"])
        save_model(model, model_path)

        # Step 9: Evaluate + save reports
        reports_dir = Path(SETTINGS["paths"]["reports_dir"])
        eval_artifacts = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            reports_dir=reports_dir,
        )

        if wandb_run is not None:
            metrics = eval_artifacts.get("metrics", {})
            if isinstance(metrics, dict):
                try:
                    # Put key headline metrics in run summary
                    wandb_run.summary["r2_log"] = float(metrics["r2_log"])
                    wandb_run.summary["adj_r2_log"] = float(metrics["adj_r2_log"])
                    wandb_run.summary["mae_dollars"] = float(metrics["mae_dollars"])
                    wandb_run.summary["rmse_dollars"] = float(metrics["rmse_dollars"])
                except Exception:
                    pass

            _log_evaluation_to_wandb(eval_artifacts)

        # Log model artifact
        if wandb_run is not None:
            model_artifact_name = _wandb_get_str(
                SETTINGS, "model_artifact_name", default="model"
            )

            model_artifact = wandb.Artifact(
                name=model_artifact_name,
                type="model",
                description="Trained scikit-learn insurance regression pipeline",
            )
            model_artifact.add_file(str(model_path))
            wandb.log_artifact(model_artifact)

            if _wandb_get_bool(SETTINGS, "log_processed_data", default=False):
                data_artifact = wandb.Artifact(
                    name=f"{model_artifact_name}-processed-data",
                    type="dataset",
                    description="Processed training dataset written by the insurance pipeline",
                )
                data_artifact.add_file(str(clean_path))
                wandb.log_artifact(data_artifact)

        # Step 10: Inference on inference data + save predictions
        infer_path = Path(SETTINGS["paths"]["inference"])
        df_infer = load_csv(infer_path)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/inference_rows": int(df_infer.shape[0]),
                    "data/inference_cols": int(df_infer.shape[1]),
                }
            )

        df_pred = run_inference(model=model, X_infer=df_infer)
        pred_path = Path(SETTINGS["paths"]["predictions"])
        save_csv(df_pred, pred_path, index=True)

        if wandb_run is not None and _wandb_get_bool(
            SETTINGS, "log_predictions_table", default=False
        ):
            n_rows = _wandb_get_int(SETTINGS, "predictions_table_rows", default=50)
            preview_df = df_pred.head(n_rows)
            wandb.log({"tables/predictions_preview": wandb.Table(dataframe=preview_df)})

        if wandb_run is not None and _wandb_get_bool(
            SETTINGS, "log_predictions", default=False
        ):
            model_artifact_name = _wandb_get_str(
                SETTINGS, "model_artifact_name", default="model"
            )
            pred_artifact = wandb.Artifact(
                name=f"{model_artifact_name}-predictions",
                type="predictions",
                description="Inference outputs written by the insurance pipeline",
            )
            pred_artifact.add_file(str(pred_path))
            wandb.log_artifact(pred_artifact)

        print("[main.main] Pipeline complete")
        print(f"[main.main] Saved model: {model_path}")
        print(f"[main.main] Saved predictions: {pred_path}")
        print(f"[main.main] Reports directory: {reports_dir.resolve()}")

    except Exception:
        if wandb_run is not None:
            wandb.finish(exit_code=1)
        raise

    finally:
        if wandb_run is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()