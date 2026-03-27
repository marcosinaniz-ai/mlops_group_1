"""
Educational Goal:
- Turn the trained ML pipeline into a minimal web product using FastAPI.
- Use Pydantic to enforce a strict data contract.
- Reuse existing pipeline modules to avoid training-serving skew.
- Implement scalable observability:
    Layer 1 -> system logs
    Layer 2 -> async batched ML logs to W&B

Key principle:
- NO NEW ML LOGIC IN THIS FILE.
- This file only handles HTTP, schema validation, startup loading,
  observability logging, and routing.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import time
import uuid
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import wandb
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, create_model

from src.clean_data import clean_dataframe
from src.infer import run_inference
from src.main import load_config, require_section
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

load_dotenv()


# -------------------------------------------------------------------
# 1) Small local helpers
# -------------------------------------------------------------------
def require_str(cfg: Dict[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config key '{key}' must be a non-empty string")
    return value.strip()


def resolve_repo_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def _require_list(cfg: Dict[str, Any], key: str) -> List[Any]:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key}")
    value = cfg.get(key)
    if not isinstance(value, list):
        raise ValueError(
            f"Config key '{key}' must be a list, got {type(value).__name__}"
        )
    return value


def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _configured_feature_columns(cfg: Dict[str, Any]) -> List[str]:
    features_cfg = require_section(cfg, "features")
    numeric_cols = [str(c) for c in _require_list(features_cfg, "numerical")]
    categorical_cols = [str(c) for c in _require_list(features_cfg, "categorical")]
    return _dedupe_preserve_order(numeric_cols + categorical_cols)


def _build_request_record_model() -> type[BaseModel]:
    """
    Build a strict Pydantic request model from config.yaml.

    For inference, only feature columns are required.
    """
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config.yaml")

    features_cfg = require_section(cfg, "features")
    numeric_cols = [str(c) for c in _require_list(features_cfg, "numerical")]
    categorical_cols = [str(c) for c in _require_list(features_cfg, "categorical")]

    field_definitions: Dict[str, tuple[Any, Any]] = {}

    for col in numeric_cols:
        field_definitions[col] = (float, ...)

    for col in categorical_cols:
        field_definitions[col] = (str, ...)

    example: Dict[str, Any] = {}
    for col in numeric_cols:
        if col == "children":
            example[col] = 2
        elif col == "age":
            example[col] = 35
        elif col == "bmi":
            example[col] = 39.71
        else:
            example[col] = 1.0

    for col in categorical_cols:
        if col == "sex":
            example[col] = "male"
        elif col == "smoker":
            example[col] = "no"
        elif col == "region":
            example[col] = "northeast"
        else:
            example[col] = "value"

    model = create_model(
        "InsuranceRecord",
        __config__=ConfigDict(
            extra="forbid",
            json_schema_extra={"example": example},
        ),
        **field_definitions,
    )
    return model


InsuranceRecord = _build_request_record_model()


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[InsuranceRecord]


class PredictionItem(BaseModel):
    row_id: int
    prediction: float


class PredictResponse(BaseModel):
    model_version: str
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


# -------------------------------------------------------------------
# 2) Lifespan: load shared resources once at API startup
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        project_root = Path(__file__).resolve().parents[1]
        app.state.global_config = load_config(project_root / "config.yaml")

        paths_cfg = require_section(app.state.global_config, "paths")
        wandb_cfg = require_section(app.state.global_config, "wandb")

        model_source = os.getenv("MODEL_SOURCE", "local").lower()

        if model_source == "wandb":
            logger.info("MODEL_SOURCE=wandb -> fetching model from W&B artifact")

            wandb_project = require_str(wandb_cfg, "project")
            artifact_name = require_str(wandb_cfg, "model_artifact_name")
            artifact_alias = os.getenv("WANDB_MODEL_ALIAS", "latest")
            wandb_entity = os.getenv("WANDB_ENTITY")
            wandb_api_key = os.getenv("WANDB_API_KEY")

            if not wandb_entity:
                raise ValueError("WANDB_ENTITY environment variable is required")
            if not wandb_api_key:
                raise ValueError("WANDB_API_KEY environment variable is required")

            artifact_path = (
                f"{wandb_entity}/{wandb_project}/{artifact_name}:{artifact_alias}"
            )

            wandb.login(key=wandb_api_key, relogin=True)
            api = wandb.Api()
            artifact = api.artifact(artifact_path)
            artifact_dir = artifact.download()
            model_filename = require_str(paths_cfg, "model").split("/")[-1]
            model_path = Path(artifact_dir) / model_filename

            logger.info("Downloaded model from W&B: %s", artifact_path)

            if not model_path.exists():
                logger.error("Model file missing inside downloaded artifact at %s", model_path)
                app.state.model_pipeline = None
                app.state.model_version = "missing"
            else:
                app.state.model_pipeline = joblib.load(model_path)
                app.state.model_version = artifact_path
                logger.info("Startup complete, model loaded from W&B artifact %s", artifact_path)

        else:
            logger.info("MODEL_SOURCE=local -> using local model artifact")

            model_path = resolve_repo_path(
                project_root,
                require_str(paths_cfg, "model"),
            )

            if not model_path.exists():
                logger.error("Model file missing at %s", model_path)
                app.state.model_pipeline = None
                app.state.model_version = "missing"
            else:
                app.state.model_pipeline = joblib.load(model_path)
                app.state.model_version = model_path.name
                logger.info("Startup complete, model loaded from %s", model_path)

    except Exception as e:
        logger.exception("Startup failed: %s", str(e))
        app.state.global_config = {}
        app.state.model_pipeline = None
        app.state.model_version = "startup_error"

    yield
    logger.info("API shutdown complete")


app = FastAPI(
    title="Insurance Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# 3) Observability Architecture (Layer 1 & Layer 2)
# -------------------------------------------------------------------

# --- Layer 1: System Monitoring Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    response = await call_next(request)

    latency = time.time() - start_time
    model_version = getattr(app.state, "model_version", "unloaded")

    logger.info(
        "correlation_id=%s path=%s method=%s status=%s latency_s=%.4f model_version=%s",
        correlation_id,
        request.url.path,
        request.method,
        response.status_code,
        latency,
        model_version,
    )

    response.headers["X-Correlation-ID"] = correlation_id
    return response


# --- Layer 2: ML Monitoring Buffer ---
LOG_BUFFER: List[Dict[str, Any]] = []
BUFFER_LOCK = Lock()
BATCH_SIZE = 10


def flush_logs_to_wandb(batch_data: List[Dict[str, Any]], project_name: str) -> None:
    """Ephemeral W&B run to securely log a batch of inference rows as a Table."""
    if not batch_data:
        return

    if os.getenv("WANDB_MODE", "").lower() == "disabled":
        logger.info("Skipping W&B flush because WANDB_MODE=disabled")
        return

    try:
        wandb_entity = os.getenv("WANDB_ENTITY")
        run = wandb.init(
            entity=wandb_entity if wandb_entity else None,
            project=project_name,
            job_type="inference-batch",
            reinit=True,
        )

        feature_keys = list(batch_data[0]["features"].keys())
        columns = [
            "correlation_id",
            "req_id",
            "timestamp",
            "path",
            "status_code",
            "model_version",
            "latency_s",
            "prediction",
        ] + feature_keys

        table = wandb.Table(columns=columns)

        for item in batch_data:
            row = [
                item["correlation_id"],
                item["req_id"],
                item["timestamp"],
                item["path"],
                item["status_code"],
                item["model_version"],
                item["latency_s"],
                item["prediction"],
            ] + [item["features"].get(k) for k in feature_keys]

            table.add_data(*row)

        run.log({"inference_logs": table})
        run.finish()
        logger.info("Flushed %s ML logs to W&B", len(batch_data))

    except Exception as e:
        logger.error("Failed to flush logs to W&B: %s", e)


# -------------------------------------------------------------------
# 4) Endpoints
# -------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Use /health or /docs to test the API"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_loaded = getattr(app.state, "model_pipeline", None) is not None
    model_version = getattr(app.state, "model_version", "unloaded")

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "model_not_loaded",
                "model_loaded": False,
                "model_version": model_version,
            },
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictResponse:
    model_pipeline = getattr(app.state, "model_pipeline", None)
    global_config = getattr(app.state, "global_config", {})
    model_version = getattr(app.state, "model_version", "unloaded")

    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check startup logs, model path, W&B credentials, artifact alias, and model availability.",
        )

    try:
        inference_start_time = time.time()
        correlation_id = getattr(
            request.state, "correlation_id", "missing_correlation_id"
        )

        records_dicts = [r.model_dump() for r in req.records]
        df_raw = pd.DataFrame(records_dicts)

        features_cfg = require_section(global_config, "features")
        wandb_cfg = require_section(global_config, "wandb")

        numeric_columns = set(str(c) for c in _require_list(features_cfg, "numerical"))
        categorical_columns = set(str(c) for c in _require_list(features_cfg, "categorical"))
        required_columns = set(_configured_feature_columns(global_config))

        df_clean = clean_dataframe(
            df_raw=df_raw,
            target_column=None,
            required_columns=required_columns,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )

        validate_dataframe(
            df=df_clean,
            required_columns=_configured_feature_columns(global_config),
            target_column=None,
        )

        df_pred = run_inference(model=model_pipeline, X_infer=df_clean)

        inference_latency = time.time() - inference_start_time
        current_time = time.time()

        preds: List[PredictionItem] = []

        with BUFFER_LOCK:
            for i in range(len(df_pred)):
                pred_val = float(df_pred.iloc[i]["prediction"])

                preds.append(
                    PredictionItem(
                        row_id=i,
                        prediction=pred_val,
                    )
                )

                LOG_BUFFER.append(
                    {
                        "correlation_id": correlation_id,
                        "req_id": i,
                        "timestamp": current_time,
                        "path": "/predict",
                        "status_code": 200,
                        "model_version": model_version,
                        "latency_s": inference_latency,
                        "prediction": pred_val,
                        "features": records_dicts[i],
                    }
                )

            if len(LOG_BUFFER) >= BATCH_SIZE:
                batch_copy = LOG_BUFFER.copy()
                LOG_BUFFER.clear()

                wandb_project = require_str(wandb_cfg, "project")
                background_tasks.add_task(
                    flush_logs_to_wandb,
                    batch_copy,
                    wandb_project,
                )

        return PredictResponse(
            model_version=model_version,
            predictions=preds,
        )

    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error") from e