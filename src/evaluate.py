"""
Module: Evaluation
------------------
Reproduces notebook-style evaluation:
- R^2, Adj R^2 on log space
- MAE/RMSE in dollars by exponentiating y_true and y_pred
- Plots:
  - coefficients barh
  - predicted vs actual (log)
  - residuals vs predicted + residual distribution
Artifacts saved into reports/
"""

from __future__ import annotations

import logging

import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import save_json, safe_exp

logger = logging.getLogger(__name__)


def _plot_coefficients(model, reports_dir: Path, top_k: int = 30) -> Path:
    # Get names from ColumnTransformer (Pipeline step "preprocess")
    preprocess = model.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    coefs = model.named_steps["model"].coef_

    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    coef_df["Feature"] = (
        coef_df["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    coef_df = coef_df.sort_values(by="Coefficient", key=lambda s: s.abs(), ascending=False).head(top_k)

    colors = ["red" if x > 0 else "blue" for x in coef_df["Coefficient"]]

    plt.figure(figsize=(8, max(4, 0.25 * len(coef_df))))
    plt.barh(coef_df["Feature"][::-1], coef_df["Coefficient"][::-1], color=colors[::-1])
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Top feature coefficients (LinearRegression)")
    plt.tight_layout()

    outpath = reports_dir / "coefficients.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


def _plot_pred_vs_actual(y_test: pd.Series, y_pred: np.ndarray, reports_dir: Path) -> Path:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)

    min_val = float(min(np.min(y_test), np.min(y_pred)))
    max_val = float(max(np.max(y_test), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect prediction")

    plt.xlabel("Actual log(charges)")
    plt.ylabel("Predicted log(charges)")
    plt.title("Predicted vs Actual log(charges)")
    plt.legend()
    plt.tight_layout()

    outpath = reports_dir / "pred_vs_actual.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


def _plot_residuals(y_test: pd.Series, y_pred: np.ndarray, reports_dir: Path) -> Path:

    logger.info("Plotting residuals vs predicted and residual distribution")

    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted log(charges)")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted log(charges)")

    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_xlabel("Residuals")
    axes[1].set_title("Distribution of Residuals")

    plt.tight_layout()
    outpath = reports_dir / "residuals.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    reports_dir: Path,
) -> Dict[str, Any]:

    logger.info("Evaluating model and saving report artifacts")

    reports_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    # For adjusted R^2, compute p from transformed feature count
    X_test_transf = model.named_steps["preprocess"].transform(X_test)
    n = len(y_test)
    p = X_test_transf.shape[1]

    r2 = float(r2_score(y_test, y_pred))
    adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if (n - p - 1) != 0 else float("nan")

    # Dollar-space metrics (exp because target is log(charges))
    y_test_dollars = safe_exp(y_test)
    y_pred_dollars = safe_exp(y_pred)

    mae_dollars = float(mean_absolute_error(y_test_dollars, y_pred_dollars))
    rmse_dollars = float(math.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars)))

    # Also compute MAE/RMSE in log space (sometimes useful)
    mae_log = float(mean_absolute_error(y_test, y_pred))
    rmse_log = float(math.sqrt(mean_squared_error(y_test, y_pred)))

    artifacts = {
        "metrics": {
            "r2_log": r2,
            "adj_r2_log": adj_r2,
            "mae_log": mae_log,
            "rmse_log": rmse_log,
            "mae_dollars": mae_dollars,
            "rmse_dollars": rmse_dollars,
            "n_test": int(n),
            "p_transformed": int(p),
        },
        "plots": {},
    }

    # Plots
    artifacts["plots"]["coefficients_png"] = str(_plot_coefficients(model, reports_dir))
    artifacts["plots"]["pred_vs_actual_png"] = str(_plot_pred_vs_actual(y_test, y_pred, reports_dir))
    artifacts["plots"]["residuals_png"] = str(_plot_residuals(y_test, y_pred, reports_dir))

    # Save metrics JSON
    save_json(artifacts, reports_dir / "metrics.json")

    return artifacts
