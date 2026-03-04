from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # important for headless CI environments

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.train import train_model
from src.evaluate import evaluate_model


def make_toy_data(n=40):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({
        "age": rng.integers(18, 65, size=n),
        "sex": rng.choice(["male", "female"], size=n),
    })
    y = pd.Series(np.log1p(rng.normal(12000, 2500, size=n).clip(min=1000)), name="charges_log")
    return X, y


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["age"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"]),
        ]
    )


def test_evaluate_model_creates_metrics_and_plots(tmp_path: Path):
    X, y = make_toy_data()
    pre = make_preprocessor()

    # train on first part, test on second part
    X_train, y_train = X.iloc[:30], y.iloc[:30]
    X_test, y_test = X.iloc[30:], y.iloc[30:]

    model = train_model(X_train=X_train, y_train=y_train, preprocessor=pre)

    reports_dir = tmp_path / "reports"
    artifacts = evaluate_model(model=model, X_test=X_test, y_test=y_test, reports_dir=reports_dir)

    # return structure
    assert "metrics" in artifacts
    assert "plots" in artifacts

    # metric fields exist (per your implementation)
    metrics = artifacts["metrics"]
    for k in ["r2_log", "adj_r2_log", "mae_log", "rmse_log", "mae_dollars", "rmse_dollars", "n_test", "p_transformed"]:
        assert k in metrics

    # files exist
    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "coefficients.png").exists()
    assert (reports_dir / "pred_vs_actual.png").exists()
    assert (reports_dir / "residuals.png").exists()

    # metrics.json is valid JSON
    with open(reports_dir / "metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metrics" in data