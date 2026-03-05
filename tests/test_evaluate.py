import math
from pathlib import Path

import pandas as pd

from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.train import train_model


def _build_preprocessor():
    return get_feature_preprocessor(
        quantile_bin_cols=[],
        categorical_onehot_cols=["cat_feature"],
        numeric_passthrough_cols=["num_feature"],
        n_bins=3,
    )


def test_evaluate_model_returns_report_dict_and_writes_artifacts(tmp_path):
    X = pd.DataFrame(
        {
            "num_feature": [0.0, 1.0, 2.0, 3.0],
            "cat_feature": ["A", "B", "A", "B"],
        }
    )
    y = pd.Series([0.0, 1.0, 1.0, 0.0], name="target")

    model = train_model(X, y, _build_preprocessor())
    report = evaluate_model(model, X, y, tmp_path)

    # 1) Return type / structure
    assert isinstance(report, dict)
    assert isinstance(report.get("metrics"), dict)

    # 2) At least one numeric, non-NaN metric exists
    metrics = report["metrics"]
    assert any(
        isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))
        for v in metrics.values()
    ), f"No usable numeric metric found in: {metrics}"

    # 3) metrics.json must be written
    assert (tmp_path / "metrics.json").exists()

    # 4) If plot paths are returned, those files should exist
    plots = report.get("plots")
    if isinstance(plots, dict):
        for plot_path in plots.values():
            if isinstance(plot_path, str) and plot_path.endswith(".png"):
                # handle absolute paths or relative paths
                p = Path(plot_path)
                if p.is_absolute():
                    assert p.exists()
                else:
                    assert (tmp_path / p.name).exists()