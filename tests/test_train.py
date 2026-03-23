import pandas as pd

from src.features import get_feature_preprocessor
from src.train import train_model


def test_train_model_returns_fitted_model():
    X = pd.DataFrame(
        {
            "num_feature": [0.0, 1.0, 2.0, 3.0],
            "cat_feature": ["A", "B", "A", "B"],
        }
    )
    y = pd.Series([0.0, 1.0, 1.0, 0.0], name="target")

    preprocessor = get_feature_preprocessor(
        numeric_cols=["num_feature"],
        categorical_cols=["cat_feature"],
    )

    model = train_model(X, y, preprocessor)

    assert hasattr(model, "predict")
    assert len(model.predict(X)) == len(X)
