import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.train import train_model


def make_toy_data(n=30):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "age": rng.integers(18, 65, size=n),
        "sex": rng.choice(["male", "female"], size=n),
    })
    # log(charges)-like target (continuous)
    y = pd.Series(np.log1p(rng.normal(10000, 2000, size=n).clip(min=1000)), name="charges_log")
    return X, y


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["age"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"]),
        ]
    )


def test_train_model_returns_fitted_pipeline():
    X, y = make_toy_data()
    pre = make_preprocessor()

    model = train_model(X_train=X, y_train=y, preprocessor=pre)

    assert isinstance(model, Pipeline)
    assert "preprocess" in model.named_steps
    assert "model" in model.named_steps

    # fitted pipeline should be able to predict
    preds = model.predict(X)
    assert len(preds) == len(X)