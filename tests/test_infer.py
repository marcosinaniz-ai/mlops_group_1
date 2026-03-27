import pandas as pd
import numpy as np

from src.infer import run_inference


class DummyModel:
    def predict(self, X):
        # Deterministic output for testing
        return np.array([42] * len(X))


def test_run_inference_returns_correct_schema_and_values():
    # Arrange
    X_infer = pd.DataFrame(
        {
            "num_feature": [0.1, 0.2, 0.3],
            "cat_feature": ["A", "B", "C"],
        },
        index=[10, 20, 30],
    )
    model = DummyModel()

    # Act
    df_pred = run_inference(model=model, X_infer=X_infer)

    # Assert: correct type
    assert isinstance(df_pred, pd.DataFrame)

    # Assert: correct column name
    assert list(df_pred.columns) == ["prediction"]

    # Assert: correct length
    assert len(df_pred) == len(X_infer)

    # Assert: correct values from DummyModel
    assert (df_pred["prediction"] == np.exp(42)).all()


def test_run_inference_preserves_index():
    # Arrange
    X_infer = pd.DataFrame(
        {"num_feature": [1, 2]},
        index=["row_1", "row_2"],
    )
    model = DummyModel()

    # Act
    df_pred = run_inference(model=model, X_infer=X_infer)

    # Assert: index is preserved exactly
    assert df_pred.index.equals(X_infer.index)


def test_run_inference_does_not_mutate_input():
    # Arrange
    X_infer = pd.DataFrame(
        {"num_feature": [1, 2], "cat_feature": ["A", "B"]}
    )
    X_before = X_infer.copy(deep=True)
    model = DummyModel()

    # Act
    _ = run_inference(model=model, X_infer=X_infer)

    # Assert: original input is unchanged
    assert X_infer.equals(X_before)
