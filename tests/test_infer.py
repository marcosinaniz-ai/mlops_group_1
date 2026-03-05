import pandas as pd

from src.infer import run_inference


class DummyModel:
    def predict(self, X):
        # Deterministic, simple output to make assertions easy
        return [1] * len(X)


def test_run_inference_returns_single_prediction_column_and_preserves_index():
    # Arrange
    X_infer = pd.DataFrame(
        {"num_feature": [0.1, 0.2], "cat_feature": ["A", "B"]},
        index=[100, 200],
    )
    X_before = X_infer.copy(deep=True)  # to verify no mutation
    model = DummyModel()

    # Act
    df_pred = run_inference(model=model, X_infer=X_infer)

    # Assert: type + schema
    assert isinstance(df_pred, pd.DataFrame)
    assert list(df_pred.columns) == ["prediction"]
    assert len(df_pred) == len(X_infer)

    # Assert: index preserved
    assert df_pred.index.equals(X_infer.index)

    # Assert: input was not mutated
    assert X_infer.equals(X_before)
