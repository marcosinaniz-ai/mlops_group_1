import pytest
from sklearn.compose import ColumnTransformer

from src.features import get_feature_preprocessor


def test_get_feature_preprocessor_returns_column_transformer():
    preprocessor = get_feature_preprocessor(
        numeric_cols=["age", "income"],
        categorical_cols=["city"]
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.remainder == "drop"


def test_get_feature_preprocessor_raises_if_no_columns():
    with pytest.raises(ValueError):
        get_feature_preprocessor(numeric_cols=[], categorical_cols=[])
