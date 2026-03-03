import yaml
from sklearn.compose import ColumnTransformer

from src.features import build_preprocessor


def test_build_preprocessor_returns_column_transformer():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    preprocessor = build_preprocessor(config)

    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.remainder == "drop"