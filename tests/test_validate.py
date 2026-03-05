import pytest
import pandas as pd
import yaml

from src.validate import validate_dataframe


@pytest.fixture
def config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "age": [30],
            "sex": ["male"],
            "bmi": [25.0],
            "children": [1],
            "smoker": ["no"],
            "region": ["southwest"],
            "charges": [2000.0],
        }
    )


def test_validate_passes_on_valid_data(valid_df, config):
    assert validate_dataframe(valid_df, config) is True


def test_validate_fails_on_missing_column(valid_df, config):
    df = valid_df.drop(columns=["age"])
    with pytest.raises(ValueError):
        validate_dataframe(df, config)


def test_validate_fails_on_negative_value(valid_df, config):
    df = valid_df.copy()
    df["age"] = -5
    with pytest.raises(ValueError):
        validate_dataframe(df, config)


def test_validate_fails_on_invalid_category(valid_df, config):
    df = valid_df.copy()
    df["sex"] = "invalid"
    with pytest.raises(ValueError):
        validate_dataframe(df, config)