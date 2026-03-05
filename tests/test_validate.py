import pytest
import pandas as pd

from src.validate import validate_dataframe


@pytest.fixture
def required_columns():
    # columns that must exist in df
    return ["age", "sex", "bmi", "children", "smoker", "region", "charges"]


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


def test_validate_passes_on_valid_data(valid_df, required_columns):
    assert validate_dataframe(valid_df, required_columns, target_column="charges") is True


def test_validate_fails_on_empty_dataframe(required_columns):
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns, target_column="charges")


def test_validate_fails_on_missing_required_column(valid_df, required_columns):
    df = valid_df.drop(columns=["age"])
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns, target_column="charges")


def test_validate_fails_on_missing_target_column(valid_df, required_columns):
    df = valid_df.drop(columns=["charges"])
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns, target_column="charges")


def test_validate_fails_when_target_not_numeric(valid_df, required_columns):
    df = valid_df.copy()
    df["charges"] = ["not-a-number"]
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns, target_column="charges")