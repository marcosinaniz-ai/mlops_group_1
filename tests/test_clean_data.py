import pandas as pd
import numpy as np
import pytest

from src.clean_data import clean_dataframe


# Shared test config (matches your function inputs)
REQUIRED_COLUMNS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
CATEGORICAL_COLUMNS = {"sex", "smoker", "region"}
NUMERIC_COLUMNS = {"age", "bmi", "children", "charges"}


def get_valid_dataframe():
    return pd.DataFrame({
        "age": [25, 30],
        "sex": ["Male", "Female"],
        "bmi": [22.5, 27.3],
        "children": [0, 1],
        "smoker": ["yes", "no"],
        "region": ["northwest", "southeast"],
        "charges": [1000.0, 2000.0],
    })


# ----------------------------------------
# SUCCESS CASE
# ----------------------------------------
def test_clean_dataframe_success():
    
    df_raw = get_valid_dataframe()

    df_clean = clean_dataframe(
        df_raw=df_raw,
        target_column="log_charges",
        required_columns=REQUIRED_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        numeric_columns=NUMERIC_COLUMNS,
    )

    # Target exists
    assert "log_charges" in df_clean.columns

    # Charges removed
    assert "charges" not in df_clean.columns

    # Values are correct
    expected = np.log(df_raw["charges"])
    assert np.allclose(df_clean["log_charges"], expected)


# ----------------------------------------
# FAILURE CASES
# ----------------------------------------

def test_clean_dataframe_raises_on_none_input():
    with pytest.raises(ValueError):
        clean_dataframe(
            None,
            "log_charges",
            REQUIRED_COLUMNS,
            CATEGORICAL_COLUMNS,
            NUMERIC_COLUMNS
        )


def test_clean_dataframe_raises_on_wrong_type():
    with pytest.raises(TypeError):
        clean_dataframe(
            "not_a_dataframe",
            "log_charges",
            REQUIRED_COLUMNS,
            CATEGORICAL_COLUMNS,
            NUMERIC_COLUMNS
        )


def test_clean_dataframe_raises_on_empty_dataframe():
    df_empty = pd.DataFrame()

    with pytest.raises(ValueError):
        clean_dataframe(
            df_empty,
            "log_charges",
            REQUIRED_COLUMNS,
            CATEGORICAL_COLUMNS,
            NUMERIC_COLUMNS
        )


def test_clean_dataframe_raises_on_missing_columns():
    df = get_valid_dataframe().drop(columns=["age"])

    with pytest.raises(ValueError):
        clean_dataframe(
            df,
            "log_charges",
            REQUIRED_COLUMNS,
            CATEGORICAL_COLUMNS,
            NUMERIC_COLUMNS
        )


def test_clean_dataframe_raises_on_non_positive_charges():
    df = get_valid_dataframe()
    df["charges"] = [1000, 0]  # invalid

    with pytest.raises(ValueError):
        clean_dataframe(
            df,
            "log_charges",
            REQUIRED_COLUMNS,
            CATEGORICAL_COLUMNS,
            NUMERIC_COLUMNS
        )
