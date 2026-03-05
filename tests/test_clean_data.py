import pandas as pd
import pytest

from src.clean_data import clean_dataframe


def test_clean_dataframe_happy_path_normalizes_and_dedupes():
    df_raw = pd.DataFrame(
        {
            "age": [20, 20],
            "sex": [" Male ", " Male "],
            "bmi": [30.0, 30.0],
            "children": [0, 0],
            "smoker": ["No", "No"],
            "region": [" Southwest ", " Southwest "],
            "charges": [1000.0, 1000.0],
        }
    )
    df_clean = clean_dataframe(df_raw, target_column="charges")

    assert len(df_clean) == 1
    assert df_clean.loc[df_clean.index[0], "sex"] == "male"
    assert df_clean.loc[df_clean.index[0], "smoker"] == "no"
    assert df_clean.loc[df_clean.index[0], "region"] == "southwest"


def test_clean_dataframe_missing_required_columns_raises():
    df_raw = pd.DataFrame({"age": [20], "charges": [1000.0]})
    with pytest.raises(ValueError):
        clean_dataframe(df_raw, target_column="charges")


def test_clean_dataframe_requires_target_present():
    df_raw = pd.DataFrame(
        {
            "age": [20],
            "sex": ["male"],
            "bmi": [30.0],
            "children": [0],
            "smoker": ["no"],
            "region": ["southwest"],
            "charges": [1000.0],
        }
    )
    with pytest.raises(ValueError):
        clean_dataframe(df_raw, target_column="not_a_real_target")


def test_clean_dataframe_numeric_casting():
    df_raw = pd.DataFrame(
        {
            "age": ["20"],
            "sex": ["male"],
            "bmi": ["30.0"],
            "children": ["0"],
            "smoker": ["no"],
            "region": ["southwest"],
            "charges": ["1000.0"],
        }
    )
    df_clean = clean_dataframe(df_raw, target_column="charges")
    assert df_clean["age"].dtype.kind in {"i", "u", "f"}
    assert df_clean["charges"].dtype.kind in {"i", "u", "f"}