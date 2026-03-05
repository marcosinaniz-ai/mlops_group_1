from pathlib import Path

import pandas as pd
import pytest

from src.utils import load_csv, save_csv, save_model, load_model


def test_save_and_load_csv_roundtrip(tmp_path: Path):
    # Arrange
    df_in = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out_path = tmp_path / "nested" / "folder" / "data.csv"

    # Act
    save_csv(df_in, out_path)
    df_out = load_csv(out_path)

    # Assert: file created and data round-trips
    assert out_path.exists()
    pd.testing.assert_frame_equal(df_out, df_in)


def test_save_csv_does_not_write_index_by_default(tmp_path: Path):
    # Arrange: give it a non-default index to catch accidental index saving
    df_in = pd.DataFrame({"a": [10, 20]}, index=[100, 200])
    out_path = tmp_path / "data.csv"

    # Act
    save_csv(df_in, out_path)
    df_out = pd.read_csv(out_path)  # direct read to inspect columns

    # Assert: no "Unnamed: 0" (common symptom of index being written)
    assert "Unnamed: 0" not in df_out.columns
    assert list(df_out.columns) == ["a"]


def test_load_csv_raises_file_not_found(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(missing_path)


def test_load_csv_raises_on_empty_csv(tmp_path: Path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")  # creates an empty file

    with pytest.raises(Exception):
        # Depending on pandas version, pd.read_csv("") may raise
        # EmptyDataError OR your ValueError if it loads empty.
        load_csv(empty_path)


def test_save_and_load_model_roundtrip(tmp_path: Path):
    # Arrange: model can be any python object for joblib purposes
    model_in = {"name": "dummy_model", "version": 1}
    model_path = tmp_path / "nested" / "models" / "model.joblib"

    # Act
    save_model(model_in, model_path)
    model_out = load_model(model_path)

    # Assert
    assert model_path.exists()
    assert model_out == model_in


def test_load_model_raises_file_not_found(tmp_path: Path):
    missing_model_path = tmp_path / "models" / "missing.joblib"
    with pytest.raises(FileNotFoundError):
        load_model(missing_model_path)