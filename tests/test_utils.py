from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    load_csv,
    save_csv,
    save_json,
    save_model,
    load_model,
    safe_exp,
)


def test_load_csv_raises_if_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(missing)


def test_load_csv_raises_if_directory(tmp_path: Path):
    directory = tmp_path / "some_dir"
    directory.mkdir()
    with pytest.raises(IsADirectoryError):
        load_csv(directory)


def test_load_csv_raises_if_empty_csv(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")  # valid file but no rows
    with pytest.raises(ValueError):
        load_csv(p)


def test_load_csv_success(tmp_path: Path):
    p = tmp_path / "data.csv"
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_in.to_csv(p, index=False)

    df_out = load_csv(p)

    assert isinstance(df_out, pd.DataFrame)
    assert df_out.shape == (2, 2)
    assert list(df_out.columns) == ["a", "b"]


def test_save_csv_creates_parent_and_saves_without_index_by_default(tmp_path: Path):
    df = pd.DataFrame({"x": [10, 20]})
    out = tmp_path / "nested" / "out.csv"

    save_csv(df, out)  # default index=False

    assert out.exists()
    # If index=False, the first column should be "x", not "Unnamed: 0"
    df_read = pd.read_csv(out)
    assert list(df_read.columns) == ["x"]
    assert df_read["x"].tolist() == [10, 20]


def test_save_csv_with_index_true_writes_index_column(tmp_path: Path):
    df = pd.DataFrame({"x": [10, 20]})
    out = tmp_path / "out_with_index.csv"

    save_csv(df, out, index=True)

    df_read = pd.read_csv(out)
    # pandas reads the saved index column back as "Unnamed: 0"
    assert "Unnamed: 0" in df_read.columns
    assert df_read["x"].tolist() == [10, 20]


def test_save_json_creates_parent_and_saves(tmp_path: Path):
    obj = {"a": 1, "b": {"c": 2}}
    out = tmp_path / "nested" / "obj.json"

    save_json(obj, out)

    assert out.exists()
    import json

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == obj


def test_save_and_load_model_roundtrip(tmp_path: Path):
    # Minimal "model" object joblib can serialize
    model = {"name": "dummy", "weights": [1, 2, 3]}
    out = tmp_path / "models" / "model.joblib"

    save_model(model, out)
    assert out.exists()

    loaded = load_model(out)
    assert loaded == model


def test_load_model_raises_if_missing(tmp_path: Path):
    missing = tmp_path / "nope.joblib"
    with pytest.raises(FileNotFoundError):
        load_model(missing)


def test_safe_exp_scalar():
    out = safe_exp(0.0)
    assert out == pytest.approx(1.0)


def test_safe_exp_numpy_array():
    arr = np.array([0.0, 1.0])
    out = safe_exp(arr)
    assert isinstance(out, np.ndarray)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(np.e)


def test_safe_exp_pandas_series():
    s = pd.Series([0.0, 1.0])
    out = safe_exp(s)
    assert isinstance(out, pd.Series)
    assert out.iloc[0] == pytest.approx(1.0)
    assert out.iloc[1] == pytest.approx(np.e)
