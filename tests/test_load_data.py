import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
from src.load_data import load_raw_data

"""
Tests for src.load_data module.
"""




class TestLoadRawData:
    """Test suite for load_raw_data function."""

    def test_load_raw_data_success(self, tmp_path):
        """Test successful loading of valid CSV file."""
        csv_file = tmp_path / "test.csv"
        df_expected = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df_expected.to_csv(csv_file, index=False)

        with patch("src.load_data.load_csv", return_value=df_expected):
            result = load_raw_data(csv_file)
            assert isinstance(result, pd.DataFrame)
            assert not result.empty

    def test_load_raw_data_none_path(self):
        """Test that None path raises ValueError."""
        with pytest.raises(ValueError, match="raw_data_path cannot be None"):
            load_raw_data(None)

    def test_load_raw_data_file_not_found(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Raw data file not found"):
            load_raw_data(Path("/nonexistent/path/file.csv"))

    def test_load_raw_data_is_directory(self, tmp_path):
        """Test that directory path raises IsADirectoryError."""
        with pytest.raises(IsADirectoryError, match="raw_data_path must be a file"):
            load_raw_data(tmp_path)

    def test_load_raw_data_empty_dataframe(self, tmp_path):
        """Test that empty dataframe raises ValueError."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        with patch("src.load_data.load_csv", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="Loaded dataframe is empty"):
                load_raw_data(csv_file)

    def test_load_raw_data_none_dataframe(self, tmp_path):
        """Test that None dataframe raises ValueError."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\n")

        with patch("src.load_data.load_csv", return_value=None):
            with pytest.raises(ValueError, match="Loaded dataframe is empty"):
                load_raw_data(csv_file)