import pytest
import pandas as pd
import numpy as np
from src.clean_data import clean_dataframe

"""
Tests for src.clean_data module.
"""



class TestCleanDataframeValidation:
    """Tests for input validation."""

    def test_none_input(self):
        """Should raise ValueError when df_raw is None."""
        with pytest.raises(ValueError, match="df_raw cannot be None"):
            clean_dataframe(None, "log_charges")

    def test_non_dataframe_input(self):
        """Should raise TypeError when df_raw is not a DataFrame."""
        with pytest.raises(TypeError, match="df_raw must be a pandas DataFrame"):
            clean_dataframe([], "log_charges")

    def test_empty_dataframe(self):
        """Should raise ValueError when df_raw is empty."""
        with pytest.raises(ValueError, match="df_raw is empty"):
            clean_dataframe(pd.DataFrame(), "log_charges")

    def test_invalid_target_column(self):
        """Should raise ValueError for invalid target_column."""
        df = pd.DataFrame({"charges": [100]})
        with pytest.raises(ValueError, match="target_column must be a non-empty string"):
            clean_dataframe(df, "")

    def test_missing_required_columns(self):
        """Should raise ValueError when required columns are missing."""
        df = pd.DataFrame({"age": [25], "charges": [1000]})
        with pytest.raises(ValueError, match="Missing required columns"):
            clean_dataframe(df, "log_charges")


class TestCleanDataframeProcessing:
    """Tests for data cleaning logic."""

    def sample_df(self):
        """Create a valid sample DataFrame."""
        return pd.DataFrame({
            "age": [25, 30, 35],
            "sex": ["Male", "Female", "Male"],
            "bmi": [22.5, 25.0, 28.0],
            "children": [0, 1, 2],
            "smoker": ["no", "yes", "no"],
            "region": ["southwest", "northwest", "southeast"],
            "charges": [1000.0, 2000.0, 3000.0]
        })

    def test_valid_cleaning(self):
        """Should successfully clean valid DataFrame."""
        df = self.sample_df()
        result = clean_dataframe(df, "log_charges")
        
        assert "log_charges" in result.columns
        assert "charges" not in result.columns
        assert len(result) == 3

    def test_duplicates_removed(self):
        """Should remove duplicate rows."""
        df = self.sample_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result = clean_dataframe(df, "log_charges")
        
        assert len(result) == 3

    def test_categorical_normalization(self):
        """Should normalize categorical values to lowercase."""
        df = self.sample_df()
        df.loc[0, "sex"] = "MALE"
        df.loc[1, "smoker"] = "YES"
        result = clean_dataframe(df, "log_charges")
        
        assert result["sex"].iloc[0] == "male"
        assert result["smoker"].iloc[1] == "yes"

    def test_smoker_alias_replacement(self):
        """Should replace y/n aliases for smoker."""
        df = self.sample_df()
        df.loc[0, "smoker"] = "y"
        df.loc[1, "smoker"] = "n"
        result = clean_dataframe(df, "log_charges")
        
        assert result["smoker"].iloc[0] == "yes"
        assert result["smoker"].iloc[1] == "no"

    def test_non_positive_charges_error(self):
        """Should raise ValueError for non-positive charges."""
        df = self.sample_df()
        df.loc[0, "charges"] = 0
        with pytest.raises(ValueError, match="'charges' must be positive"):
            clean_dataframe(df, "log_charges")

    def test_log_charges_computation(self):
        """Should correctly compute log of charges."""
        df = self.sample_df()
        result = clean_dataframe(df, "log_charges")
        
        expected_log = np.log(df["charges"].values)
        assert np.allclose(result["log_charges"].values, expected_log)

    def test_numeric_coercion(self):
        """Should coerce numeric columns to float."""
        df = self.sample_df()
        df["age"] = df["age"].astype(str)
        result = clean_dataframe(df, "log_charges")
        
        assert pd.api.types.is_numeric_dtype(result["age"])

    def test_column_name_normalization(self):
        """Should strip and standardize column names."""
        df = self.sample_df()
        df.columns = [f" {c} " for c in df.columns]
        result = clean_dataframe(df, "log_charges")
        
        assert "charges" not in result.columns
        assert "log_charges" in result.columns