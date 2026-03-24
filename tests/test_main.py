import pandas as pd
import pytest
from unittest.mock import MagicMock

import src.main as main_module

TARGET_COLUMN = "log_charges"

@pytest.fixture
def dummy_dataframe():
    """Minimal valid dataframe matching SETTINGS contract."""
    return pd.DataFrame(
        {
            "age": [30, 40],
            "bmi": [25.0, 30.0],
            "children": [1, 2],
            "sex": ["male", "female"],
            "smoker": ["no", "yes"],
            "region": ["southwest", "southeast"],
            "log_charges": [10.0, 11.0],
        }
    )


def test_main_orchestrates_pipeline(monkeypatch, dummy_dataframe):
    """
    This test verifies that main():
      - Calls each pipeline step
      - Passes expected arguments
      - Saves model and predictions
    It does NOT test business logic inside individual modules.
    """

    # -------------------------
    # Mock external functions
    # -------------------------

    mock_load = MagicMock(return_value=dummy_dataframe)
    mock_clean = MagicMock(return_value=dummy_dataframe)
    mock_save_csv = MagicMock()
    mock_validate = MagicMock()
    mock_preprocessor = MagicMock()
    mock_train = MagicMock(return_value="trained_model")
    mock_save_model = MagicMock()
    mock_evaluate = MagicMock(return_value={"rmse": 0.5})
    mock_infer = MagicMock(return_value=dummy_dataframe)

    # Apply monkeypatching
    monkeypatch.setattr(main_module, "load_raw_data", mock_load)
    monkeypatch.setattr(main_module, "clean_dataframe", mock_clean)
    monkeypatch.setattr(main_module, "save_csv", mock_save_csv)
    monkeypatch.setattr(main_module, "validate_dataframe", mock_validate)
    monkeypatch.setattr(main_module, "get_feature_preprocessor", MagicMock(return_value=mock_preprocessor))
    monkeypatch.setattr("src.train.train_model", mock_train)
    monkeypatch.setattr(main_module, "save_model", mock_save_model)
    monkeypatch.setattr(main_module, "evaluate_model", mock_evaluate)
    monkeypatch.setattr(main_module, "run_inference", mock_infer)

    # -------------------------
    # Execute main
    # -------------------------
    main_module.main()

    # -------------------------
    # Assertions
    # -------------------------

    # Load was called once
    mock_load.assert_called_once()

    # Clean was called with correct target column
    mock_clean.assert_called_once()
    assert mock_clean.call_args.kwargs["target_column"] == TARGET_COLUMN

    # Validation was called
    mock_validate.assert_called_once()

    # Model training occurred
    mock_train.assert_called_once()
    assert mock_train.call_args.kwargs["preprocessor"] == mock_preprocessor

    # Model saved
    mock_save_model.assert_called_once()

    # Evaluation occurred
    mock_evaluate.assert_called_once()

    # Inference occurred
    mock_infer.assert_called_once()

    # Predictions saved
    assert mock_save_csv.call_count >= 2  # clean + predictions
