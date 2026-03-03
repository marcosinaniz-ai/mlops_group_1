import pandas as pd

"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""

"""

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: Fitted scikit-learn Pipeline.
    - X_infer: Inference features DataFrame (same raw columns as training).
    Outputs:
    - df_pred: DataFrame with a SINGLE column named 'prediction' preserving
    input index.
    Why this contract matters for reliable ML delivery:
    - A stable prediction schema simplifies integrations (batch jobs, APIs)
    and reduces downstream breaking changes.
    """
    print(
        "[infer.run_inference] Running inference and returning prediction."
    )  # TODO: replace with logging later

    preds = model.predict(X_infer)
    df_pred = pd.DataFrame({"prediction": preds}, index=X_infer.index)

    return df_pred
