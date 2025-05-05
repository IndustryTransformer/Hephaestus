from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    pass


@dataclass
class Results:
    """Data class to store model results.

    Attributes:
        numeric_out (torch.Tensor): Numeric output from the model.
        categorical_out (torch.Tensor): Categorical output from the model.
        numeric_inputs (torch.Tensor): Numeric inputs to the model.
        categorical_inputs (torch.Tensor): Categorical inputs to the model.
    """

    numeric_out: torch.Tensor
    categorical_out: torch.Tensor
    numeric_inputs: torch.Tensor
    categorical_inputs: torch.Tensor


def return_results(model, dataset, idx=0, mask_start: int = None):
    # """Return model results for a given dataset index.

    # Args:
    #     model: The model to generate results from.
    #     dataset: The dataset to use.
    #     idx (int, optional): Index of the dataset to use. Defaults to 0.
    #     mask_start (int, optional): Index to start masking inputs. Defaults to None.

    # Returns:
    #     Results: The results from the model.
    # """
    # model.eval()
    # date_set_base = dataset[idx]

    # if mask_start:
    #     numeric_inputs = numeric_inputs[:, :mask_start]
    #     categorical_inputs = categorical_inputs[:, :mask_start]

    # numeric_inputs = numeric_inputs.unsqueeze(0)
    # categorical_inputs = categorical_inputs.unsqueeze(0)

    # out = model.predict_step()

    # numeric_out, categorical_out = out.numeric, out.categorical
    # return Results(numeric_out, categorical_out, numeric_inputs, categorical_inputs)
    pass


def process_results(arr: torch.Tensor, col_names: list, config):
    """Process model results into a DataFrame.

    Args:
        arr (torch.Tensor): Array of results.
        col_names (list): List of column names.
        config: Configuration for the time series.

    Returns:
        pd.DataFrame: DataFrame of processed results.
    """
    arr = arr.squeeze().cpu().numpy()
    # arr = arr.squeeze().detach().cpu().numpy()
    if len(arr.shape) == 3:
        # Check if there is a logit array for example if there are 3 dims then the
        # last dim is the logit array. We need to get the argmax of the last dim
        # to get the actual values of the array and replace the logit array with the
        # actual values
        arr = np.argmax(arr, axis=-1)
    df = pd.DataFrame(arr.T)
    df.columns = col_names
    return df


@dataclass
class DFComparison:
    """Data class to store input and output DataFrames for comparison.

    Attributes:
        input_df (pd.DataFrame): Input DataFrame.
        output_df (pd.DataFrame): Output DataFrame.
    """

    input_df: pd.DataFrame
    output_df: pd.DataFrame


def show_results_df(
    model, time_series_config, dataset, idx: int = 0, mask_start: int = None
):
    """Show results as DataFrames for a given dataset index.

    Args:
        model: The model to generate results from.
        time_series_config: Configuration for the time series.
        dataset: The dataset to use.
        idx (int, optional): Index of the dataset to use. Defaults to 0.
        mask_start (int, optional): Index to start masking inputs. Defaults to None.

    Returns:
        DFComparison: DataFrames of input and output results.
    """
    results = model.return_results(dataset, idx=idx, mask_start=mask_start)

    input_categorical = process_results(
        results.categorical_inputs,
        time_series_config.categorical_col_tokens,
        time_series_config,
    )
    input_numeric = process_results(
        results.numeric_inputs,
        time_series_config.numeric_col_tokens,
        time_series_config,
    )
    output_categorical = process_results(
        results.categorical_out,
        time_series_config.categorical_col_tokens,
        time_series_config,
    )
    output_numeric = process_results(
        results.numeric_out, time_series_config.numeric_col_tokens, time_series_config
    )
    input_df = pd.concat([input_categorical, input_numeric], axis=1)
    output_df = pd.concat([output_categorical, output_numeric], axis=1)

    # Model predicts the next time step, so we need to shift the output DataFrame
    # to align with the input DataFrame
    output_df = output_df.iloc[:-1].reset_index(drop=True)
    input_df = input_df.iloc[1:].reset_index(drop=True)
    return DFComparison(input_df, output_df)


@dataclass
class AutoRegressiveResults:
    """Data class to store auto-regressive results.

    Attributes:
        numeric_inputs (torch.Tensor): Numeric inputs for auto-regressive predictions.
        categorical_inputs (torch.Tensor): Categorical inputs for auto-regressive predictions.
    """

    numeric: torch.Tensor
    categorical: torch.Tensor

    @classmethod
    def from_ds(cls, ds: Any, idx: int, stop_idx: int = 10):
        """Create AutoRegressiveResults from a dataset.

        Args:
            ds: The dataset to use.
            idx (int): Index of the dataset to use.
            stop_idx (int, optional): Index to stop at. Defaults to 10.

        Returns:
            AutoRegressiveResults: The auto-regressive results.
        """
        inputs = ds[idx]
        numeric = inputs.numeric[:, :stop_idx]
        categorical = inputs.categorical[:, :stop_idx]
        numeric = (
            torch.tensor(numeric) if not isinstance(numeric, torch.Tensor) else numeric
        )
        categorical = (
            torch.tensor(categorical)
            if not isinstance(categorical, torch.Tensor)
            else categorical
        )
        return cls(numeric, categorical)


def auto_regressive_predictions(
    model, inputs: AutoRegressiveResults, device: str = "cuda"
) -> AutoRegressiveResults:
    """Generate auto-regressive predictions.

    Args:
        model: The model to generate predictions from.
        inputs (AutoRegressiveResults): The inputs for auto-regressive predictions.

    Returns:
        AutoRegressiveResults: The updated inputs.
    """

    # Ensure 3D shape (batch, features, time)
    if inputs.numeric.dim() == 2:
        inputs.numeric = inputs.numeric.unsqueeze(0)
        inputs.categorical = inputs.categorical.unsqueeze(0)

    # Track nan columns
    numeric_nan_columns = torch.isnan(inputs.numeric).all(dim=2)
    categorical_nan_columns = torch.isnan(inputs.categorical).all(dim=2)

    # Get model predictions
    outputs = model.predict_step(inputs)

    # Handle numeric predictions
    numeric_next = outputs.numeric[:, :, -1:]  # Shape: (batch, features, 1)

    # Handle categorical predictions
    categorical_out = outputs.categorical  # Shape: (batch, features, time, classes)
    categorical_last = categorical_out[:, :, -1]  # Shape: (batch, features, classes)
    categorical_next = torch.argmax(
        categorical_last, dim=-1
    )  # Shape: (batch, features)
    # Reshape to match input dimensions (batch, features, 1)
    categorical_next = categorical_next.unsqueeze(-1)  # Add time dimension

    # Ensure categorical_next has the same batch and feature dimensions as categorical_inputs
    categorical_next = categorical_next.expand(
        inputs.categorical.shape[0], inputs.categorical.shape[1], 1
    )

    # Concatenate new predictions
    inputs.numeric = torch.cat([inputs.numeric, numeric_next], dim=2)
    inputs.categorical = torch.cat(
        [inputs.categorical, categorical_next.to(torch.float32)], dim=2
    )

    # Restore nan values
    inputs.numeric[numeric_nan_columns] = float("nan")
    inputs.categorical[categorical_nan_columns] = float("nan")

    return AutoRegressiveResults(inputs.numeric, inputs.categorical)


def create_test_inputs_df(test_inputs, time_series_config):
    """Create a DataFrame of test inputs.

    Args:
        test_inputs (AutoRegressiveResults): The test inputs.
        time_series_config: Configuration for the time series.

    Returns:
        pd.DataFrame: DataFrame of test inputs.
    """
    # Extract numeric and categorical inputs from test_inputs
    numeric_inputs = test_inputs.numeric
    categorical_inputs = test_inputs.categorical
    numeric_inputs = numeric_inputs.squeeze().cpu().numpy()
    categorical_inputs = categorical_inputs.squeeze().cpu().numpy()

    # Get column names from time_series_config
    numeric_col_names = time_series_config.numeric_col_tokens
    categorical_col_names = time_series_config.categorical_col_tokens

    # Create DataFrames for numeric and categorical inputs
    numeric_df = pd.DataFrame(numeric_inputs.T, columns=numeric_col_names)
    categorical_df = pd.DataFrame(categorical_inputs.T, columns=categorical_col_names)

    # Combine the DataFrames
    test_inputs_df = pd.concat([numeric_df, categorical_df], axis=1)
    return test_inputs_df


def create_non_auto_df(res, time_series_config):
    """Create a DataFrame of non-auto-regressive results.

    Args:
        res (dict): Dictionary of results.
        time_series_config: Configuration for the time series.

    Returns:
        pd.DataFrame: DataFrame of non-auto-regressive results.
    """
    numeric_out = res["numeric"]
    categorical_out = res["categorical"]
    numeric_df = process_results(
        numeric_out, time_series_config.numeric_col_tokens, time_series_config
    )
    categorical_df = process_results(
        categorical_out,
        time_series_config.categorical_col_tokens,
        time_series_config,
    )
    return pd.concat([categorical_df, numeric_df], axis=1)
