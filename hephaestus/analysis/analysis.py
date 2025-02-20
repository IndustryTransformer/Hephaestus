import jax.numpy as jnp
import numpy as np
import pandas as pd
import seaborn as sns
from flax.struct import dataclass
from matplotlib import pyplot as plt

from ..models.models import TimeSeriesConfig
from ..training.training import time_series_regressor


@dataclass
class Results:
    numeric_out: jnp.array
    categorical_out: jnp.array
    numeric_inputs: jnp.array
    categorical_inputs: jnp.array


def return_results(model, dataset, idx=0, mask_start: int = None):
    numeric_inputs, categorical_inputs = dataset[idx]
    if mask_start:
        numeric_inputs = numeric_inputs[:, :mask_start]
        categorical_inputs = categorical_inputs[:, :mask_start]
    numeric_inputs = jnp.array([numeric_inputs])
    categorical_inputs = jnp.array([categorical_inputs])
    out = time_series_regressor(
        numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
    )
    numeric_out, categorical_out = out["numeric_out"], out["categorical_out"]
    return Results(numeric_out, categorical_out, numeric_inputs, categorical_inputs)


def process_results(arr: jnp.array, col_names: list, config: TimeSeriesConfig):
    arr = jnp.squeeze(arr)
    if arr.ndim == 3:
        # Check if there is a logit array for example if there are 3 dims then the
        # last dim is the logit array. We need to get the argmax of the last dim
        # to get the actual values of the array and replace the logit array with the
        # actual values
        arr = jnp.argmax(arr, axis=-1)
    df = pd.DataFrame(arr.T)
    df.columns = col_names
    return df


@dataclass
class DFComparison:
    input_df: pd.DataFrame
    output_df: pd.DataFrame


def show_results_df(
    model, time_series_config, dataset, idx: int = 0, mask_start: int = None
):
    results = return_results(model, dataset, idx=idx, mask_start=mask_start)

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

    return DFComparison(input_df, output_df)


@dataclass
class AutoRegressiveResults:
    numeric_inputs: jnp.array
    categorical_inputs: jnp.array

    @classmethod
    def from_ds(cls, ds: TimeSeriesConfig, idx: int, stop_idx: int = 10):
        inputs = ds[idx]
        numeric_inputs = inputs[0][:, :stop_idx]
        categorical_inputs = inputs[1][:, :stop_idx]
        return cls(numeric_inputs, categorical_inputs)


def auto_regressive_predictions(
    model: TimeSeriesConfig,
    inputs: AutoRegressiveResults,
) -> jnp.array:
    numeric_inputs = inputs.numeric_inputs
    categorical_inputs = inputs.categorical_inputs

    # get the first row that contains all nan vales
    # if nan_rows_start >= stop_idx:
    #     return inputs
    # numeric_inputs = inputs.numeric_inputs
    # categorical_inputs = inputs.categorical_inputs

    # Expand first dim to make it a batch
    if numeric_inputs.ndim == 2:
        print("Expanding dims")
        numeric_inputs = jnp.array(jnp.array([numeric_inputs]))
        categorical_inputs = jnp.array([categorical_inputs])

    numeric_nan_columns = jnp.isnan(numeric_inputs).all(axis=2)
    categorical_nan_columns = jnp.isnan(categorical_inputs).all(axis=2)

    outputs = time_series_regressor(
        numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
    )
    numeric_out = outputs["numeric_out"]  #  jnp.squeeze(outputs["numeric_out"])
    # print(f"{numeric_out.shape=}")
    categorical_out = outputs[
        "categorical_out"
    ]  # jnp.squeeze(outputs["categorical_out"])

    categorical_out = jnp.argmax(categorical_out, axis=-1)

    final_numeric_row = np.array(numeric_out[:, :, -1])

    final_numeric_row = final_numeric_row[None, :]  # New axis
    final_numeric_row = jnp.transpose(final_numeric_row, (0, 2, 1))
    # print(f"{final_numeric_row.shape=}")
    final_categorical_row = np.array(categorical_out[:, :, -1])
    final_categorical_row = final_categorical_row[:, None]  # New axis
    final_categorical_row = jnp.transpose(final_categorical_row, (0, 2, 1))
    # print(f"{numeric_inputs.shape=}, {final_numeric_row.shape=}")
    numeric_inputs = jnp.concatenate(
        [numeric_inputs, final_numeric_row], axis=2
    )  # here

    categorical_inputs = jnp.concatenate(
        [categorical_inputs, final_categorical_row], axis=2
    )
    numeric_inputs = numeric_inputs.at[jnp.array(numeric_nan_columns)].set(jnp.nan)
    categorical_inputs = categorical_inputs.at[jnp.array(categorical_nan_columns)].set(
        jnp.nan
    )
    inputs = (numeric_inputs, categorical_inputs)

    return AutoRegressiveResults(numeric_inputs, categorical_inputs)
    # return auto_regressive_predictions(state, inputs, stop_idx)


def plot_column_variants(
    df_pred: pd.DataFrame, df_actual: pd.DataFrame, column: str, offset=0
):
    plt.figure(figsize=(15, 10))
    plt.plot(df_pred[column], label="Autogregressive")
    plt.plot(df_actual[column], label="Actual")
    plt.title(f"{column} Predictions")
    plt.legend()
    # Show ticks and grid lines every 1 step
    plt.xticks(np.arange(0, len(df_pred), 1))
    plt.grid()
    # add black line at 0 on the y axis to show the difference
    plt.axhline(0, color="black")
    plt.show()


def create_test_inputs_df(test_inputs, time_series_config):
    # Extract numeric and categorical inputs from test_inputs
    numeric_inputs = test_inputs.numeric_inputs
    categorical_inputs = test_inputs.categorical_inputs
    numeric_inputs = jnp.squeeze(numeric_inputs)
    categorical_inputs = jnp.squeeze(categorical_inputs)
    # Get column names from time_series_config
    numeric_col_names = time_series_config.numeric_col_tokens
    categorical_col_names = time_series_config.categorical_col_tokens

    # Create DataFrames for numeric and categorical inputs
    numeric_df = pd.DataFrame(numeric_inputs.T, columns=numeric_col_names)
    categorical_df = pd.DataFrame(categorical_inputs.T, columns=categorical_col_names)

    # Combine the DataFrames
    test_inputs_df = pd.concat([numeric_df, categorical_df], axis=1)
    return test_inputs_df


def plot_comparison(actual_df, one_off_auto_df, auto_regressive_df, column):
    plt.figure(figsize=(15, 10))

    # Plot actual data
    sns.lineplot(data=actual_df, x=actual_df.index, y=column, label="Actual")

    # Plot one-off auto data
    sns.lineplot(
        data=one_off_auto_df, x=one_off_auto_df.index, y=column, label="One-Off Auto"
    )

    # Plot auto-regressive data
    sns.lineplot(
        data=auto_regressive_df,
        x=auto_regressive_df.index,
        y=column,
        label="Auto-Regressive",
    )

    plt.title(f"Comparison of {column}")
    plt.xlabel("Row Index")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.show()


def create_non_auto_df(res, time_series_config):
    numeric_out = res["numeric_out"]
    categorical_out = res["categorical_out"]
    numeric_df = process_results(
        numeric_out, time_series_config.numeric_col_tokens, time_series_config
    )
    categorical_df = process_results(
        categorical_out,
        time_series_config.categorical_col_tokens,
        time_series_config,
    )
    return pd.concat([categorical_df, numeric_df], axis=1)
