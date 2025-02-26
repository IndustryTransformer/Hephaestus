# %%
import ast
import os
import re

import icecream
import numpy as np
import pandas as pd
import torch

# Add these imports
from icecream import ic

# from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from transformers import BertTokenizerFast, FlaxBertModel

from hephaestus.models.models import TimeSeriesConfig, TimeSeriesDecoder, TimeSeriesDS

# %%
print("Hello")
# %%
icecream.install()
ic_disable = False  # Global variable to disable ic
if ic_disable:
    ic.disable()
ic.configureOutput(includeContext=True, contextAbsPath=True)
pd.options.mode.copy_on_write = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
model = FlaxBertModel.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)


# %%
def line2df(line, idx):
    data_rows = []
    line = ast.literal_eval(line)
    for i, time_step in enumerate(line["data"]):
        row = {"time_step": i}
        # Add position data for each planet
        for j, position in enumerate(time_step):
            row[f"planet{j}_x"] = position[0]
            row[f"planet{j}_y"] = position[1]
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    description = line.pop("description")
    step_size = description.pop("stepsize")
    for k, v in description.items():
        for k_prop, v_prop in v.items():
            df[f"{k}_{k_prop}"] = v_prop
    df["time_step"] = df["time_step"] * step_size
    df.insert(0, "idx", idx)

    return df


# %%
files = os.listdir("data")
if "planets.parquet" not in files:
    with open("data/planets.data") as f:
        data = f.read().splitlines()

        dfs = []
        for idx, line in enumerate(tqdm(data)):
            dfs.append(line2df(line, idx))
        df = pd.concat(dfs)
    df.to_parquet("data/planets.parquet")
else:
    df = pd.read_parquet("data/planets.parquet")


# Combine total mass of all planets into one column `planet<n>_m`
mass_regex = re.compile(r"planet(\d+)_m")
mass_cols = [col for col in df.columns if mass_regex.match(col)]
df["total_mass"] = df[mass_cols].sum(axis=1)
# Introduce categorical columns for the number of planets choose non null columns with mass
df["n_planets"] = df[mass_cols].notnull().sum(axis=1).astype("object")
df["n_planets"] = df["n_planets"].apply(lambda x: f"{x}_planets")
# Create category acceleration if the sum of plane/d_[x,y, z] is greater than 0
df["acceleration_x"] = df[
    [col for col in df.columns if "planet" in col and "_x" in col]
].sum(axis=1)
# Set acceleration_x to "increasing" if greater than 0 else "decreasing"
df["acceleration_x"] = (
    df["acceleration_x"]
    .apply(lambda x: "increasing" if x > 0 else "decreasing")
    .astype("object")
)
df["acceleration_y"] = df[
    [col for col in df.columns if "planet" in col and "_y" in col]
].sum(axis=1)
df["acceleration_y"] = df["acceleration_y"].apply(
    lambda x: "increasing" if x > 0 else "decreasing"
)


df.describe()

# %%
df.head()

# %%
df_categorical = df.select_dtypes(include=["object"]).astype(str)
unique_values_per_column = df_categorical.apply(
    pd.Series.unique
).values  # .flatten().tolist()
flattened_unique_values = np.concatenate(unique_values_per_column).tolist()
unique_values = list(set(flattened_unique_values))
unique_values

# %%
df.select_dtypes(include="object").groupby(
    df.select_dtypes(include="object").columns.tolist()
).size().reset_index(name="count")

# %%
df = df.reset_index(drop=True)

# %%
time_series_config = TimeSeriesConfig.generate(df=df)

# %%
time_series_config = TimeSeriesConfig.generate(df=df)
train_idx = int(df.idx.max() * 0.8)
train_df = df.loc[df.idx < train_idx].copy()
test_df = df.loc[df.idx >= train_idx].copy()
# del df
train_ds = TimeSeriesDS(train_df, time_series_config)
test_ds = TimeSeriesDS(test_df, time_series_config)
len(train_ds), len(test_ds)

# %%
train_ds[0]


def make_batch(ds: TimeSeriesDS, start: int, length: int):
    """Create a batch of data from the dataset.

    Args:
        ds (hp.TimeSeriesDS): The dataset to create a batch from.
        start (int): The starting index of the batch.
        length (int): The length of the batch.

    Returns:
        dict: Dictionary containing numeric and categorical data.
    """
    numeric = []
    categorical = []
    for i in range(start, length + start):
        numeric.append(ds[i][0])
        categorical.append(ds[i][1])

    # Convert to numpy arrays first to avoid the warning
    numeric_array = np.array(numeric)
    categorical_array = np.array(categorical)

    return {
        "numeric": torch.tensor(numeric_array, dtype=torch.float32),
        "categorical": torch.tensor(categorical_array, dtype=torch.float32),
    }


# %%
tabular_decoder = TimeSeriesDecoder(time_series_config, d_model=512, n_heads=8)


# After creating the model, add debugging function
def check_model_for_nans(model):
    """Check for NaNs in model parameters"""
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in parameter {name}")
            has_nan = True
    return has_nan


# %%
# Create a sample batch from the training dataset (batch size = 1)
example_batch = make_batch(train_ds, 0, 1)
numeric_data = example_batch["numeric"]
categorical_data = example_batch["categorical"]

# Optionally move tensors to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numeric_data = numeric_data.to(device)
categorical_data = categorical_data.to(device)
tabular_decoder.to(device)

# Add this just before the prediction
print("Checking model parameters for NaNs:")
has_nan_params = check_model_for_nans(tabular_decoder)
if has_nan_params:
    print("NaNs found in model parameters. Reinitializing...")
    # Reinitialize the model to try to resolve NaN issues
    tabular_decoder = TimeSeriesDecoder(time_series_config, d_model=512, n_heads=8)
    tabular_decoder.to(device)

# Before making a prediction, do a sanity check on inputs
print("Input shapes:", numeric_data.shape, categorical_data.shape)
print("Input contains NaN (numeric):", torch.isnan(numeric_data).any().item())
print("Input contains NaN (categorical):", torch.isnan(categorical_data).any().item())

# Make a prediction using the model
with torch.no_grad():
    # Use try/except to catch and debug any runtime errors
    try:
        prediction = tabular_decoder(numeric_data, categorical_data)

        # Check if prediction contains NaNs
        if any(torch.isnan(tensor).any() for tensor in prediction.values()):
            print("Warning: NaNs in prediction. Applying nan_to_num...")
            for key in prediction:
                prediction[key] = torch.nan_to_num(prediction[key], nan=0.0)
    except RuntimeError as e:
        print(f"Error during prediction: {e}")
        # Try with smaller batch or simpler model configuration
        # For debugging only
        prediction = {
            "numeric": torch.zeros_like(numeric_data[:, :, 0]),
            "categorical": torch.zeros_like(categorical_data[:, :, 0]),
        }

# Print prediction summary instead of all values
print("Prediction numeric shape:", prediction["numeric"].shape)
print("Prediction categorical shape:", prediction["categorical"].shape)
print(
    "Prediction contains NaN (numeric):",
    torch.isnan(prediction["numeric"]).any().item(),
)
print(
    "Prediction contains NaN (categorical):",
    torch.isnan(prediction["categorical"]).any().item(),
)
# %%
