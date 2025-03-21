# %% [markdown]
# # Turbine Model
#
# ## Load Libs
#

# %%
import glob
import os
from datetime import datetime as dt
from pathlib import Path

import icecream
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from icecream import ic
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import hephaestus as hp
from hephaestus.timeseries_models import tabular_collate_fn

# %%
torch.set_default_dtype(torch.float32)
# %%
icecream.install()
ic_disable = True  # Global variable to disable ic
if ic_disable:
    ic.disable()
ic.configureOutput(includeContext=True, contextAbsPath=True)
# pd.options.mode.copy_on_write = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("MPS available")
else:
    print("CUDA not available. Checking why...")
    import os

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# %% [markdown]
# ## Load Data
#
# We use `line2df` to load the original data from
# [PolyMathic](https://polymathic-ai.org/blog/xval/). Since we've already transformed the
# data into a parquet file, we can load it directly.
#

# %%
# Read all csv files in data/nox
# Get all csv files in the directory
csv_files = glob.glob("data/nox/*.csv")

# Read and combine all files
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    filename = Path(file).stem  # Get filename without extension
    temp_df["filename"] = filename
    df_list.append(temp_df)

# Concatenate all dataframes
df = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(csv_files)} files with total {len(df)} rows")
# %%

df.columns = df.columns.str.lower()

scaler = StandardScaler()
df["cat_column"] = "category"

# Create more explicit is_odd column
df["is_odd"] = np.where(df.index % 2 == 0, "even", "odd")

# Add an integer representation alongside categorical for easier learning
df["is_odd_numeric"] = df.index % 2  # 0 for even, 1 for odd


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df["idx"] = df.index // 64
print(f"Dataframe shape: {df.shape}")
print(f"First 10 rows of is_odd column: {df['is_odd'].head(10).tolist()}")
df.head()


# %%
# df_categorical = df.select_dtypes(include=["object"]).astype(str)
# unique_values_per_column = df_categorical.apply(
#     pd.Series.unique
# ).values  # .flatten().tolist()
# flattened_unique_values = np.concatenate(unique_values_per_column).tolist()
# unique_values

# %%
# df.select_dtypes(include="object").groupby(
#     df.select_dtypes(include="object").columns.tolist()
# ).size().reset_index(name="count")

# %%
df = df.reset_index(drop=True)
# df = df.head(
#     5_000
# )  # For testing purposes only, train on the full dataset in production

# %%
# Get train test split at 80/20
time_series_config = hp.TimeSeriesConfig.generate(df=df)
train_idx = int(df.idx.max() * 0.8)
train_df = df.loc[df.idx < train_idx].copy()
test_df = df.loc[df.idx >= train_idx].copy()
# del df
train_ds = hp.TimeSeriesDS(train_df, time_series_config)
test_ds = hp.TimeSeriesDS(test_df, time_series_config)
len(train_ds), len(test_ds)

# %%
time_series_config.n_columns

# %%
N_HEADS = 16
BATCH_SIZE = 128
D_MODEL = 64
# tabular_decoder = TimeSeriesDecoder(time_series_config, d_model=512, n_heads=N_HEADS)
tabular_decoder = hp.TabularDecoder(
    time_series_config, d_model=D_MODEL, n_heads=N_HEADS
)

# %%
logger_variant_name = "NOXSmall"
logger_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
logger_name = f"{logger_time}_{logger_variant_name}"
print(f"Using logger name: {logger_name}")
logger = TensorBoardLogger(
    "runs",
    name=logger_name,
)
early_stopping = EarlyStopping(monitor="val_loss", patience=20, mode="min")
trainer = L.Trainer(
    max_epochs=200, logger=logger, callbacks=[early_stopping], log_every_n_steps=1
)
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=tabular_collate_fn,
    # num_workers=7,
    # persistent_workers=True,
)
test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=tabular_collate_fn,
    # num_workers=7,
    # persistent_workers=True,
)
trainer.fit(tabular_decoder, train_dl, test_dl)

# %%
# After training, analyze how well the model learned the alternating pattern
sample_idx = 0  # Choose a test sample for analysis
batch = test_dl.collate_fn(
    [test_ds[sample_idx]]
)  # Use the collate function to get proper format

# Run a forward pass
with torch.no_grad():
    outputs = tabular_decoder(batch)

# Find the is_odd column index
is_odd_idx = None
for i, col in enumerate(time_series_config.categorical_col_tokens):
    if "is_odd" in col:
        is_odd_idx = i
        break

if is_odd_idx is not None:
    # Get the model's predictions for the is_odd column
    pred_logits = outputs.categorical[:, is_odd_idx]
    # Get the predicted class (index of highest probability)
    pred_classes = pred_logits.argmax(dim=-1).cpu().numpy()[0]
    # Get true classes
    true_classes = batch.categorical[:, is_odd_idx].long().cpu().numpy()[0]

    # Print them for comparison
    print("\nAnalyzing is_odd pattern learning:")
    print(f"Predicted sequence: {pred_classes[:20]}")
    print(f"True sequence:      {true_classes[:20]}")

    # Check how many are alternating
    alternating_pred = 0
    alternating_true = 0
    for i in range(1, len(pred_classes)):
        if pred_classes[i] != pred_classes[i - 1]:
            alternating_pred += 1
        if true_classes[i] != true_classes[i - 1]:
            alternating_true += 1

    print(f"Predicted alternating rate: {alternating_pred/(len(pred_classes)-1):.2f}")
    print(f"True alternating rate:      {alternating_true/(len(true_classes)-1):.2f}")

# %%
df_comp = hp.show_results_df(
    model=tabular_decoder,
    time_series_config=time_series_config,
    dataset=test_ds,
    idx=0,
)

# %%
col_name = "nox"
hp.plot_col_error(df_comp, col_name)

# %%
hp.plot_col_comparison(df_comp, col_name)
# %%
hp.plot_col_comparison(df_comp, "is_odd_numeric")
# %%
# Create initial inputs from the test dataset
test_idx = 5  # Choose a test sample
stop_idx = 10  # Start predictions after this timestep
inputs = hp.AutoRegressiveResults.from_ds(test_ds, test_idx, stop_idx=stop_idx)

# Generate predictions for multiple steps
num_prediction_steps = 20
for _ in range(num_prediction_steps):
    inputs = hp.auto_regressive_predictions(tabular_decoder, inputs)

# Process the results into DataFrames for analysis
actual_inputs = test_ds[test_idx]
actual_df = hp.create_test_inputs_df(
    hp.AutoRegressiveResults(actual_inputs.numeric, actual_inputs.categorical),
    time_series_config,
)
predicted_df = hp.create_test_inputs_df(inputs, time_series_config)

# Plot comparisons for several key columns
# Create a 2x2 grid of plots for different planet positions
plt.figure(figsize=(16, 12))
columns_to_plot = numeric_cols

for i, col in enumerate(columns_to_plot):
    # Calculate position in a 2-column grid with dynamic number of rows
    plt.subplot(
        len(columns_to_plot) // 2 + (1 if len(columns_to_plot) % 2 else 0), 2, i + 1
    )

    # Only plot from stop_idx onward
    ax = plt.gca()

    # Use sequence_position as x-axis for better visualization
    ax.plot(
        predicted_df[col],
        "r-",
        label="Predicted",
        linewidth=2,
    )
    ax.plot(
        actual_df[col],
        "b--",
        label="Actual",
        linewidth=2,
    )

    # Add vertical line to mark prediction start
    ax.axvline(
        # x=predicted_df["sequence_position"][stop_idx],
        x=stop_idx,
        color="gray",
        linestyle="--",
        alpha=0.7,
    )

    # Highlight the oscillating pattern more clearly

    ax.set_title(f"{col} Auto-Regressive Predictions")

    ax.set_xlabel("Time Step")
    ax.set_ylabel(f"{col} Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

plt.suptitle("Autoregressive Predictions vs Actual Values", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the suptitle
plt.show()

# %%
df_comp2 = hp.show_results_df(
    model=tabular_decoder,
    time_series_config=time_series_config,
    dataset=test_ds,
    idx=test_idx,
    mask_start=stop_idx,
)
df_comp2.input_df = df_comp2.input_df.iloc[:]

# %%
