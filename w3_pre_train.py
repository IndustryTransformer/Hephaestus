# %% [markdown]
# # Planets Model
#
# ## Load Libs
#

# %%
import ast
import os
import re
from datetime import datetime as dt

import icecream
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from icecream import ic
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import hephaestus as hp
from hephaestus.timeseries_models import tabular_collate_fn

# %%
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("medium")
# %%
icecream.install()
ic_disable = True  # Global variable to disable ic
if ic_disable:
    ic.disable()
ic.configureOutput(includeContext=True, contextAbsPath=True)
# pd.options.mode.copy_on_write = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEQUENCE_LENGTH = 256
# %%
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")  # type: ignore
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
# [PolyMathic](https://polymathic-ai.org/blog/xval/).
# Since we've already transformed the
# data into a parquet file, we can load it directly.
#


# %%
df = pd.read_parquet("data/combined_3w_real_sample.parquet")
df = df.head(100_000)  # Reduce dataset size for debugging
df.drop(
    columns=[
        "original_filename",
        "file_class_label",
        "system_id",
        "timestamp",
        "instance_id",
        "state",
    ],
    inplace=True,
)
# Drop all-NaN columns
df = df.dropna(axis=1, how='all')

# Drop constant columns (no variance)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].nunique() <= 1:
        df = df.drop(columns=[col])
        print(f"Dropped constant column: {col}")

# MAke idx a sequential index of step size SEQUENCE_LENGTH
df["idx"] = np.arange(0, len(df) * SEQUENCE_LENGTH, SEQUENCE_LENGTH)
# Convert time_step to a datetime object
# %%
df.head()

# %%
df_categorical = df.select_dtypes(include=["object"]).astype(str)
unique_values_per_column = df_categorical.apply(
    pd.Series.unique
).values  # .flatten().tolist()
flattened_unique_values = np.concatenate(unique_values_per_column).tolist()
unique_values = list(set(flattened_unique_values))
unique_values  # noqa: B018

# %%
df.select_dtypes(include="object").groupby(
    df.select_dtypes(include="object").columns.tolist()
).size().reset_index(name="count")

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
N_HEADS = 8 * 4
# tabular_decoder = TimeSeriesDecoder(time_series_config, d_model=512, n_heads=N_HEADS)
tabular_decoder = hp.TabularDecoder(
    time_series_config,
    d_model=512,
    n_heads=N_HEADS,
    attention_type="flash",  # Use flash attention for efficiency
)

# %%
logger = TensorBoardLogger("runs", name=f"{dt.now()}_attention_3w_real_sample")
early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
trainer = L.Trainer(max_epochs=5, logger=logger, callbacks=[early_stopping])
train_dl = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    collate_fn=tabular_collate_fn,
    num_workers=1,
    persistent_workers=True,
)
test_dl = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    collate_fn=tabular_collate_fn,
    num_workers=1,
    persistent_workers=True,
)
trainer.fit(tabular_decoder, train_dl, test_dl)

# %%
df_comp = hp.show_results_df(
    model=tabular_decoder,
    time_series_config=time_series_config,
    dataset=train_ds,
    idx=0,
)
