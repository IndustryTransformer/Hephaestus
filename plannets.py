# %% [markdown]
# # Planets Model
#
# ## Load Libs
#

# jax.config.update("jax_disable_jit", False)
import ast
import os
import re
from datetime import datetime as dt

import icecream

# %%
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx
from icecream import ic

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange
from transformers import BertTokenizerFast, FlaxBertModel

import hephaestus as hp
import hephaestus.training as ht

jax.config.update("jax_default_matmul_precision", "float32")

icecream.install()
ic_disable = True  # Global variable to disable ic
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

# %% [markdown]
# ## Load Data
#


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
batch = hp.make_batch(train_ds, 0, 4)
print(batch["numeric"].shape, batch["categorical"].shape)

# %%
multiplier = 1
# n_heads =
# n_heads = max(d_model // 64, 1)  # Scale heads with model size
tabular_decoder = hp.TimeSeriesDecoder(
    time_series_config, d_model=512, n_heads=8 * multiplier, rngs=nnx.Rngs(0)
)
# nnx.display(tabular_decoder)

# %%
res = tabular_decoder(
    numeric_inputs=batch["numeric"],
    categorical_inputs=batch["categorical"],
    deterministic=False,
)

# %%
res["numeric_out"].shape, res["categorical_out"].shape

# %%
if jnp.isnan(res["numeric_out"]).any():
    raise ValueError("NaN in numeric_out")
if jnp.isnan(res["categorical_out"]).any():
    raise ValueError("NaN in categorical_out")

# %%
ic.disable()

# %%
metric_history = ht.create_metric_history()

learning_rate = 4e-3
momentum = 0.9
optimizer = ht.create_optimizer(tabular_decoder, learning_rate, momentum)

metrics = ht.create_metrics()
eval_metrics = ht.create_metrics()
writer_name = "sum-to-float-32"
# Get git commit hash for model name?
writer_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
commit_hash = hp.get_git_commit_hash()
model_name = f"{writer_time}_{writer_name}_{commit_hash}"
writer_path = f"runs/{model_name}"


train_data_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_data_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
train_step = ht.create_train_step(
    model=tabular_decoder, optimizer=optimizer, metrics=metrics
)
eval_step = ht.create_eval_step(model=tabular_decoder, metrics=eval_metrics)

early_stop = None
step = 0
eval_every = 100
N_EPOCHS = 1
summary_writer = SummaryWriter(writer_path)
for epoch in trange(N_EPOCHS):
    for batch in tqdm(train_data_loader):
        if early_stop and step > early_stop:
            break
        batch = {"numeric": jnp.array(batch[0]), "categorical": jnp.array(batch[1])}
        train_step(tabular_decoder, batch, optimizer, metrics)
        for metric, value in metrics.compute().items():
            metric_history[metric].append(value)
            if jnp.isnan(value).any():
                raise ValueError("Nan Values")
            summary_writer.add_scalar(f"train/{metric}", np.array(value), step)
        step += 1
        metrics.reset()
        if step % eval_every == 0:
            for batch in val_data_loader:
                batch = {
                    "numeric": jnp.array(batch[0]),
                    "categorical": jnp.array(batch[1]),
                }
                eval_step(tabular_decoder, batch, eval_metrics)
            for metric, value in eval_metrics.compute().items():
                summary_writer.add_scalar(f"eval/{metric}", np.array(value), step)
            eval_metrics.reset()

# Final evaluation
for batch in val_data_loader:
    batch = {"numeric": jnp.array(batch[0]), "categorical": jnp.array(batch[1])}
    eval_step(tabular_decoder, batch, eval_metrics)
for metric, value in eval_metrics.compute().items():
    summary_writer.add_scalar(f"eval/{metric}", np.array(value), step)
eval_metrics.reset()
summary_writer.close()

# %%
df_comp = hp.show_results_df(
    model=tabular_decoder,
    time_series_config=time_series_config,
    dataset=train_ds,
    idx=0,
)

# %%
df_comp.output_df.loc[:, time_series_config.categorical_col_tokens].head()

# %%
df_comp.output_df.loc[:, time_series_config.numeric_col_tokens].head()

# %%
test_inputs = hp.AutoRegressiveResults.from_ds(train_ds, 0, 13)


for i in trange(21):
    test_inputs = hp.auto_regressive_predictions(tabular_decoder, test_inputs)

# %%
auto_df = hp.create_test_inputs_df(test_inputs, time_series_config)

# %%
auto_df.head(20)

# %%
res1 = tabular_decoder(
    numeric_inputs=jnp.array([train_ds[0][0][:, :10]]),
    categorical_inputs=jnp.array([train_ds[0][1][:, :10]]),
)


res_df = hp.create_non_auto_df(res1, time_series_config)
actual_df = hp.create_non_auto_df(
    {"numeric_out": train_ds[0][0], "categorical_out": train_ds[0][1]},
    time_series_config,
)

# %%
hp.create_non_auto_df(
    {"numeric_out": train_ds[0][0], "categorical_out": train_ds[0][1]},
    time_series_config,
)

# %%
hp.plot_comparison(actual_df, auto_df, res_df, "planet1_x")
