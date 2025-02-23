# %%
import ast
import os
import re
from datetime import datetime as dt

import icecream
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx
from icecream import ic
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm, trange
from transformers import BertTokenizerFast, FlaxBertModel

import hephaestus as hp
import hephaestus.training as ht

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

# Get the embeddings matrix
embeddings = model.params["embeddings"]["word_embeddings"]["embedding"]

# Now you can access specific embeddings like this:
# For example, to get embeddings for tokens 23, 293, and 993:
selected_embeddings = jnp.take(embeddings, jnp.array([23, 293, 993]), axis=0)

# If you want to get embeddings for specific words:
words = ["hello", "world", "example"]
tokens = tokenizer.convert_tokens_to_ids(words)
word_embeddings = jnp.take(embeddings, jnp.array(tokens), axis=0)
word_embeddings.shape


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
# df = df.reset_index(drop=True)
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
def make_batch(ds: hp.TimeSeriesDS, start: int, length: int):
    numeric = []
    categorical = []
    for i in range(start, length + start):
        numeric.append(ds[i][0])
        categorical.append(ds[i][1])
    # print index of None values
    return {"numeric": jnp.array(numeric), "categorical": jnp.array(categorical)}


batch = make_batch(train_ds, 0, 4)
print(batch["numeric"].shape, batch["categorical"].shape)

# (4, 27, 59) (4, 3, 59)
# batch

# %%
multiplier = 1
time_series_regressor = hp.TimeSeriesDecoder(
    time_series_config, d_model=1024, n_heads=8 * multiplier, rngs=nnx.Rngs(0)
)
# nnx.display(time_series_regressor)

# %%
res = time_series_regressor(
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
causal_mask = False
# time_series_regressor.train()

# %%


# %%
metric_history = ht.create_metric_history()

learning_rate = 4e-2
momentum = 0.9
optimizer = ht.create_optimizer(time_series_regressor, learning_rate, momentum)

metrics = ht.create_metrics()
writer_name = "LowerRate"
# Get git commit hash for model name?
writer_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
commit_hash = hp.get_git_commit_hash()
model_name = f"{writer_time}_{writer_name}_{commit_hash}"
summary_writer = SummaryWriter("runs/" + model_name)


train_data_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
train_step = ht.create_train_step(
    model=time_series_regressor, optimizer=optimizer, metrics=metrics
)
step_counter = 0
for epoch in trange(5):
    for step, batch in enumerate(tqdm(train_data_loader)):
        batch = {"numeric": jnp.array(batch[0]), "categorical": jnp.array(batch[1])}
        train_step(time_series_regressor, batch, optimizer, metrics)
        for metric, value in metrics.compute().items():
            # Only shows `loss`

            metric_history[metric].append(value)
            if jnp.isnan(value).any():
                raise ValueError("Nan Values")
            summary_writer.add_scalar(f"train/{metric}", np.array(value), step_counter)
            step_counter += 1
        metrics.reset()

# %%
# import orbax.checkpoint as ocp

# ckpt_dir = ocp.test_utils.erase_and_create_empty("/tmp/my-checkpoints1/")
# _, state = nnx.split(time_series_regressor)
# state = state.to_pure_dict()
# # nnx.display(state)

# checkpointer = ocp.StandardCheckpointer()
# checkpointer.save(ckpt_dir / "state", state)

# %%
x = hp.return_results(time_series_regressor, train_ds, 0)
x.categorical_out.shape

# %%
causal_mask = False
causal_mask = False


df_comp = hp.show_results_df(
    model=time_series_regressor,
    time_series_config=time_series_config,
    dataset=train_ds,
    idx=0,
)

# %%
df_comp.output_df.loc[:, time_series_config.categorical_col_tokens].tail()

# %%
df_comp.output_df.loc[:, time_series_config.numeric_col_tokens].tail()

# %%
test_inputs = hp.AutoRegressiveResults.from_ds(train_ds, 0, 13)

# inputs_test = train_ds[0]
# test_numeric = inputs_test[0]
# test_categorical = inputs_test[1]
# print(inputs_test.shape)
for i in trange(21):
    test_inputs = hp.auto_regressive_predictions(time_series_regressor, test_inputs)

# x = auto_regressive_predictions(state, test_ds[0], 10)

# %%
auto_df = hp.create_test_inputs_df(test_inputs, time_series_config)

# %%
auto_df.tail()

# %% [markdown]
#

# %%
res1 = time_series_regressor(
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
hp.plot_comparison(actual_df, auto_df, res_df, "planet0_x")

# %%
test_inputs.numeric_inputs.shape

# %%
# TODO This appears to only work if the model size is 1024

# %%


# %%


# %%
# Error here
#     "Categorical after swap": 'Categorical after swap'
#     categorical_out.shape: (4, 3, 59, 41)
#   0%|          | 0/5 [00:00<?, ?it/s]
#   0%|          | 0/6250 [00:00<?, ?it/s]
#   0%|          | 0/6250 [00:00<?, ?it/s]
#   0%|          | 0/6250 [00:00<?, ?it/s]
#   0%|          | 0/6250 [00:00<?, ?it/s]
#   0%|          | 0/6250 [00:00<?, ?it/s]
#   0%|          | 0/21 [00:00<?, ?it/s]
# Expanding dims
# Traceback (most recent call last):
#   File "/home/ubuntu/environment/Hephaestus/plannets.py", line 274, in <module>
#     test_inputs = hp.auto_regressive_predictions(time_series_regressor, test_inputs)
#                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/environment/Hephaestus/hephaestus/analysis/analysis.py", line 133, in auto_regressive_predictions
#     final_categorical_row = np.array(categorical_out[:, :, -1])
#                                      ~~~~~~~~~~~~~~~^^^^^^^^^^
#   File "/home/ubuntu/environment/Hephaestus/.venv/lib/python3.11/site-packages/jax/_src/array.py", line 370, in __getitem__
#     return lax_numpy._rewriting_take(self, idx)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/environment/Hephaestus/.venv/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 11411, in _rewriting_take
#     return _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/environment/Hephaestus/.venv/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 11420, in _gather
#     indexer = _index_to_gather(shape(arr), idx)  # shared with _scatter_update
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/environment/Hephaestus/.venv/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 11528, in _index_to_gather
#     idx = _canonicalize_tuple_index(len(x_shape), idx)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/environment/Hephaestus/.venv/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 11852, in _canonicalize_tuple_index
#     raise IndexError(
# IndexError: Too many indices: 2-dimensional array indexed with 3 regular indices.
# :~/environment/Hephaestus $
