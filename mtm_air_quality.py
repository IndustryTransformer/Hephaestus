# %% [markdown]
# # MTM Air Quality
#
# ## Load Libs
#

# %%
import os
from datetime import datetime as dt

import icecream
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx
from icecream import ic
from sklearn.preprocessing import StandardScaler

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange
from transformers import BertTokenizerFast, FlaxBertModel

import hephaestus as hp
import hephaestus.training as ht

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
# ## Load and Process Data
#

# %%
# Load and preprocess the dataset (assuming you have a CSV file)
# Select numeric columns

csvs = [
    os.path.join("./data/air_quality/", f)
    for f in os.listdir("./data/air_quality/")
    if f.endswith(".csv")
]
dfs = [pd.read_csv(csv) for csv in csvs]
df = pd.concat(dfs, ignore_index=True)
del dfs
time_cols = ["year", "month", "day", "hour"]
df = df.sort_values(time_cols).reset_index(drop=True).drop("No", axis=1)
# Convert time columns to strings
for col in time_cols:
    df[col] = df[col].astype(str)
# replace . and lower case column names
df.columns = [c.replace(".", "_").lower() for c in df.columns]
# df = df.dropna()
df_no_na = df.dropna()
print(df.shape)
df.dropna(subset=["pm2_5"], inplace=True)
print(df.shape)
df = df.reset_index(drop=True)

df["idx"] = df.index // 32
# df = df.drop(["year", "month", "day", "hour"], axis=1)
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
numeric_cols.remove("idx")  # Remove idx column from scaling

# Create and fit scaler
scale_data = True
if scale_data:
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df.head()

# %%
df_categorical = df.select_dtypes(include=["object"]).astype(str)
unique_values_per_column = df_categorical.apply(
    pd.Series.unique
).values  # .flatten().tolist()
flattened_unique_values = np.concatenate(unique_values_per_column).tolist()
unique_values = list(set(flattened_unique_values))
unique_values

# %% [markdown]
# ## Initialize Model
#

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

# (4, 27, 59) (4, 3, 59)
# batch

# %%
multiplier = 8
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
ic.disable()

# %%
metric_history = ht.create_metric_history()

learning_rate = 0.001
momentum = 0.9
optimizer = ht.create_optimizer(tabular_decoder, learning_rate, momentum)

metrics = ht.create_metrics()
eval_metrics = ht.create_metrics()

writer_name = "mtm-tabular-longer-run-50"
# Get git commit hash for model name?
writer_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
commit_hash = hp.get_git_commit_hash()
model_name = f"{writer_time}_{writer_name}_{commit_hash}"
writer_path = f"runs/{model_name}"


train_data_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_data_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

train_step = ht.create_train_step(
    model=tabular_decoder, optimizer=optimizer, metrics=metrics
)
eval_every = 100
eval_step = ht.create_eval_step(model=tabular_decoder, metrics=eval_metrics)
early_stop = None
step = 0
N_EPOCHS = 50
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
hp.plot_comparison(actual_df, auto_df, res_df, "dewp")

# %%
