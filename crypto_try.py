# %%
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
import hephaestus.training.training as ht

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
df = pd.read_csv("data/HeliusData.csv")
# Lower case column names and replace spaces with underscores
# df.drop(columns=["json_info"], inplace=True)
# df.columns = df.columns.str.lower().str.replace(" ", "_")
df.drop(columns=["description", "is_current"], inplace=True)
# create idx column for every 10 rows incrementing by 1 so the first 10 rows will have idx 0, the next 10 rows will have idx 1, and so on
df["idx"] = df.index // 10


# df.to_csv("data/HeliusData.csv", index=False)
# %%

# %%
df.head(20)


# %%
def simplify_hashes(df, hash_columns, description_column=None):
    """
    Simplifies hash values in specified columns and within description text by replacing them
    with numbered versions (hash1, hash2, etc.) within each group.

    Parameters:
    df (pandas.DataFrame): Input DataFrame with an 'idx' column for grouping
    hash_columns (list): List of column names containing hash values
    description_column (str): Optional column name containing text with embedded hashes

    Returns:
    pandas.DataFrame: DataFrame with simplified hash values
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # Process each group separately
    for group_id in result_df["idx"].unique():
        group_mask = result_df["idx"] == group_id

        # Collect all unique hash values across all hash columns in this group
        all_hashes = set()

        # Get hashes from regular columns
        for col in hash_columns:
            values = result_df.loc[group_mask, col].fillna("NaN").astype(str)
            valid_values = values[values != "NaN"].unique()
            all_hashes.update(valid_values)

        # Get hashes from description text if specified
        if description_column:
            descriptions = result_df.loc[group_mask, description_column].fillna("")
            # Find all words that look like hashes (base58/base64 strings)
            for desc in descriptions:
                if isinstance(desc, str):
                    # Match strings that could be hashes (alphanumeric, typically 32+ chars)
                    hash_pattern = r"\b[A-Za-z0-9]{32,}\b"
                    found_hashes = re.findall(hash_pattern, desc)
                    all_hashes.update(found_hashes)

        # Create mapping for all hash values in this group
        hash_mapping = {
            hash_val: f"hash{i+1}" for i, hash_val in enumerate(sorted(all_hashes))
        }
        hash_mapping["NaN"] = np.nan

        # Apply mapping to regular hash columns
        for col in hash_columns:
            temp_series = result_df.loc[group_mask, col].fillna("NaN").astype(str)
            result_df.loc[group_mask, col] = temp_series.map(hash_mapping)

        # Apply mapping to description text
        if description_column:

            def replace_hashes_in_text(text):
                if pd.isna(text):
                    return text
                result = text
                for old_hash, new_hash in hash_mapping.items():
                    if old_hash != "NaN":
                        result = result.replace(old_hash, new_hash)
                return result

            result_df.loc[group_mask, description_column] = result_df.loc[
                group_mask, description_column
            ].apply(replace_hashes_in_text)

    return result_df


# %%
df = simplify_hashes(
    df,
    [
        "wallet_id",
        "wallet_address",
        "transaction_hash",
        "in_from",
        "id",
        "transaction_detail_id",
        "fee_paid_by",
    ],
)

# %%
# %%


def enrich_datetimes(df: pd.DataFrame, col: str):
    df[col] = pd.to_datetime(df[col], utc=True)

    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
    df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)
    df[f"{col}_day_sin"] = np.sin(2 * np.pi * df[col].dt.day / 31)
    df[f"{col}_day_cos"] = np.cos(2 * np.pi * df[col].dt.day / 31)
    df[f"{col}_hour_sin"] = np.sin(2 * np.pi * df[col].dt.hour / 24)
    df[f"{col}_hour_cos"] = np.cos(2 * np.pi * df[col].dt.hour / 24)
    df[f"{col}_minute_sin"] = np.sin(2 * np.pi * df[col].dt.minute / 60)
    df[f"{col}_minute_cos"] = np.cos(2 * np.pi * df[col].dt.minute / 60)
    df[f"{col}_second_sin"] = np.sin(2 * np.pi * df[col].dt.second / 60)
    df[f"{col}_second_cos"] = np.cos(2 * np.pi * df[col].dt.second / 60)

    return df.drop(columns=[col])


# %%
df = enrich_datetimes(df, "date")
# %%
df.head()
# %%
# %%
# Get train test split at 80/20
time_series_config = hp.TimeSeriesConfig.generate(df=df)
train_idx = int(df.idx.max() * 0.8)
train_df = df.loc[df.idx < train_idx].copy()
test_df = df.loc[df.idx >= train_idx].copy()
# del df
train_ds = hp.TimeSeriesDS(train_df, time_series_config)
test_ds = hp.TimeSeriesDS(test_df, time_series_config)


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
# %%
multiplier = 4
time_series_regressor = hp.TimeSeriesDecoder(
    time_series_config, d_model=512, n_heads=8 * multiplier, rngs=nnx.Rngs(0)
)
# nnx.display(time_series_regressor)
# %%
res = time_series_regressor(
    numeric_inputs=batch["numeric"],
    categorical_inputs=batch["categorical"],
    deterministic=False,
)
# %%
ic.disable()
causal_mask = True

metric_history = ht.create_metric_history()

learning_rate = 1e-3
momentum = 0.9
optimizer = ht.create_optimizer(time_series_regressor, learning_rate, momentum)

metrics = ht.create_metrics()
writer_name = "CryptoTest1"

writer_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
model_name = writer_time + writer_name
summary_writer = SummaryWriter("runs/" + model_name)


train_data_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
train_step = ht.create_train_step(
    model=time_series_regressor, optimizer=optimizer, metrics=metrics
)

# %%
train_len = len(train_data_loader)
for epoch in trange(10):
    for step, batch in enumerate(tqdm(train_data_loader)):
        batch = {"numeric": jnp.array(batch[0]), "categorical": jnp.array(batch[1])}
        train_step(time_series_regressor, batch, optimizer, metrics)
        for metric, value in metrics.compute().items():
            # Only shows `loss`

            metric_history[metric].append(value)
            if jnp.isnan(value).any():
                raise ValueError("Nan Values")
            summary_writer.add_scalar(
                f"train/{metric}", np.array(value), step + epoch * train_len
            )
        metrics.reset()

# %%
