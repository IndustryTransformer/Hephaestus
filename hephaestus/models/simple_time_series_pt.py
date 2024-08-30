# %%
# import jax
import math
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def split_complex_word(word):
    # Step 1: Split by underscore, preserving content within square brackets
    parts = re.split(r"(_|\[.*?\])", word)
    parts = [p for p in parts if p]  # Remove empty strings

    # Step 2: Split camelCase for parts not in square brackets
    def split_camel_case(s):
        if s.startswith("[") and s.endswith("]"):
            return [s]
        return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", s)

    # Step 3: Apply camelCase splitting to each part and flatten the result
    result = [item for part in parts for item in split_camel_case(part)]

    return result


def convert_object_to_int_tokens(df, token_dict):
    """Converts object columns to integer tokens using a token dictionary."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(token_dict)
    return df


def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


# def create_padding_mask(self, tensor, pad_idx=0):
#     return (tensor == pad_idx).transpose(0, 1)


class SimpleDS(Dataset):
    def __init__(self, df):
        # Add nan padding to make sure all sequences are the same length
        # use the idx column to group by
        self.max_seq_len = df.groupby("idx").count().time_step.max()
        # Set df.idx to start from 0
        df.idx = df.idx - df.idx.min()
        df = df.set_index("idx")
        df.index.name = None

        self.df_categorical = df.select_dtypes(include=["object"]).astype(str)
        self.df_numeric = df.select_dtypes(include="number")
        self.batch_size = self.max_seq_len
        self.numeric_token = "[NUMERIC_EMBEDDING]"
        self.special_tokens = [
            "[PAD]",
            "[NUMERIC_MASK]",
            "[MASK]",
            "[UNK]",
            self.numeric_token,
        ]
        self.numeric_mask = "[NUMERIC_MASK]"
        # Remove check on idx
        self.numeric_col_tokens = [
            col_name for col_name in df.select_dtypes(include="number").columns
        ]
        self.categorical_col_tokens = [
            col_name for col_name in df.select_dtypes(include="object").columns
        ]
        # Get all the unique values in the categorical columns and add them to the tokens
        self.object_tokens = (
            self.df_categorical.apply(pd.Series.unique).values.flatten().tolist()
        )
        self.tokens = (
            self.special_tokens
            + self.numeric_col_tokens
            + self.object_tokens
            + self.categorical_col_tokens
        )

        self.token_dict = {token: i for i, token in enumerate(self.tokens)}
        self.token_decoder_dict = {i: token for i, token in enumerate(self.tokens)}
        self.n_tokens = len(self.tokens)
        self.numeric_indices = torch.tensor(
            [self.tokens.index(i) for i in self.numeric_col_tokens]
        )
        self.categorical_indices = torch.tensor(
            [self.tokens.index(i) for i in self.categorical_col_tokens]
        )

        self.numeric_mask_token = self.tokens.index(self.numeric_mask)
        # Make custom vocab by splitting on snake case, camel case, spaces and numbers
        reservoir_vocab = [split_complex_word(word) for word in self.token_dict.keys()]
        # flatten the list, make a set and then list again
        self.reservoir_vocab = list(
            set([item for sublist in reservoir_vocab for item in sublist])
        )
        # Get reservoir embedding tokens
        reservoir_tokens_list = [
            self.token_decoder_dict[i] for i in range(len(self.token_decoder_dict))
        ]  # ensures they are in the same order
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.reservoir_encoded = self.tokenizer(
            reservoir_tokens_list,
            padding="max_length",
            max_length=8,  # TODO Make this dynamic
            truncation=True,
            return_tensors="jax",
            add_special_tokens=False,
        )[
            "input_ids"
        ]  # TODO make this custom to reduce dictionary size

        self.df_categorical = convert_object_to_int_tokens(
            self.df_categorical, self.token_dict
        )

    def __len__(self):
        # return self.df.idx.max() + 1  # probably should be max idx + 1 thanks
        return self.df_numeric.index.nunique()

    def get_data(self, df_name, set_idx):
        """Gets self.df_<df_name> for a given index"""
        df = getattr(self, df_name)

        batch = df.loc[df.index == set_idx, :]
        batch = np.array(batch.values)

        # Add padding

        batch_len, n_cols = batch.shape
        pad_len = self.max_seq_len - batch_len
        padding = np.full((pad_len, n_cols), np.nan)
        batch = np.concatenate([batch, padding], axis=0)
        batch = np.swapaxes(batch, 0, 1)
        # if df_name == "df_categorical":
        #     # Cast to int
        #     batch = batch.astype(int)
        return batch

    def __getitem__(self, set_idx):
        if self.df_categorical.empty:
            categorical_inputs = None
        else:
            categorical_inputs = self.get_data("df_categorical", set_idx)
        numeric_inputs = self.get_data("df_numeric", set_idx)

        return numeric_inputs, categorical_inputs


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, deterministic: bool = False):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout1(x) if not deterministic else x
        x = self.linear2(x)
        x = self.dropout2(x) if not deterministic else x
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        residual = q
        q = self.layernorm1(q)
        q, _ = self.self_attention(q, k, v, attn_mask=mask)
        q += residual

        residual = q
        q = self.layernorm2(q)
        q = self.ffn(q, deterministic=deterministic)
        q += residual

        return q


class ReservoirEmbedding(nn.Module):
    """
    A module for performing reservoir embedding on a given input.

    Args:
        dataset (SimpleDS): The dataset used for embedding.
        features (int): The number of features in the embedding.
        frozen_index (int, optional): The index of the embedding to freeze. Defaults to 0.
    """

    def __init__(self, dataset: SimpleDS, features: int):
        super().__init__()
        self.embedding = nn.Embedding(dataset.n_tokens, features)

    def forward(self, base_indices: torch.Tensor):
        ic(base_indices.shape)
        reservoir_encoded = self.embedding(base_indices)
        ic(reservoir_encoded.shape)
        return reservoir_encoded  # .sum(axis=-1) # TODO Make this sum over sub dims


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        dataset: SimpleDS,
        d_model: int = 64,
        n_heads: int = 4,
        time_window: int = 10_000,
    ):
        super().__init__()
        self.dataset = dataset
        self.d_model = d_model
        self.n_heads = n_heads
        self.time_window = time_window
        self.embedding = ReservoirEmbedding(
            self.dataset,
            features=self.d_model,
        )
        self.positional_encoding = PositionalEncoding(
            max_len=self.time_window, d_model=self.d_model
        )
        self.transformer_block_1 = TransformerBlock(
            d_model=self.d_model, num_heads=self.n_heads, d_ff=256, dropout_rate=0.1
        )

    def forward(
        self,
        numeric_inputs,
        categorical_inputs,
        deterministic=False,
        mask_data: bool = True,
    ):
        nan_mask = torch.isnan(numeric_inputs)
        # replace nan with 0
        numeric_inputs = torch.where(nan_mask, 0.0, numeric_inputs)
        # Cat Embedding
        if categorical_inputs is not None:
            categorical_inputs = torch.nan_to_num(
                categorical_inputs,
                torch.tensor(self.dataset.token_dict["[NUMERIC_MASK]"]),
            )
            # Make sure nans are set to <NAN> token

            categorical_embeddings = self.embedding(categorical_inputs)

        else:
            categorical_embeddings = None

        # Numeric Embedding
        # repeated_numeric_indices
        repeated_numeric_indices = self.dataset.numeric_indices.repeat(
            numeric_inputs.shape[2], 1
        )

        # repeated_numeric_indices = jnp.swapaxes(repeated_numeric_indices, 0, 1)
        repeated_numeric_indices = repeated_numeric_indices.T
        ic("Numeric Pre embedding", repeated_numeric_indices.shape)
        numeric_col_embeddings = self.embedding(repeated_numeric_indices)
        ic("Numeric Post embedding", numeric_col_embeddings.shape)
        # Nan Masking
        numeric_col_embeddings = numeric_col_embeddings.unsqueeze(0).expand(
            numeric_inputs.shape[0], -1, -1, -1
        )
        if categorical_embeddings is not None:
            categorical_col_embeddings = self.dataset.categorical_indices.repeat(
                categorical_inputs.shape[2], 1
            )
            # Delete second dimension as it's empty

            repeated_categorical_indices = categorical_col_embeddings.unsqueeze(
                0
            ).expand(categorical_inputs.shape[0], -1, -1, -1)
            repeated_categorical_indices = repeated_categorical_indices.mT
            ic("Categorical Pre embedding", repeated_categorical_indices.shape)
            categorical_col_embeddings = self.embedding(repeated_categorical_indices)
            ic("Categorical Post embedding", categorical_col_embeddings.shape)
            # categorical_col_embeddings = categorical_col_embeddings.unsqueeze(0).expand(
            #     categorical_inputs.shape[0], -1, -1, -1
            # )
        else:
            categorical_col_embeddings = None
        numeric_col_tokens = torch.tensor(
            self.dataset.token_dict[self.dataset.numeric_token]
        )
        ic("Col tokens", numeric_col_tokens.shape)
        numeric_embedding = self.embedding(numeric_col_tokens)

        numeric_broadcast = numeric_inputs[:, :, :, None] * numeric_embedding

        numeric_broadcast = torch.where(
            # nan_mask,
            # jnp.expand_dims(nan_mask, axis=-1),
            nan_mask[:, :, :, None],
            self.embedding(torch.tensor(self.dataset.numeric_mask_token)),
            numeric_broadcast,
        )
        # End Nan Masking
        # ic(f"Nan values in out: {jnp.isnan(numeric_broadcast).any()}")

        # ic(f"Nan values in out positional: {jnp.isnan(numeric_broadcast).any()}")
        # ic("Starting Attention")
        # ic(numeric_broadcast.shape)
        if categorical_embeddings is not None:
            ic(numeric_broadcast.shape, categorical_embeddings.shape)
            ic("Categorical Pre Squueze", categorical_col_embeddings.shape)
            categorical_col_embeddings = torch.squeeze(categorical_col_embeddings, 1)
            ic("Categorical Post Squeeze", categorical_col_embeddings.shape)
            tabular_data = torch.concatenate(
                [numeric_broadcast, categorical_embeddings], dim=1
            )
        else:
            ic("No Categorical Embeddings")
            tabular_data = numeric_broadcast
        if categorical_embeddings is not None:
            ic("Masking for categorical data")
            mask_input = torch.concatenate([numeric_inputs, categorical_inputs], dim=1)

        else:
            ic("No Masking for categorical data")
            mask_input = numeric_inputs
        ic(mask_input.shape)
        if mask_data:
            mask = create_causal_mask(mask_input.shape[1])
            # causal_mask = nn.make_causal_mask(mask_input)
            # pad_mask = nn.make_attention_mask(
            #     mask_input, mask_input
            # )  # TODO Add in the mask for the categorical data
            # mask = nn.combine_masks(causal_mask, pad_mask)
            ic(mask.shape)
        else:
            mask = None
        pos_dim = 0  # 2048
        ic(tabular_data.shape, numeric_col_embeddings.shape)

        if categorical_embeddings is not None:
            ic("Concatenating Embeddings")
            ic(numeric_col_embeddings.shape, categorical_col_embeddings.shape)
            ic(numeric_col_embeddings.dtype, categorical_col_embeddings.dtype)
            col_embeddings = torch.concatenate(
                [
                    numeric_col_embeddings,  # TODO Add categorical_col_embeddings
                    categorical_col_embeddings,
                ],
                axis=1,
            )
        else:
            col_embeddings = numeric_col_embeddings

        out = self.transformer_block_1(q=tabular_data, k=col_embeddings, v=tabular_data)
        return out


class SimplePred(nn.Module):
    def __init__(self, dataset: SimpleDS, d_model: int, n_heads: int):
        super().__init__()
        self.dataset = dataset
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(
        self, numeric_inputs, categorical_inputs, deterministic=False, mask_data=True
    ):
        out = TimeSeriesTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs,
            categorical_inputs=categorical_inputs,
            deterministic=deterministic,
        )
        ic(out.shape)
        ic(f"Nan values in simplePred out 1: {torch.isnan(out).any()}")
        numeric_out = out.swapaxes(1, 2)
        numeric_out = numeric_out.reshape(
            numeric_out.shape[0], numeric_out.shape[1], -1
        )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# %%
