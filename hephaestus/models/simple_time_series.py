# %%
# import jax
import re
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import linen as nn
from icecream import ic
from jax.lax import stop_gradient
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
        self.numeric_indices = jnp.array(
            [self.tokens.index(i) for i in self.numeric_col_tokens]
        )
        self.categorical_indices = jnp.array(
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
        )["input_ids"]  # TODO make this custom to reduce dictionary size

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
        return batch

    def __getitem__(self, set_idx):
        if self.df_categorical.empty:
            categorical_inputs = None
        else:
            categorical_inputs = self.get_data("df_categorical", set_idx)
        numeric_inputs = self.get_data("df_numeric", set_idx)

        return numeric_inputs, categorical_inputs


class FeedForwardNetwork(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool):
        # Feed Forward Network
        x = nn.Dense(self.d_ff)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class TransformerBlock(nn.Module):
    num_heads: int
    d_model: int
    d_ff: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self,
        q: jnp.array,
        k: jnp.array,
        v: jnp.array,
        deterministic: bool,
        mask: jnp.array = None,
    ):
        # Multi-head self-attention
        # causal_mask = causal_mask = nn.make_causal_mask(q[:, :, :, 0])

        # Write out the jax array to a file for more debugging
        ic("Query shapes 222222", q.shape, k.shape, v.shape)
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(q, k, v, deterministic=deterministic, mask=mask)
        out = q + attention
        out = nn.LayerNorm()(out)
        # Feed Forward Network
        ffn = FeedForwardNetwork(
            d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
        )(out, deterministic=deterministic)
        out = out + ffn
        out = nn.LayerNorm()(out)
        return out


class ReservoirEmbedding(nn.Module):
    """
    A module for performing reservoir embedding on a given input.

    Args:
        dataset (SimpleDS): The dataset used for embedding.
        features (int): The number of features in the embedding.
        frozen_index (int, optional): The index of the embedding to freeze. Defaults to 0.
    """

    dataset: SimpleDS
    features: int
    frozen_index: int = 0  # The index of the embedding to freeze

    @nn.compact
    def __call__(self, base_indices: jnp.array):
        """
        Perform reservoir embedding on the given input.

        Args:
            base_indices (jnp.array): The base indices for embedding.

        Returns:
            jnp.array: The ultimate embedding after reservoir embedding.
        """

        embedding = self.param(
            "embedding",
            nn.initializers.normal(stddev=0.02),
            (self.dataset.tokenizer.vocab_size, self.features),
        )

        # Create a mask for the frozen embedding
        frozen_mask = jnp.arange(self.dataset.tokenizer.vocab_size) == self.frozen_index

        # Set the frozen embedding to zero
        frozen_embedding = jnp.where(frozen_mask[:, None], 0.0, embedding)

        # Stop gradient for the frozen embedding
        penultimate_embedding = stop_gradient(frozen_embedding) + jnp.where(
            frozen_mask[:, None], 0.0, embedding - frozen_embedding
        )
        token_reservoir_lookup = self.dataset.reservoir_encoded
        reservoir_indices = token_reservoir_lookup[base_indices]

        ultimate_embedding = penultimate_embedding[reservoir_indices]
        ultimate_embedding = jnp.sum(ultimate_embedding, axis=-2)

        return ultimate_embedding


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for time series data.

    Args:
        dataset (SimpleDS): The dataset object containing the time series data.
        d_model (int, optional): The dimensionality of the model. Defaults to 64.
        n_heads (int, optional): The number of attention heads. Defaults to 4.
        time_window (int, optional): The maximum length of the time window. Defaults to 10000.

    Methods:
        __call__(self, numeric_inputs: jnp.array, deterministic: bool, mask_data: bool = True) -> jnp.array:
            Applies the transformer model to the input time series data.

    Attributes:
        dataset (SimpleDS): The dataset object containing the time series data.
        d_model (int): The dimensionality of the model.
        n_heads (int): The number of attention heads.
        time_window (int): The maximum length of the time window.
    """

    dataset: SimpleDS
    d_model: int = 64
    n_heads: int = 4
    time_window: int = 10_000

    @nn.compact
    def __call__(
        self,
        numeric_inputs: Optional[jnp.array] = None,
        categorical_inputs: Optional[jnp.array] = None,
        deterministic: bool = False,
        mask_data: bool = True,
    ):
        embedding = ReservoirEmbedding(
            self.dataset,
            features=self.d_model,
        )

        ########### NUMERIC INPUTS ###########
        ic(numeric_inputs.shape)
        nan_mask = stop_gradient(jnp.isnan(numeric_inputs))
        ic("Here Again???")
        numeric_inputs = jnp.where(nan_mask, 0.0, numeric_inputs)
        # Cat Embedding
        if categorical_inputs is not None:
            # Make sure nans are set to <NAN> token
            categorical_inputs = jnp.where(
                jnp.isnan(categorical_inputs),
                jnp.array(self.dataset.token_dict["[NUMERIC_MASK]"]),
                categorical_inputs,
            )
            categorical_embeddings = embedding(categorical_inputs)

        else:
            categorical_embeddings = None

        # Numeric Embedding
        repeated_numeric_indices = jnp.tile(
            self.dataset.numeric_indices, (numeric_inputs.shape[2], 1)
        )

        # repeated_numeric_indices = jnp.swapaxes(repeated_numeric_indices, 0, 1)
        repeated_numeric_indices = repeated_numeric_indices.T
        numeric_col_embeddings = embedding(repeated_numeric_indices)
        # Nan Masking
        numeric_col_embeddings = jnp.tile(
            numeric_col_embeddings[None, :, :, :],
            (numeric_inputs.shape[0], 1, 1, 1),
        )
        if categorical_embeddings is not None:
            repeated_categorical_indices = jnp.tile(
                self.dataset.categorical_indices, (categorical_inputs.shape[2], 1)
            )
            repeated_categorical_indices = repeated_categorical_indices.T
            categorical_col_embeddings = embedding(repeated_categorical_indices)
            categorical_col_embeddings = jnp.tile(
                categorical_col_embeddings[None, :, :, :],
                (categorical_inputs.shape[0], 1, 1, 1),
            )
        else:
            categorical_col_embeddings = None
        numeric_embedding = embedding(
            jnp.array(self.dataset.token_dict[self.dataset.numeric_token])
        )
        numeric_broadcast = numeric_inputs[:, :, :, None] * numeric_embedding

        numeric_broadcast = jnp.where(
            # nan_mask,
            # jnp.expand_dims(nan_mask, axis=-1),
            nan_mask[:, :, :, None],
            embedding(jnp.array(self.dataset.numeric_mask_token)),
            numeric_broadcast,
        )
        # End Nan Masking
        # ic(f"Nan values in out: {jnp.isnan(numeric_broadcast).any()}")

        numeric_broadcast = PositionalEncoding(
            max_len=self.time_window, d_pos_encoding=self.d_model
        )(numeric_broadcast)
        ic(numeric_broadcast.shape, numeric_col_embeddings.shape)
        # ic(f"Nan values in out positional: {jnp.isnan(numeric_broadcast).any()}")
        # ic("Starting Attention")
        # ic(numeric_broadcast.shape)
        if categorical_embeddings is not None:
            tabular_data = jnp.concatenate(
                [numeric_broadcast, categorical_embeddings], axis=1
            )
        else:
            ic("No Categorical Embeddings")
            tabular_data = numeric_broadcast
        if categorical_embeddings is not None:
            ic("Masking for categorical data")
            mask_input = jnp.concatenate([numeric_inputs, categorical_inputs], axis=1)

        else:
            ic("No Masking for categorical data")
            mask_input = numeric_inputs
        ic(mask_input.shape)
        if mask_data:
            causal_mask = nn.make_causal_mask(mask_input)
            pad_mask = nn.make_attention_mask(
                mask_input, mask_input
            )  # TODO Add in the mask for the categorical data
            mask = nn.combine_masks(causal_mask, pad_mask)
            ic(mask.shape)
        else:
            mask = None
        pos_dim = 0  # 2048
        ic(tabular_data.shape, numeric_col_embeddings.shape)

        if categorical_embeddings is not None:
            ic("Concatenating Embeddings")
            ic(numeric_col_embeddings.shape, categorical_col_embeddings.shape)
            ic(numeric_col_embeddings.dtype, categorical_col_embeddings.dtype)
            col_embeddings = jnp.concatenate(
                [
                    numeric_col_embeddings,  # TODO Add categorical_col_embeddings
                    categorical_col_embeddings,
                ],
                axis=1,
            )
        else:
            col_embeddings = numeric_col_embeddings
        out = TransformerBlock(
            d_model=self.d_model + pos_dim,  # TODO add pos_dim to call/init
            num_heads=self.n_heads,
            d_ff=64,
            dropout_rate=0.1,
            name="TransformerBlock_Initial",
        )(
            q=tabular_data,
            k=col_embeddings,
            v=tabular_data,  # TODO differentiate this
            deterministic=deterministic,
            mask=mask,
        )
        for i in range(3):
            out = TransformerBlock(
                d_model=self.d_model + pos_dim,  # TODO Make this more elegant
                num_heads=self.n_heads,
                d_ff=64,
                dropout_rate=0.1,
                name=f"TransformerBlock_{i}",
            )(
                q=out,
                k=col_embeddings,
                v=out,
                deterministic=deterministic,
                mask=mask,
            )

        # ic(f"Nan values in in out 2nd mha: {jnp.isnan(out).any()}")

        return out


class SimplePred(nn.Module):
    dataset: SimpleDS
    d_model: int = 64 * 10
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        numeric_inputs: jnp.array,
        categorical_inputs: Optional[jnp.array] = None,
        deterministic: bool = False,
        mask_data: bool = True,
    ) -> jnp.array:
        """ """
        out = TimeSeriesTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs,
            categorical_inputs=categorical_inputs,
            deterministic=deterministic,
            mask_data=mask_data,
        )
        ic(out.shape)
        ic(f"Nan values in simplePred out 1: {jnp.isnan(out).any()}")
        numeric_out = nn.Sequential(
            [
                nn.Dense(name="RegressionDense1", features=self.d_model * 2),
                nn.relu,
                nn.Dense(
                    name="RegressionDense2", features=len(self.dataset.numeric_indices)
                ),
            ],
            name="RegressionOutputChain",
        )(out)
        if categorical_inputs is not None:
            categorical_out = nn.Sequential(
                [
                    nn.Dense(name="CategoricalDense1", features=self.d_model * 2),
                    nn.relu,
                    nn.Dense(
                        name="CategoricalDense2",
                        features=len(self.dataset.categorical_indices),
                    ),
                ],
                name="CategoricalOutputChain",
            )(out)
            return {"numeric_out": numeric_out, "categorical_out": categorical_out}
        else:
            return {"numeric_out": numeric_out, "categorical_out": None}

        # ic(f"Penultimate Simple OUT: {numeric_out.shape}, {categorical_out.shape}")
        # numeric_out = jnp.squeeze(numeric_out, axis=-1)
        # categorical_out = jnp.squeeze(categorical_out, axis=-1)

        # out = jnp.reshape(out, (out.shape[0], -1))
        # ic(f"Final Simple OUT: {numeric_out.shape}, {categorical_out.shape}")
        # out = nn.Dense(name="RegressionFlatten", features=16)(out)


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module using @nn.compact. This module injects information
    about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that
    the two can be summed. This allows the model to learn the relative positions
    of tokens within the sequence.
    """

    max_len: int  # Maximum length of the input sequences
    d_pos_encoding: int  # Dimensionality of the embeddings/inputs

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the positional encoding. Concatenates positional encoding to
        the input.

        Args:
            x: Input data. Shape: (batch_size, seq_len, d_model)

        Returns:
            Output with positional encoding added. Shape: (batch_size, seq_len, d_model)
        """
        n_epochs, n_columns, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} is larger than the",
                f"maximum length {self.max_len}",
            )

        # Calculate positional encoding
        position = jnp.arange(self.max_len)[:, jnp.newaxis]
        div_term = jnp.exp(
            jnp.arange(0, self.d_pos_encoding, 2)
            * -(jnp.log(10000.0) / self.d_pos_encoding)
        )
        pe = jnp.zeros((self.max_len, self.d_pos_encoding))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[:seq_len, :]
        pe = pe[None, :, :, None]
        ic("pe before tiling", pe.shape)
        pe = jnp.tile(pe, (n_epochs, 1, 1, n_columns))
        ic("pe after tiling", pe.shape)
        pe = pe.transpose((0, 3, 1, 2))  #
        ic("pe after transpose", pe.shape)
        # concatenate the positional encoding with the input
        result = x + pe
        # result = jnp.concatenate([x, pe], axis=3)
        ic("PE Result shape", result.shape)

        # Add positional encoding to the input embedding
        return result


# %%
