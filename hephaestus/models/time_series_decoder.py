# %%
# import jax
import re
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import linen as nn
from flax.struct import dataclass
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


@dataclass
class TimeSeriesConfig:
    numeric_token: str = None
    numeric_mask: str = None
    numeric_col_tokens: list = None
    categorical_col_tokens: list = None
    tokens: list = None
    token_dict: dict = None
    token_decoder_dict: dict = None
    n_tokens: int = None
    numeric_indices: jnp.array = None
    categorical_indices: jnp.array = None
    object_tokens: list = None
    numeric_mask_token: int = None
    reservoir_vocab: list = None
    reservoir_encoded: jnp.array = None
    tokenizer: AutoTokenizer = None

    @classmethod
    def generate(cls, df: pd.DataFrame) -> "TimeSeriesConfig":
        max_seq_len = df.groupby("idx").count().time_step.max()
        # Set df.idx to start from 0
        df.idx = df.idx - df.idx.min()
        df = df.set_index("idx")
        df.index.name = None

        df_categorical = df.select_dtypes(include=["object"]).astype(str)
        numeric_token = "[NUMERIC_EMBEDDING]"
        cls_dict = {}
        cls_dict["numeric_token"] = numeric_token
        special_tokens = [
            "[PAD]",
            "[NUMERIC_MASK]",
            "[MASK]",
            "[UNK]",
            numeric_token,
        ]
        cls_dict["numeric_mask"] = "[NUMERIC_MASK]"
        numeric_mask = cls_dict["numeric_mask"]
        # Remove check on idx
        cls_dict["numeric_col_tokens"] = [
            col_name for col_name in df.select_dtypes(include="number").columns
        ]
        cls_dict["categorical_col_tokens"] = [
            col_name for col_name in df.select_dtypes(include="object").columns
        ]
        # Get all the unique values in the categorical columns and add them to the tokens
        cls_dict["object_tokens"] = (
            df_categorical.apply(pd.Series.unique).values.flatten().tolist()
        )
        cls_dict["tokens"] = (
            special_tokens
            + cls_dict["numeric_col_tokens"]
            + cls_dict["object_tokens"]
            + cls_dict["categorical_col_tokens"]
        )
        tokens = cls_dict["tokens"]
        numeric_col_tokens = cls_dict["numeric_col_tokens"]
        categorical_col_tokens = cls_dict["categorical_col_tokens"]

        token_dict = {token: i for i, token in enumerate(tokens)}
        cls_dict["token_dict"] = token_dict
        token_decoder_dict = {i: token for i, token in enumerate(tokens)}
        cls_dict["token_decoder_dict"] = token_decoder_dict
        n_tokens = len(cls_dict["tokens"])
        cls_dict["n_tokens"] = n_tokens
        numeric_indices = jnp.array([tokens.index(i) for i in numeric_col_tokens])
        cls_dict["numeric_indices"] = numeric_indices
        categorical_indices = jnp.array(
            [tokens.index(i) for i in categorical_col_tokens]
        )
        cls_dict["categorical_indices"] = categorical_indices

        numeric_mask_token = tokens.index(numeric_mask)
        cls_dict["numeric_mask_token"] = numeric_mask_token
        # Make custom vocab by splitting on snake case, camel case, spaces and numbers
        reservoir_vocab = [split_complex_word(word) for word in token_dict.keys()]
        # flatten the list, make a set and then list again
        reservoir_vocab = list(
            set([item for sublist in reservoir_vocab for item in sublist])
        )
        # Get reservoir embedding tokens
        reservoir_tokens_list = [
            token_decoder_dict[i] for i in range(len(token_decoder_dict))
        ]  # ensures they are in the same order
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls_dict["tokenizer"] = tokenizer
        reservoir_encoded = tokenizer(
            reservoir_tokens_list,
            padding="max_length",
            max_length=8,  # TODO Make this dynamic
            truncation=True,
            return_tensors="jax",
            add_special_tokens=False,
        )[
            "input_ids"
        ]  # TODO make this custom to reduce dictionary size
        cls_dict["reservoir_encoded"] = reservoir_encoded
        cls_dict["reservoir_vocab"] = reservoir_vocab

        df_categorical = convert_object_to_int_tokens(df_categorical, token_dict)

        return cls(**cls_dict)


class TimeSeriesDS(Dataset):
    def __init__(self, df: pd.DataFrame, config: TimeSeriesConfig):
        # Add nan padding to make sure all sequences are the same length
        # use the idx column to group by
        self.max_seq_len = df.groupby("idx").count().time_step.max()
        # Set df.idx to start from 0
        df.idx = df.idx - df.idx.min()
        df = df.set_index("idx")
        df.index.name = None

        def convert_object_to_int_tokens(df, token_dict):
            """Converts object columns to integer tokens using a token dictionary."""
            df = df.copy()
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].map(token_dict)
            return df

        self.df_categorical = df.select_dtypes(include=["object"]).astype(str)
        self.df_categorical = convert_object_to_int_tokens(
            self.df_categorical, config.token_dict
        )
        self.df_numeric = df.select_dtypes(include="number")
        self.batch_size = self.max_seq_len

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

    config: TimeSeriesConfig
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
            (self.config.tokenizer.vocab_size, self.features),
        )

        # Create a mask for the frozen embedding
        frozen_mask = jnp.arange(self.config.tokenizer.vocab_size) == self.frozen_index

        # Set the frozen embedding to zero
        frozen_embedding = jnp.where(frozen_mask[:, None], 0.0, embedding)

        # Stop gradient for the frozen embedding
        penultimate_embedding = stop_gradient(frozen_embedding) + jnp.where(
            frozen_mask[:, None], 0.0, embedding - frozen_embedding
        )
        token_reservoir_lookup = self.config.reservoir_encoded
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

    config: TimeSeriesConfig  # TODO Make this a flax data class with limited
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
            self.config,
            features=self.d_model,
        )

        ########### NUMERIC INPUTS ###########

        nan_mask = stop_gradient(jnp.isnan(numeric_inputs))

        numeric_inputs = jnp.where(nan_mask, 0.0, numeric_inputs)
        # Cat Embedding
        if categorical_inputs is not None:
            # Make sure nans are set to <NAN> token
            categorical_inputs = jnp.where(
                jnp.isnan(categorical_inputs),
                jnp.array(self.config.token_dict["[NUMERIC_MASK]"]),
                categorical_inputs,
            )
            categorical_embeddings = embedding(categorical_inputs)

        else:
            categorical_embeddings = None

        # Numeric Embedding
        repeated_numeric_indices = jnp.tile(
            self.config.numeric_indices, (numeric_inputs.shape[2], 1)
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
                self.config.categorical_indices, (categorical_inputs.shape[2], 1)
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
            jnp.array(self.config.token_dict[self.config.numeric_token])
        )
        numeric_broadcast = numeric_inputs[:, :, :, None] * numeric_embedding

        numeric_broadcast = jnp.where(
            # nan_mask,
            # jnp.expand_dims(nan_mask, axis=-1),
            nan_mask[:, :, :, None],
            embedding(jnp.array(self.config.numeric_mask_token)),
            numeric_broadcast,
        )
        # End Nan Masking
        #

        numeric_broadcast = PositionalEncoding(
            max_len=self.time_window, d_pos_encoding=self.d_model
        )(numeric_broadcast)

        #
        #
        #
        if categorical_embeddings is not None:
            tabular_data = jnp.concatenate(
                [numeric_broadcast, categorical_embeddings], axis=1
            )
        else:

            tabular_data = numeric_broadcast
        if categorical_embeddings is not None:

            mask_input = jnp.concatenate([numeric_inputs, categorical_inputs], axis=1)

        else:

            mask_input = numeric_inputs

        if mask_data:
            causal_mask = nn.make_causal_mask(mask_input)
            pad_mask = nn.make_attention_mask(
                mask_input, mask_input
            )  # TODO Add in the mask for the categorical data
            mask = nn.combine_masks(causal_mask, pad_mask)

        else:
            mask = None
        pos_dim = 0  # 2048

        if categorical_embeddings is not None:

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

        #

        return out


class SimplePred(nn.Module):
    config: TimeSeriesConfig
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
        out = TimeSeriesTransformer(self.config, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs,
            categorical_inputs=jnp.astype(categorical_inputs, jnp.int32),
            deterministic=deterministic,
            mask_data=mask_data,
        )

        numeric_out = out.swapaxes(1, 2)
        numeric_out = numeric_out.reshape(
            numeric_out.shape[0], numeric_out.shape[1], -1
        )  # TODO This is wrong. Make this
        #  TODO WORK HERE!!!!! be of shape (batch_size, )

        numeric_out = nn.Sequential(
            [
                nn.Dense(name="RegressionDense1", features=self.d_model * 2),
                nn.relu,
                nn.Dense(
                    name="RegressionDense2", features=len(self.config.numeric_indices)
                ),
            ],
            name="RegressionOutputChain",
        )(numeric_out)
        numeric_out = numeric_out.swapaxes(1, 2)

        if categorical_inputs is not None:

            # categorical_out = out.swapaxes(1, 2)
            categorical_out = out.copy()
            # categorical_out = categorical_out.reshape(
            #     categorical_out.shape[0], categorical_out.shape[1], -1
            # )

            categorical_out = nn.Dense(
                name="CategoricalDense1",
                features=len(self.config.token_decoder_dict.items()),
            )(categorical_out)

            categorical_out = nn.relu(categorical_out)

            categorical_out = categorical_out.swapaxes(1, 3)

            categorical_out = nn.Dense(
                name="CategoricalDense2",
                features=len(self.config.categorical_col_tokens),
            )(categorical_out)

            categorical_out = categorical_out.swapaxes(1, 3)

        else:

            categorical_out = None

        return {"numeric_out": numeric_out, "categorical_out": categorical_out}


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

        pe = jnp.tile(pe, (n_epochs, 1, 1, n_columns))

        pe = pe.transpose((0, 3, 1, 2))  #

        # concatenate the positional encoding with the input
        result = x + pe
        # result = jnp.concatenate([x, pe], axis=3)

        # Add positional encoding to the input embedding
        return result


# %%
