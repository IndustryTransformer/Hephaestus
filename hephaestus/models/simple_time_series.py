# %%
# import jax


import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from icecream import ic
from jax.lax import stop_gradient
from torch.utils.data import Dataset


class SimpleDS(Dataset):
    def __init__(self, df, custom_attention: bool = True):
        # Add nan padding to make sure all sequences are the same length
        # use the idx column to group by
        self.max_seq_len = df.groupby("idx").count().time_step.max()
        self.custom_attention = custom_attention
        self.df = df
        self.batch_size = self.max_seq_len

        self.special_tokens = ["[PAD]", "[NUMERIC_MASK]", "[MASK]"]
        self.cat_mask = "[MASK]"
        self.numeric_mask = "[NUMERIC_MASK]"

        self.col_tokens = [col_name for col_name in df.columns if col_name != "idx"]

        self.tokens = self.special_tokens + self.col_tokens

        self.token_dict = {token: i for i, token in enumerate(self.tokens)}
        self.token_decoder_dict = {i: token for i, token in enumerate(self.tokens)}
        self.n_tokens = len(self.tokens)
        self.numeric_indices = jnp.array(
            [self.tokens.index(i) for i in self.col_tokens]
        )

        self.numeric_mask_token = self.tokens.index(self.numeric_mask)

    def __len__(self):
        return self.df.idx.max() + 1  # probably should be max idx + 1 thanks

    def __getitem__(self, set_idx):
        batch = self.df.loc[
            self.df.idx == set_idx, [col for col in self.df.columns if col != "idx"]
        ]
        batch = np.array(batch.values)
        # Add padding
        batch_len, n_cols = batch.shape
        pad_len = self.max_seq_len - batch_len
        padding = np.full((pad_len, n_cols), jnp.nan)
        batch = np.concatenate([batch, padding], axis=0)
        # batch = batch.T
        return batch


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


# def create_masks(batch_input):
#     # Input shape: (batch_size, n_rows, n_columns, embedding_dim)
#     batch_size, n_rows, n_columns = batch_input.shape

#     # Create padding mask for jnp.nan values
#     padding_mask = ~jnp.isnan(batch_input).any(
#         axis=-1
#     )  # shape: (batch_size, n_rows, n_columns)

#     # Create a causal mask
#     causal_mask = jnp.tril(jnp.ones((n_rows, n_rows)), k=0)  # shape: (n_rows, n_rows)

#     # Combine the masks
#     combined_mask = jnp.logical_and(
#         padding_mask[..., jnp.newaxis], causal_mask[jnp.newaxis, :, :]
#     )


#     # The combined_mask will be of shape: (batch_size, n_rows, n_columns, n_rows)
#     return combined_mask
# def create_padding_mask(batch_input):
#     # Input shape: (batch_size, n_rows, n_columns, embedding_dim)
#     padding_mask = ~jnp.isnan(batch_input).any(
#         axis=-1
#     )  # shape: (batch_size, n_rows, n_columns)
#     padding_mask = jnp.all(padding_mask, axis=-1)  # shape: (batch_size, n_rows)
#     return padding_mask[
#         :, jnp.newaxis, jnp.newaxis, :
#     ]  # shape: (batch_size, 1, 1, n_rows)


# def create_causal_mask(n_rows):
#     return jnp.tril(
#         jnp.ones((n_rows, n_rows), dtype=bool), k=0
#     )  # shape: (n_rows, n_rows)


# def combine_masks(padding_mask, causal_mask):
#     # Ensure both masks are boolean
#     padding_mask = padding_mask.astype(bool)
#     causal_mask = causal_mask.astype(bool)

#     # Expand causal mask to match the batch size and heads dimension
#     batch_size = padding_mask.shape[0]
#     causal_mask = causal_mask[
#         jnp.newaxis, jnp.newaxis, :, :
#     ]  # shape: (1, 1, n_rows, n_rows)

#     # Combine the masks
#     combined_mask = padding_mask & causal_mask  # shape: (batch_size, 1, n_rows, n_rows)
#     return combined_mask


# def create_masks(batch_input):
#     n_rows = batch_input.shape[1]
#     padding_mask = create_padding_mask(batch_input)
#     causal_mask = create_causal_mask(n_rows)
#     combined_mask = combine_masks(padding_mask, causal_mask)
#     return combined_mask


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for time series data.

    Args:
        dataset (TabularDS): Tabular dataset object containing the column indices and
        number of tokens.
        d_model (int): Dimensionality of the model.
        n_heads (int): Number of attention heads.
        time_window (int): Length of the time window.

    Attributes:
        dataset (TabularDS): Tabular dataset object containing the column indices and
        number of tokens.
        d_model (int): Dimensionality of the model.
        n_heads (int): Number of attention heads.
        time_window (int): Length of the time window.

    Methods:
        __call__(numeric_inputs): Applies the transformer to the
        inputs.

    Example:
        >>> transformer = TimeSeriesTransformer(dataset, d_model=64, n_heads=4,
        time_window=100)
        >>> numeric_inputs =
            jnp.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        >>> output = transformer(numeric_inputs)
    """

    dataset: SimpleDS
    d_model: int = 64
    n_heads: int = 4
    time_window: int = 10_000

    @nn.compact
    def __call__(self, numeric_inputs: jnp.array, deterministic: bool):
        embedding = nn.Embed(
            num_embeddings=self.dataset.n_tokens,
            features=self.d_model,
            name="embedding",
        )
        # numeric_indices = jnp.array([i for i in range(len(self.dataset.df.columns))])
        # Embed column indices
        # causal_mask = create_causal_mask(numeric_inputs)
        # attention_mask = create_padding_mask(numeric_inputs)
        # mask = combine_masks(attention_mask, causal_mask)
        numeric_inputs = jnp.swapaxes(numeric_inputs, 1, 2)
        causal_mask = nn.make_causal_mask(numeric_inputs)
        pad_mask = nn.make_attention_mask(numeric_inputs, numeric_inputs)
        mask = nn.combine_masks(causal_mask, pad_mask)
        ic(mask.shape)
        # causal_mask = nn.make_causal_mask(numeric_inputs[:, :, :, 0])
        col_embeddings = embedding(self.dataset.numeric_indices)
        ic(col_embeddings.shape, numeric_inputs.shape)
        repeated_numeric_indices = jnp.tile(
            self.dataset.numeric_indices, (numeric_inputs.shape[2], 1)
        )
        ic("before swap", repeated_numeric_indices.shape)
        # repeated_numeric_indices = jnp.swapaxes(repeated_numeric_indices, 0, 1)
        repeated_numeric_indices = repeated_numeric_indices.T
        ic("after swap", repeated_numeric_indices.shape)
        numeric_col_embeddings = embedding(repeated_numeric_indices)
        ic("Embedding!!", numeric_col_embeddings.shape)
        # Nan Masking
        nan_mask = stop_gradient(jnp.isnan(numeric_inputs))
        # Add dimension to numeric_col_embeddings and tile it so that it has the same
        # Batch size as numeric_inputs
        numeric_col_embeddings = jnp.tile(
            numeric_col_embeddings[None, :, :, :], (numeric_inputs.shape[0], 1, 1, 1)
        )
        ic("Retiling", numeric_col_embeddings.shape)
        # numeric_inputs = stop_gradient(
        #     jnp.where(jnp.isnan(numeric_inputs), 0.0, numeric_inputs)
        # )
        numeric_inputs = jnp.where(nan_mask, 0.0, numeric_inputs)
        ic("Before Broadcast", numeric_inputs.shape, numeric_col_embeddings.shape)
        # numeric_inputs = stop_gradient(jnp.where(nan_mask, 0.0, numeric_inputs))
        numeric_broadcast = (
            numeric_inputs[:, :, :, None] * numeric_col_embeddings[:, :, :, :]
        )
        # numeric_inputs = numeric_inputs[:, :, :, None]
        # ic(
        #     "here!!!!!!",
        #     numeric_inputs.shape,
        #     numeric_col_embeddings.shape,
        #     nan_mask.shape,
        #     numeric_broadcast.shape,
        # )
        ic("Before where", numeric_broadcast.shape, nan_mask.shape)
        numeric_broadcast = jnp.where(
            # nan_mask,
            # jnp.expand_dims(nan_mask, axis=-1),
            nan_mask[:, :, :, None],
            embedding(jnp.array(self.dataset.numeric_mask_token)),
            numeric_broadcast,
        )
        # End Nan Masking
        # ic(f"Nan values in out: {jnp.isnan(numeric_broadcast).any()}")

        # ic(kv_embeddings.shape, col_embeddings.shape)
        # TODO Add positional encoding
        # numeric_broadcast.shape: (4, 26, 59, 59, 256)
        ic("Before Positional Encoding", numeric_broadcast.shape)

        numeric_broadcast = PositionalEncoding(
            max_len=self.time_window, d_pos_encoding=16
        )(numeric_broadcast)
        ic(numeric_broadcast.shape, numeric_col_embeddings.shape)
        # ic(f"Nan values in out positional: {jnp.isnan(numeric_broadcast).any()}")
        # ic("Starting Attention")
        # ic(numeric_broadcast.shape)
        out = TransformerBlock(
            d_model=self.d_model + 16,  # TODO Make this more elegant
            num_heads=self.n_heads,
            d_ff=64,
            dropout_rate=0.1,
        )(
            q=numeric_broadcast,
            k=numeric_col_embeddings,
            v=numeric_broadcast,  # TODO differentiate this
            deterministic=deterministic,
            mask=mask,
        )

        # ic(f"Nan values in out 1st mha: {jnp.isnan(out).any()}")
        out = TransformerBlock(
            d_model=self.d_model + 16,  # TODO Make this more elegant
            num_heads=self.n_heads,
            d_ff=64,
            dropout_rate=0.1,
        )(q=out, k=out, v=out, deterministic=deterministic, mask=mask)

        # ic(f"Nan values in in out 2nd mha: {jnp.isnan(out).any()}")

        return out


class SimplePred(nn.Module):
    dataset: SimpleDS
    d_model: int = 64 * 10
    n_heads: int = 4

    @nn.compact
    def __call__(
        self, numeric_inputs: jnp.array, deterministic: bool = False
    ) -> jnp.array:
        """ """
        out = TimeSeriesTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs, deterministic=deterministic
        )
        ic(out.shape)
        ic(f"Nan values in simplePred out 1: {jnp.isnan(out).any()}")
        out = nn.Sequential(
            [
                nn.Dense(name="RegressionDense1", features=self.d_model * 2),
                nn.relu,
                nn.Dense(name="RegressionDense2", features=1),
            ],
            name="RegressionOutputChain",
        )(out)
        ic(f"Nan values in simplePred out after seq: {jnp.isnan(out).any()}")

        ic(f"Penultimate Simple OUT: {out.shape}")
        out = jnp.squeeze(out, axis=-1)
        ic(f"Nan values in simplePred out after seq: {jnp.isnan(out).any()}")

        # out = jnp.reshape(out, (out.shape[0], -1))
        ic(f"FInal Simple OUT: {out.shape}")
        # out = nn.Dense(name="RegressionFlatten", features=16)(out)
        return out


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
        result = jnp.concatenate([x, pe], axis=3)

        # Add positional encoding to the input embedding
        return result


# %%
