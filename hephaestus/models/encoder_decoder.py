# %%
# import jax


import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from icecream import ic
from jax.lax import stop_gradient
from torch.utils.data import Dataset


class DecoderDS(Dataset):
    def __init__(self, df, custom_attention: bool = True):
        # Add nan padding to make sure all sequences are the same length
        # use the idx column to group by
        self.max_seq_len = df.groupby("idx").count().time_step.max()
        self.custom_attention = custom_attention
        # Set df.idx to start from 0
        if df.idx.max() - df.idx.min() != df.idx.nunique() - 1:
            raise ValueError("idx column should start from 0 and be continuous")
        df.idx = df.idx - df.idx.min()
        self.df = df
        self.batch_size = self.max_seq_len
        self.numeric_token = "[NUMERIC_EMBEDDING]"
        self.special_tokens = ["[PAD]", "[NUMERIC_MASK]", "[MASK]", self.numeric_token]
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
        # return self.df.idx.max() + 1  # probably should be max idx + 1 thanks
        return self.df.idx.nunique()

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
        batch = np.swapaxes(batch, 0, 1)
        return batch


class FeedForwardNetwork(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool):
        # Feed Forward Network
        # using sequential container
        x = nn.Sequential(
            [
                nn.Dense(self.d_ff),
                nn.relu,
                nn.Dropout(rate=self.dropout_rate),
                nn.Dense(self.d_model),
                nn.Dropout(rate=self.dropout_rate),
            ]
        )(x)
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

    dataset: DecoderDS
    d_model: int = 64
    n_heads: int = 4
    time_window: int = 10_000

    @nn.compact
    def __call__(
        self, numeric_inputs: jnp.array, deterministic: bool, mask_data: bool = True
    ):
        embedding = nn.Embed(
            num_embeddings=self.dataset.n_tokens,
            features=self.d_model,
            name="embedding",
        )
        if mask_data:
            causal_mask = nn.make_causal_mask(numeric_inputs)
            pad_mask = nn.make_attention_mask(numeric_inputs, numeric_inputs)
            mask = nn.combine_masks(causal_mask, pad_mask)
            ic(mask.shape)
        else:
            mask = None
        # causal_mask = nn.make_causal_mask(numeric_inputs[:, :, :, 0])

        nan_mask = stop_gradient(jnp.isnan(numeric_inputs))
        numeric_inputs = jnp.where(nan_mask, 0.0, numeric_inputs)

        col_wise_embeddings = False
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
        numeric_col_embeddings = jnp.tile(
            numeric_col_embeddings[None, :, :, :],
            (numeric_inputs.shape[0], 1, 1, 1),
        )
        ic("Re-tiling", numeric_col_embeddings.shape)
        if col_wise_embeddings:
            numeric_broadcast = (
                numeric_inputs[:, :, :, None] * numeric_col_embeddings[:, :, :, :]
            )
        else:
            numeric_embedding = embedding(
                jnp.array(self.dataset.token_dict[self.dataset.numeric_token])
            )
            numeric_broadcast = numeric_inputs[:, :, :, None] * numeric_embedding

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
            max_len=self.time_window, d_pos_encoding=self.d_model
        )(numeric_broadcast)
        ic(numeric_broadcast.shape, numeric_col_embeddings.shape)
        # ic(f"Nan values in out positional: {jnp.isnan(numeric_broadcast).any()}")
        # ic("Starting Attention")
        # ic(numeric_broadcast.shape)
        pos_dim = 0  # 2048
        # First MHA
        out = TransformerBlock(
            d_model=self.d_model + pos_dim,  # TODO add pos_dim to call/init
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
        # MHA Loop
        for _ in range(3):
            out = TransformerBlock(
                d_model=self.d_model + pos_dim,  # TODO Make this more elegant
                num_heads=self.n_heads,
                d_ff=64,
                dropout_rate=0.1,
            )(
                q=out,
                k=numeric_col_embeddings,
                v=out,
                deterministic=deterministic,
                mask=mask,
            )

        # ic(f"Nan values in in out 2nd mha: {jnp.isnan(out).any()}")

        return out


class SimplePred(nn.Module):
    dataset: DecoderDS
    d_model: int = 64 * 10
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        numeric_inputs: jnp.array,
        deterministic: bool = False,
        mask_data: bool = True,
    ) -> jnp.array:
        """ """
        out = TimeSeriesTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs,
            deterministic=deterministic,
            mask_data=mask_data,
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

        ic(f"Penultimate Simple OUT: {out.shape}")
        out = jnp.squeeze(out, axis=-1)

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
        result = x + pe
        # result = jnp.concatenate([x, pe], axis=3)
        ic("PE Result shape", result.shape)

        # Add positional encoding to the input embedding
        return result


# %%
