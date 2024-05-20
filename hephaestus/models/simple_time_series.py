# %%
# import jax
import jax.numpy as jnp
from flax import linen as nn
from icecream import ic
from jax.lax import stop_gradient

from ..utils.data_utils import TabularDS

ic.configureOutput(includeContext=True, contextAbsPath=True)


# ic.disable()
# %%
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
    def __call__(self, x, deterministic: bool):
        # Multi-head self-attention
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(x, deterministic=deterministic)
        x = x + attention
        x = nn.LayerNorm()(x)

        # Feed Forward Network
        ffn = FeedForwardNetwork(
            d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = x + ffn
        x = nn.LayerNorm()(x)
        return x


class TransformerBlockOld(nn.Module):
    d_model: int
    n_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self,
        q=None,
        k=None,
        v=None,
        qkv=None,
        mask=None,
        train=True,
    ):
        if qkv is not None:
            q, k, v = (qkv, qkv, qkv)
        #  out = nn.MultiHeadAttention(num_heads=self.n_heads, qkv_features=None)(out)
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=None
        )(q, k, v, mask=mask, deterministic=not train)
        # ic(attn_output.shape, k.shape)
        # TODO Try adding q instead of k.

        out = nn.LayerNorm()(k + attn_output)
        ff_out = nn.Sequential(
            [nn.Dense(self.d_model * 2), nn.relu, nn.Dense(self.d_model)]
        )
        out = nn.LayerNorm()(out + ff_out(out))

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

    dataset: TabularDS
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

        col_embeddings = embedding(self.dataset.numeric_indices)
        ic(col_embeddings.shape, numeric_inputs.shape)
        repeated_numeric_indices = jnp.tile(
            self.dataset.numeric_indices, (numeric_inputs.shape[1], 1)
        )
        ic(repeated_numeric_indices.shape)

        numeric_col_embeddings = embedding(repeated_numeric_indices)
        # Nan Masking
        nan_mask = stop_gradient(jnp.isnan(numeric_inputs))

        # numeric_inputs = stop_gradient(
        #     jnp.where(jnp.isnan(numeric_inputs), 0.0, numeric_inputs)
        # )
        numeric_inputs = jnp.where(nan_mask, 0.0, numeric_inputs)
        ic(numeric_inputs.shape, numeric_col_embeddings.shape)
        # numeric_inputs = stop_gradient(jnp.where(nan_mask, 0.0, numeric_inputs))
        numeric_mat_mull = numeric_inputs[:, :, :, None] * numeric_col_embeddings
        # numeric_inputs = numeric_inputs[:, :, :, None]
        ic(
            "here!!!!!!",
            numeric_inputs.shape,
            numeric_col_embeddings.shape,
            nan_mask.shape,
            numeric_mat_mull.shape,
        )
        out = jnp.where(
            # nan_mask,
            jnp.expand_dims(nan_mask, axis=-1),
            embedding(jnp.array(self.dataset.numeric_mask_token)),
            numeric_mat_mull,
        )
        # End Nan Masking
        ic(f"Nan values in out: {jnp.isnan(out).any()}")

        # ic(kv_embeddings.shape, col_embeddings.shape)
        # TODO Add positional encoding
        out = PositionalEncoding(max_len=self.time_window, d_pos_encoding=16)(out)
        ic(out.shape)
        ic(f"Nan values in out positional: {jnp.isnan(out).any()}")
        ic("Starting Attention")
        ic(out.shape)
        out = TransformerBlock(
            d_model=self.d_model + 16,  # TODO Make this more elegant
            num_heads=self.n_heads,
            d_ff=64,
            dropout_rate=0.1,
        )(x=out, deterministic=deterministic)

        ic(f"Nan values in out 1st mha: {jnp.isnan(out).any()}")
        out = TransformerBlock(
            d_model=self.d_model + 16,  # TODO Make this more elegant
            num_heads=self.n_heads,
            d_ff=64,
            dropout_rate=0.1,
        )(x=out, deterministic=deterministic)

        ic(f"Nan values in in out 2nd mha: {jnp.isnan(out).any()}")

        return out


class SimplePred(nn.Module):
    dataset: TabularDS
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
        n_epochs, seq_len, n_features, _ = x.shape
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
        pe = jnp.tile(pe, (n_epochs, 1, 1, n_features))
        pe = pe.transpose((0, 1, 3, 2))  # (batch_size, seq_len, n_features, d_model)
        # concatenate the positional encoding with the input
        result = jnp.concatenate([x, pe], axis=3)

        # Add positional encoding to the input embedding
        return result
