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


class MultiheadAttention(nn.Module):
    n_heads: int
    d_model: int

    @nn.compact
    def __call__(
        self, q=None, k=None, v=None, qkv=None, input_feed_forward=True, mask=None
    ):
        """Self Attention"""
        # Assert that either qkv or q, k, v are provided
        assert (qkv is not None) or (
            q is not None and k is not None and v is not None
        ), "Either qkv or q, k, v must be provided"
        if input_feed_forward:
            if qkv is not None:
                q, k, v = (qkv, qkv, qkv)
            q = nn.Dense(name="q_linear", features=self.d_model)(q)
            k = nn.Dense(name="k_linear", features=self.d_model)(k)
            v = nn.Dense(name="v_linear", features=self.d_model)(v)

        ic(q.shape, k.shape, v.shape)
        # split the d_model into n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        k = k.reshape(
            k.shape[0],
            k.shape[1],
            k.shape[2],
            self.n_heads,
            self.d_model // self.n_heads,
        )  # .transpose(0, 2, 1, 3)

        if qkv is not None:
            q = q.reshape(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                self.n_heads,
                self.d_model // self.n_heads,
            )
        else:
            q = q.reshape(
                q.shape[0],
                self.n_heads,
                self.d_model // self.n_heads,
            )  # .transpose(0, 2, 1, 3)
        v = v.reshape(
            v.shape[0],
            v.shape[1],
            v.shape[2],
            self.n_heads,
            self.d_model // self.n_heads,
        )  # .transpose(0, 2, 1, 3)
        # ic(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
        # here _ = attention_weights
        attention_out, _ = self.scaled_dot_product_attention(q, k, v, mask)
        ic(attention_out.shape)
        # attention_out = attention_output.transpose(0, 2, 1, 3)
        # .reshape(  # TODO Check if this is correct
        #     attention_output.shape[0], -1, self.d_model
        # )
        attention_out = attention_out.reshape(
            attention_out.shape[0],
            attention_out.shape[1],
            attention_out.shape[2],
            -1,
        )
        ic(attention_out.shape)

        out = nn.Dense(features=self.d_model)(attention_out)

        return out

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights."""
        matmul_qk = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        # ic(f"Matmul shape: {matmul_qk.shape}")
        d_k = q.shape[-1]
        matmul_qk = matmul_qk / jnp.sqrt(d_k)
        ic(f"Before Masking matmul_qk: {matmul_qk.shape=}")
        if mask is not None:
            matmul_qk += mask * -1e9

        attention_weights = nn.softmax(matmul_qk, axis=-1)

        output = jnp.matmul(attention_weights, v)
        # ic(f"Scaled Output shape: {output.shape}")

        return output, attention_weights


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
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
        input_feed_forward=True,
    ):
        attn_output = MultiheadAttention(n_heads=self.n_heads, d_model=self.d_model)(
            q, k, v, qkv, mask=mask, input_feed_forward=input_feed_forward
        )
        # ic(attn_output.shape, k.shape)
        # TODO Try adding q instead of k.
        if qkv is not None:
            k = qkv
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
    def __call__(
        self,
        numeric_inputs: jnp.array,
    ):
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
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            dropout_rate=0.1,
        )(qkv=out)
        # out = nn.MultiHeadAttention(num_heads=self.n_heads, qkv_features=None)(out)
        ic(out.shape)
        ic(f"Nan values in out 1st mha: {jnp.isnan(out).any()}")

        # out = nn.MultiHeadAttention(num_heads=self.n_heads, qkv_features=None)(out)
        out = TransformerBlock(
            d_model=self.d_model + 16,  # TODO Make this more elegant
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            dropout_rate=0.1,
        )(qkv=out)  # Check if we should reuse the col embeddings here
        ic(f"Nan values in in out 2nd mha: {jnp.isnan(out).any()}")

        return out


class SimplePred(nn.Module):
    dataset: TabularDS
    d_model: int = 64 * 10
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        numeric_inputs: jnp.array,
    ) -> jnp.array:
        """ """
        out = TimeSeriesTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs
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
