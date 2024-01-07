# %%
# import jax
import jax.numpy as jnp
from flax import linen as nn
from icecream import ic
from jax.lax import stop_gradient

from ..utils.data_utils import TabularDS

ic.configureOutput(includeContext=True)
ic.disable()
# %%


class MultiheadAttention(nn.Module):
    n_heads: int
    d_model: int

    @nn.compact
    def __call__(self, q, k, v, input_feed_forward=True, mask=None):
        """Self Attention"""
        if input_feed_forward:
            q = nn.Dense(name="q_linear", features=self.d_model)(q)
            k = nn.Dense(name="k_linear", features=self.d_model)(k)
            v = nn.Dense(name="v_linear", features=self.d_model)(v)

        # ic(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
        # here _ = attention_weights
        attention_output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        atten_out = attention_output.transpose(0, 2, 1).reshape(
            attention_output.shape[0], -1, self.d_model
        )
        out = nn.Dense(features=self.d_model)(atten_out)

        return out

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights."""
        matmul_qk = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        # ic(f"Matmul shape: {matmul_qk.shape}")
        d_k = q.shape[-1]
        matmul_qk = matmul_qk / jnp.sqrt(d_k)

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
    def __call__(self, q, k, v, mask=None, train=True, input_feed_forward=True):
        attn_output = MultiheadAttention(n_heads=self.n_heads, d_model=self.d_model)(
            q, k, v, mask=mask, input_feed_forward=input_feed_forward
        )
        out = nn.LayerNorm()(q + attn_output)
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
        __call__(categorical_inputs, numeric_inputs): Applies the transformer to the
        inputs.

    Example:
        >>> transformer = TimeSeriesTransformer(dataset, d_model=64, n_heads=4,
        time_window=100)
        >>> categorical_inputs = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> numeric_inputs =
            jnp.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        >>> output = transformer(categorical_inputs, numeric_inputs)
    """

    dataset: TabularDS
    d_model: int = 64
    n_heads: int = 4
    time_window: int = 100

    @nn.compact
    def __call__(
        self,
        categorical_inputs: jnp.array,
        numeric_inputs: jnp.array,
    ):
        embedding = nn.Embed(
            num_embeddings=self.dataset.n_tokens,
            features=self.d_model,
            name="embedding",
        )
        repeated_numeric_indices = jnp.tile(
            self.dataset.numeric_indices, (numeric_inputs.shape[0],)
        )
        # Embed column indices

        col_embeddings = embedding(self.dataset.col_indices)
        ic(f"{col_embeddings.shape=}")
        cat_embeddings = embedding(categorical_inputs)
        ic(f"Numeric inputs shape: {numeric_inputs.shape}")
        repeated_numeric_indices = jnp.tile(
            self.dataset.numeric_indices, (numeric_inputs.shape[1], 1)
        )

        # Nan Masking
        numeric_col_embeddings = embedding(repeated_numeric_indices)
        nan_mask = stop_gradient(jnp.isnan(numeric_inputs))
        base_numeric = jnp.zeros_like(numeric_col_embeddings)
        numeric_inputs = stop_gradient(jnp.where(nan_mask, 0.0, numeric_inputs))
        numeric_mat_mull = numeric_col_embeddings * numeric_inputs[:, :, :, None]
        # ic(
        #     f"{base_numeric.shape=}",
        #     f"{numeric_col_embeddings.shape=}",
        #     f"{numeric_inputs.shape=}",
        #     f"{numeric_inputs[:, :, :, None].shape=}",
        #     f"{nan_mask[:, :, :].shape=}",
        #     f"{jnp.expand_dims(nan_mask, axis=-1).shape=}",
        #     f"{numeric_mat_mull.shape=}",
        # )

        base_numeric = jnp.where(
            # nan_mask[:, :, :, None],
            jnp.expand_dims(nan_mask, axis=-1),
            base_numeric,
            numeric_mat_mull,
        )

        base_numeric = jnp.where(
            nan_mask[:, :, :, None], self.dataset.numeric_mask_token, base_numeric
        )
        # End Nan Masking
        ic(cat_embeddings.shape, base_numeric.shape)
        kv_embeddings = jnp.concatenate([cat_embeddings, base_numeric], axis=2)
        ic(kv_embeddings.shape)
        kv_embeddings = kv_embeddings.reshape(
            kv_embeddings.shape[0], -1, kv_embeddings.shape[3]
        )
        # kv_embeddings = jnp.expand_dims(
        #     kv_embeddings.reshape(kv_embeddings.shape[0], -1, kv_embeddings.shape[3]),
        #     axis=0,
        # )
        ic(kv_embeddings.shape, col_embeddings.shape)
        col_embeddings = jnp.tile(col_embeddings, (kv_embeddings.shape[0], 1, 1))
        ic(kv_embeddings.shape, col_embeddings.shape)
        # TODO Add positional encoding
        # kv_embeddings = PositionalEncoding(d_model=self.d_model)(kv_embeddings)
        # col_embeddings = PositionalEncoding(d_model=self.d_model)(col_embeddings)
        ic(col_embeddings.shape, kv_embeddings.shape)
        out = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            dropout_rate=0.1,
        )(q=col_embeddings, k=kv_embeddings, v=kv_embeddings)
        ic(f"First MHA out shape: {out.shape}")
        out = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            dropout_rate=0.1,
        )(
            q=col_embeddings, k=out, v=out
        )  # Check if we should reuse the col embeddings here
        ic(f"Second MHA out shape: {out.shape}")
        return out


# %%


def multivariate_regression(
    dataset: TabularDS,
    d_model: int = 64,
    n_heads: int = 4,
) -> nn.Module:
    """ """
    return TimeSeriesRegression(dataset, d_model, n_heads)


class TimeSeriesRegression(nn.Module):
    dataset: TabularDS
    d_model: int = 64 * 10
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        categorical_inputs: jnp.array,
        numeric_inputs: jnp.array,
    ) -> jnp.array:
        """ """
        n_vals = categorical_inputs.shape[0]
        ic(
            f"Cat inputs shape: {categorical_inputs.shape}",
            f"Numeric inputs shape: {numeric_inputs.shape}",
        )
        out = TimeSeriesTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
        )
        ic(out.shape)
        out = nn.Sequential(
            [
                nn.Dense(name="RegressionDense1", features=self.d_model * 2),
                nn.relu,
                nn.Dense(name="RegressionDense2", features=1),
            ],
            name="RegressionOutputChain",
        )(out)
        # ic(f"Out shape: {out.shape}")
        out = jnp.reshape(out, (out.shape[0], -1))
        out = nn.Dense(name="RegressionFlatten", features=n_vals)(out)
        return out


class MaskedTimeSeries(nn.Module):
    dataset: TabularDS
    d_model: int = 64 * 10
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        categorical_inputs: jnp.array,
        numeric_inputs: jnp.array,
    ) -> jnp.array:
        """ """
        # n_vals = categorical_inputs.shape[0]
        ic(
            f"Cat inputs shape: {categorical_inputs.shape}",
            f"Numeric inputs shape: {numeric_inputs.shape}",
        )
        out = TimeSeriesTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
        )
        # ic(f"Out shape: {out.shape}")
        # out = nn.Sequential(
        #     [
        #         nn.Dense(name="RegressionDense1", features=self.d_model * 2),
        #         nn.relu,
        #         nn.Dense(name="RegressionDense2", features=1),
        #     ],
        #     name="RegressionOutputChain",
        # )(out)
        # ic(f"Out shape: {out.shape}")
        ic(out.shape)
        categorical_out = nn.Dense(
            name="CategoricalOut", features=self.dataset.n_tokens
        )(out)

        numeric_out = nn.Sequential(
            [
                nn.Dense(name="numeric_dense_1", features=self.d_model * 6),
                nn.relu,
                nn.Dense(
                    name="numeric_dense_2", features=len(self.dataset.numeric_columns)
                ),
            ],
            name="NumericOutputChain",
        )(out)
        return categorical_out, numeric_out


# class PositionalEncoding(nn.Module):
#     d_model: int
#     dropout: float = 0.1
#     max_len = 10000

#     @nn.compact
#     def __call__(self, X):
#         # dropout = nn.Dropout(self.dropout) train=True
#         pe = jnp.zeros((1, self.max_len, self.d_model))
#         x = jnp.arange(self.max_len, dtype=jnp.float32).reshape(-1, 1) / jnp.power(
#             10000,
#             jnp.arange(0, self.d_model, 2, dtype=jnp.float32) / self.d_model,
#         )
#         pe = pe.at[:, :, 0::2].set(jnp.sin(x))
#         pe = pe.at[:, :, 1::2].set(jnp.cos(x))
#         X = X + pe[:, : X.shape[2], :]
#         # return dropout(X, deterministic=not train)

#         return X


class StaticPositionalEmbedding:
    def __init__(self, n_rows, n_features):
        # Initialize the static positional embedding
        self.positional_embedding = self._create_positional_embedding(
            n_rows, n_features
        )

    def _create_positional_embedding(self, n_rows, n_features):
        # Adjust n_features to be even for the sinusoidal generation, then clip
        n_features_adjusted = n_features + (n_features % 2)  # Make even if odd
        position = jnp.arange(n_rows)[:, jnp.newaxis]
        div_term = jnp.exp(
            jnp.arange(0, n_features_adjusted, 2)
            * -(jnp.log(10000.0) / n_features_adjusted)
        )
        pe = jnp.zeros((n_rows, n_features_adjusted))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        # Clip to the original n_features if n_features was odd
        return pe[:, :n_features]

    def add_positional_embedding(self, batch):
        # This function concatenates the positional embedding to each row in the batch.
        batch_size, n_rows, n_features = batch.shape
        # Replicate the positional embedding for the whole batch
        pe_batch = jnp.tile(self.positional_embedding, (batch_size, 1, 1))
        # Concatenate along the feature dimension
        return jnp.concatenate([batch, pe_batch], axis=2)


# class PositionalEncoding(nn.Module):
#     d_model: int  # Hidden dimensionality of the input.
#     max_len: int = 5000  # Maximum length of a sequence to expect.

#     def setup(self):
#         # Create matrix of [SeqLen, HiddenDim] representing the positional
# encoding for max_len inputs
#         pe = jnp.zeros((self.max_len, self.d_model))
#         position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
#         div_term = jnp.exp(
#             jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model)
#         )
#         pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
#         pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
#         pe = pe[None]
#         self.pe = jax.device_put(pe)

#     def __call__(self, x):
#         x = x + self.pe[:, : x.shape[1]]
#         return x
