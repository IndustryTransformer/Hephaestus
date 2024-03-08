# %%
import jax.numpy as jnp
from flax import linen as nn
from icecream import ic
from jax.lax import stop_gradient

from ..utils.data_utils import TabularDS

# %%


class MultiheadAttention(nn.Module):
    n_heads: int
    d_model: int

    @nn.compact
    def __call__(self, q, k, v, input_feed_forward=True, mask=None):
        if input_feed_forward:
            q = nn.Dense(name="q_linear", features=self.d_model)(q)
            k = nn.Dense(features=self.d_model)(k)
            v = nn.Dense(features=self.d_model)(v)

        # print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
        attention_output, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        atten_out = attention_output.transpose(0, 2, 1).reshape(
            attention_output.shape[0], -1, self.d_model
        )
        out = nn.Dense(features=self.d_model)(atten_out)

        return out

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        # print(f"Matmul shape: {matmul_qk.shape}")
        d_k = q.shape[-1]
        matmul_qk = matmul_qk / jnp.sqrt(d_k)

        if mask is not None:
            matmul_qk += mask * -1e9

        attention_weights = nn.softmax(matmul_qk, axis=-1)

        output = jnp.matmul(attention_weights, v)
        # print(f"Scaled Output shape: {output.shape}")

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
        ic(attn_output.shape, q.shape, k.shape, v.shape)
        out = nn.LayerNorm()(q + attn_output)
        ff_out = nn.Sequential(
            [nn.Dense(self.d_model * 2), nn.relu, nn.Dense(self.d_model)]
        )
        out = nn.LayerNorm()(out + ff_out(out))

        return out


class TimeSeriesTransformer(nn.Module):
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

        # Embed column indices
        col_embeddings = embedding(jnp.array(self.dataset.col_indices))
        cat_embeddings = embedding(categorical_inputs)
        # TODO implement no grad here
        repeated_numeric_indices = jnp.tile(
            self.dataset.numeric_indices, (numeric_inputs.shape[0], 1)
        )

        # Nan Masking
        numeric_col_embeddings = embedding(repeated_numeric_indices)
        nan_mask = stop_gradient(jnp.isnan(numeric_inputs))
        base_numeric = jnp.zeros_like(numeric_col_embeddings)
        numeric_inputs = stop_gradient(jnp.where(nan_mask, 0.0, numeric_inputs))
        base_numeric = jnp.where(
            nan_mask[:, :, None],
            base_numeric,
            numeric_col_embeddings * numeric_inputs[:, :, None],
        )

        base_numeric = jnp.where(
            nan_mask[:, :, None], self.dataset.numeric_mask_token, base_numeric
        )
        # End Nan Masking
        kv_embeddings = jnp.concatenate([cat_embeddings, base_numeric], axis=1)
        kv_embeddings = jnp.expand_dims(
            kv_embeddings.reshape(-1, kv_embeddings.shape[2]), axis=0
        )
        # print(f"KV Embedding shape: {kv_embeddings.shape}")
        ic(col_embeddings.shape, kv_embeddings.shape)
        out = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            dropout_rate=0.1,
        )(q=col_embeddings, k=kv_embeddings, v=kv_embeddings)
        ic(out.shape)
        # print(f"First MHA out shape: {out.shape}")
        out = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            dropout_rate=0.1,
        )(
            q=col_embeddings, k=out, v=out
        )  # Check if we should reuse the col embeddings here
        # print(f"Second MHA out shape: {out.shape}")
        ic(out.shape)
        return out


# %%
class TabTransformer(nn.Module):
    dataset: TabularDS
    d_model: int = 64
    n_heads: int = 4

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

        # Embed column indices
        repeated_col_indices = jnp.tile(
            self.dataset.col_indices, (numeric_inputs.shape[0], 1)
        )
        col_embeddings = embedding(repeated_col_indices)
        cat_embeddings = embedding(categorical_inputs)
        # TODO implement no grad here
        repeated_numeric_indices = jnp.tile(
            self.dataset.numeric_indices, (numeric_inputs.shape[0], 1)
        )

        # Nan Masking
        numeric_col_embeddings = embedding(repeated_numeric_indices)
        nan_mask = stop_gradient(jnp.isnan(numeric_inputs))
        base_numeric = jnp.zeros_like(numeric_col_embeddings)
        numeric_inputs = stop_gradient(jnp.where(nan_mask, 0.0, numeric_inputs))
        base_numeric = jnp.where(
            nan_mask[:, :, None],
            base_numeric,
            numeric_col_embeddings * numeric_inputs[:, :, None],
        )

        base_numeric = jnp.where(
            nan_mask[:, :, None], self.dataset.numeric_mask_token, base_numeric
        )
        # End Nan Masking
        kv_embeddings = jnp.concatenate([cat_embeddings, base_numeric], axis=1)
        # print(
        #     f"KV Embedding shape: {kv_embeddings.shape}",
        #     f"Col Embedding shape: {col_embeddings.shape}",
        # )

        out = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            dropout_rate=0.1,
        )(q=col_embeddings, k=kv_embeddings, v=kv_embeddings)

        out = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            dropout_rate=0.1,
        )(q=out, k=out, v=out)

        return out


class MTM(nn.Module):
    dataset: TabularDS
    d_model: int = 64
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        categorical_inputs: jnp.array,
        numeric_inputs: jnp.array,
    ):
        out = TabTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
        )
        categorical_out = nn.Dense(
            name="categorical_out", features=self.dataset.n_tokens
        )(out)

        out = jnp.reshape(out, (out.shape[0], -1))
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


class TRM(nn.Module):
    dataset: TabularDS
    d_model: int = 64
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        categorical_inputs: jnp.array,
        numeric_inputs: jnp.array,
    ):
        out = TabTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
        )

        out = nn.Sequential(
            [
                nn.Dense(name="RegressionDense1", features=self.d_model * 2),
                nn.relu,
                nn.Dense(name="RegressionDense2", features=1),
            ],
            name="RegressionOutputChain",
        )(out)
        out = jnp.reshape(out, (out.shape[0], -1))
        out = nn.Dense(name="RegressionFlatten", features=1)(out)
        return out


class TimeSeriesRegression(nn.Module):
    dataset: TabularDS
    d_model: int = 64
    n_heads: int = 4

    @nn.compact
    def __call__(
        self,
        categorical_inputs: jnp.array,
        numeric_inputs: jnp.array,
    ):
        # print(
        #     f"Cat inputs shape: {categorical_inputs.shape}",
        #     f"Numeric inputs shape: {numeric_inputs.shape}",
        # )
        out = TimeSeriesTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
        )
        # print(f"Out shape: {out.shape}")
        out = nn.Sequential(
            [
                nn.Dense(name="RegressionDense1", features=self.d_model * 2),
                nn.relu,
                nn.Dense(name="RegressionDense2", features=1),
            ],
            name="RegressionOutputChain",
        )(out)
        # print(f"Out shape: {out.shape}")
        out = jnp.reshape(out, (out.shape[0], -1))
        out = nn.Dense(name="RegressionFlatten", features=1)(out)
        return out
