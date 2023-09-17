# %%
import time
from dataclasses import dataclass, field
from itertools import chain

import jax
import jax.numpy as jnp
import pandas as pd
from flax import linen as nn
from flax import struct  # Flax dataclasses
from jax import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %%
@dataclass
class TabularDS:
    df: pd.DataFrame = field(repr=False)
    target_column: str
    seed: int = 42
    special_tokens: list = field(
        default_factory=lambda: ["[PAD]", "[NUMERIC_MASK]", "[MASK]"]
    )
    cat_mask: str = "[MASK]"
    cat_mask_token: int = field(init=False)
    n_tokens: int = field(init=False)
    n_cat_cols: int = field(init=False)
    n_numeric_cols: int = field(init=False)
    numeric_col_tokens: jnp.array = field(init=False)
    category_columns: list = field(init=False)
    col_tokens: list = field(init=False)
    numeric_mask_token: list = field(init=False)
    numeric_indices: jnp.array = field(init=False)
    col_indices: jnp.array = field(init=False)

    def __post_init__(self):
        self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )  # This is where randomness is introduced
        self.category_columns = self.df.select_dtypes(
            include=["object"]
        ).columns.tolist()
        self.numeric_columns = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.target_column = [self.target_column]
        self.tokens = list(
            chain(
                self.special_tokens,
                self.df.columns.to_list(),
                list(set(self.df[self.category_columns].values.flatten().tolist())),
            )
        )
        self.token_dict = {token: i for i, token in enumerate(self.tokens)}
        self.token_decoder_dict = {i: token for i, token in enumerate(self.tokens)}
        self.cat_mask_token = self.token_dict[self.cat_mask]
        self.scaler = StandardScaler()
        self.numeric_columns.remove(self.target_column[0])
        self.col_tokens = self.category_columns + self.numeric_columns
        self.n_cat_cols = len(self.category_columns)
        # self.numeric_columns = self.numeric_columns.remove(self.target_column[0])
        numeric_scaled = self.scaler.fit_transform(self.df[self.numeric_columns])
        self.df[self.numeric_columns] = numeric_scaled
        self.n_numeric_cols = len(self.numeric_columns)
        for col in self.category_columns:
            self.df[col] = self.df[col].map(self.token_dict)

        self.numeric_indices = jnp.array(
            [self.tokens.index(col) for col in self.numeric_columns]
        )
        self.numeric_mask_token = jnp.array(self.token_dict["[NUMERIC_MASK]"])
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2
        )
        self.n_tokens = len(self.tokens)
        self.numeric_col_tokens = jnp.array(
            [self.token_dict[i] for i in self.numeric_columns]
        )
        self.col_indices = jnp.array(
            [self.tokens.index(col) for col in self.col_tokens]
        )
        X_train_numeric = self.X_train[self.numeric_columns]
        X_train_categorical = self.X_train[self.category_columns]
        X_test_numeric = self.X_test[self.numeric_columns]
        X_test_categorical = self.X_test[self.category_columns]

        self.X_train_numeric = jnp.array(X_train_numeric.values)

        self.X_train_categorical = jnp.array(X_train_categorical.values)

        self.X_test_numeric = jnp.array(X_test_numeric.values)

        self.X_test_categorical = jnp.array(X_test_categorical.values)

        self.y_train = jnp.array(self.y_train.values)

        self.y_test = jnp.array(self.y_test.values)


class MultiheadAttention(nn.Module):
    n_heads: int
    d_model: int

    @nn.compact
    def __call__(self, q, k, v, input_feed_forward=False, mask=None):
        if input_feed_forward:
            q = nn.Dense(name="q_linear", features=self.d_model)(q)
            k = nn.Dense(features=self.d_model)(k)
            v = nn.Dense(features=self.d_model)(v)

        attention_output, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        atten_out = attention_output.transpose(0, 2, 1).reshape(
            attention_output.shape[0], -1, self.d_model
        )
        out = nn.Dense(features=self.d_model)(atten_out)

        return out

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output, attention_weights
        """
        matmul_qk = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        d_k = q.shape[-1]
        matmul_qk = matmul_qk / jnp.sqrt(d_k)

        if mask is not None:
            matmul_qk += mask * -1e9

        attention_weights = nn.softmax(matmul_qk, axis=-1)

        output = jnp.matmul(attention_weights, v)

        return output, attention_weights


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float

    @nn.compact
    def __call__(self, q, k, v, mask=None, train=True):
        attn_output = MultiheadAttention(n_heads=self.n_heads, d_model=self.d_model)(
            q, k, v, mask=mask
        )
        out = nn.LayerNorm()(q + attn_output)
        ff_out = nn.Sequential(
            [nn.Dense(self.d_model * 2), nn.relu, nn.Dense(self.d_model)]
        )
        out = nn.LayerNorm()(out + ff_out(out))

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

        numeric_col_embeddings = embedding(repeated_numeric_indices)
        nan_mask = jnp.isnan(numeric_inputs)
        assert nan_mask.shape == numeric_col_embeddings.shape[:2]
        base_numeric = jnp.zeros_like(numeric_col_embeddings)
        numeric_inputs = jnp.where(nan_mask, 0, numeric_inputs)
        base_numeric = jnp.where(
            nan_mask[:, :, None],
            numeric_col_embeddings * numeric_inputs[:, :, None],
            base_numeric,
        )

        base_numeric = jnp.where(
            nan_mask[:, :, None], self.dataset.numeric_mask_token, base_numeric
        )

        query_embeddings = jnp.concatenate([cat_embeddings, base_numeric], axis=1)
        out = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            dropout_rate=0.1,
        )(q=col_embeddings, k=query_embeddings, v=query_embeddings)

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
        # categorical_out = nn.softmax(categorical_out, axis=-1)
        numeric_out = jnp.reshape(out, (out.shape[0], -1))
        numeric_out = nn.Sequential(
            [
                nn.Dense(name="numeric_dense1", features=self.d_model * 4),
                nn.relu,
                nn.Dense(
                    name="numeric_dense2", features=len(self.dataset.numeric_columns)
                ),
            ]
        )(numeric_out)

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
                nn.Dense(name="dense1", features=self.d_model * 2),
                nn.relu,
                nn.Dense(name="dense2", features=1),
            ]
        )(out)
        out = jnp.reshape(out, (out.shape[0], -1))
        return out


# %%


@struct.dataclass
class ModelInputs:
    categorical_mask: jnp.ndarray
    numeric_mask: jnp.ndarray
    numeric_targets: jnp.ndarray
    categorical_targets: jnp.ndarray


def create_mi(
    dataset: TabularDS,
    idx: int = None,
    batch_size: int = None,
    set: str = "train",
    probability=0.8,
    device=None,
):
    if device is None:
        device = jax.devices()[0]
    if set == "train":
        categorical_values = dataset.X_train_categorical
        numeric_targets = dataset.X_train_numeric
    elif set == "test":
        categorical_values = dataset.X_test_categorical
        numeric_targets = dataset.X_test_numeric
    else:
        raise ValueError("set must be either 'train' or 'test'")

    if idx is None:
        idx = 0
    if batch_size is None:
        batch_size = numeric_targets.shape[0]

    categorical_values = categorical_values[idx : idx + batch_size, :]
    numeric_targets = numeric_targets[idx : idx + batch_size, :]

    categorical_mask = mask_tensor(categorical_values, dataset, probability=probability)
    numeric_mask = mask_tensor(numeric_targets, dataset, probability=probability)

    numeric_col_tokens = dataset.numeric_col_tokens.clone()
    repeated_numeric_col_tokens = jnp.tile(
        numeric_col_tokens, (categorical_values.shape[0], 1)
    )
    categorical_targets = jnp.concatenate(
        [
            categorical_values,
            repeated_numeric_col_tokens,
        ],
        axis=1,
    )
    categorical_targets = categorical_targets.at[jnp.isnan(categorical_targets)].set(
        dataset.cat_mask_token
    )

    numeric_targets = numeric_targets.at[jnp.isnan(numeric_targets)].set(0.0)

    mi = ModelInputs(
        categorical_mask=jax.device_put(categorical_mask, device=device),
        numeric_mask=jax.device_put(numeric_mask, device=device),
        numeric_targets=jax.device_put(numeric_targets, device=device),
        categorical_targets=jax.device_put(categorical_targets, device=device),
    )
    return mi


def show_mask_pred(params, model, i, dataset, probability=0.8, set="train"):
    mi = create_mi(dataset, idx=i, batch_size=1, set=set, probability=probability)

    logits, numeric_preds = model.apply(
        {"params": params}, mi.categorical_mask, mi.numeric_mask
    )
    cat_preds = logits.argmax(axis=-1)

    # Get the words from the tokens
    decoder_dict = dataset.token_decoder_dict
    cat_preds = [decoder_dict[i.item()] for i in cat_preds[0]]

    results_dict = {k: cat_preds[i] for i, k in enumerate(dataset.col_tokens)}
    for i, k in enumerate(dataset.col_tokens[dataset.n_cat_cols :]):
        results_dict[k] = numeric_preds[0][i].item()
    # Get the masked values
    categorical_masked = [decoder_dict[i.item()] for i in mi.categorical_mask[0]]
    numeric_masked = mi.numeric_mask[0].tolist()
    masked_values = categorical_masked + numeric_masked
    # zip the masked values with the column names
    masked_dict = dict(zip(dataset.col_tokens, masked_values))
    # Get the original values
    categorical_values = [
        decoder_dict[i.item()]
        for i in mi.categorical_targets[0][0 : dataset.n_cat_cols]
    ]

    numeric_values = mi.numeric_targets.tolist()[0]
    original_values = categorical_values
    original_values.extend(numeric_values)
    # zip the original values with the column names
    original_dict = dict(zip(dataset.col_tokens, original_values))
    # print(numeric_masked)
    # print(categorical_masked)
    result_dict = {
        "masked": masked_dict,
        "actual": original_dict,
        "pred": results_dict,
    }

    return result_dict


def mask_tensor(tensor, dataset, probability=0.8):
    if tensor.dtype == "float32" or tensor.dtype == "float64":
        is_numeric = True
    elif tensor.dtype == "int32" or tensor.dtype == "int64":
        is_numeric = False
    else:
        raise ValueError(f"Task {tensor.dtype} not supported.")

    tensor = tensor.copy()
    seed = int(time.time() * 1000000)
    key = random.PRNGKey(seed)
    bit_mask = random.normal(key=key, shape=tensor.shape) > probability
    if is_numeric:
        tensor = tensor.at[bit_mask].set(float("nan"))
    else:
        tensor = tensor.at[bit_mask].set(dataset.cat_mask_token)
    return tensor
