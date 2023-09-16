# %%
import time
from dataclasses import dataclass, field
from datetime import datetime as dt
from itertools import chain

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from flax import linen as nn
from jax import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange

# Load and preprocess the dataset (assuming you have a CSV file)
df = pd.read_csv("../data/diamonds.csv")
df.head()


# %%
# def initialize_parameters(module):
#     if isinstance(module, (nn.Dense, nn.Conv2d)):
#         init.xavier_uniform_(module.weight, gain=1)
#         if module.bias is not None:
#             init.constant_(module.bias, 0.000)
#     elif isinstance(module, nn.Embedding):
#         init.uniform_(module.weight, 0.0, 0.5)
#     elif isinstance(module, nn.LayerNorm):
#         init.normal_(module.weight, mean=0, std=1)
#         init.constant_(module.bias, 0.01)


# %%


# class EarlyStopping:
#     def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.restore_best_weights = restore_best_weights
#         self.best_model = None
#         self.best_loss = None
#         self.counter = 0
#         self.status = ""

#     def __call__(self, model, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#             self.best_model = copy.deepcopy(model)
#         elif self.best_loss - val_loss > self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             self.best_model.load_state_dict(model.state_dict())
#         elif self.best_loss - val_loss < self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.status = f"Stopped on {self.counter}"
#                 if self.restore_best_weights:
#                     model.load_state_dict(self.best_model.state_dict())
#                 return True
#         self.status = f"{self.counter}/{self.patience}"
#         return False


# %%
@dataclass
class TabularDS:
    df: pd.DataFrame = field(repr=False)
    target_column: str
    seed: int = 42
    # device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # special_tokens: list =
    cat_mask_token: str = "[MASK]"
    special_tokens: list = field(
        default_factory=lambda: ["[PAD]", "[NUMERIC_MASK]", "[MASK]"]
    )

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

        self.scaler = StandardScaler()
        self.numeric_columns.remove(self.target_column[0])
        # self.numeric_columns = self.numeric_columns.remove(self.target_column[0])
        numeric_scaled = self.scaler.fit_transform(self.df[self.numeric_columns])
        self.df[self.numeric_columns] = numeric_scaled
        for col in self.category_columns:
            self.df[col] = self.df[col].map(self.token_dict)

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2
        )
        self.n_tokens = len(self.tokens)
        self.numeric_col_tokens = jnp.array(
            [self.token_dict[i] for i in self.numeric_columns]
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


# %%
df = pd.read_csv("../data/diamonds.csv")
df.head()
# %%
ds = TabularDS(df, "price")
# %%
ds
# %%


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


def expand_mask(mask):
    assert (
        mask.ndim > 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )
        self.k_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )
        self.v_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )  # TODO try making the same as q
        self.o_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )

    def __call__(
        self,
        X: jnp.array = None,
        q: jnp.array = None,
        k: jnp.array = None,
        v: jnp.array = None,
        mask: jnp.array = None,
        input_feed_forward: bool = False,
    ):
        if mask is not None:
            mask = expand_mask(mask)
        if X is not None:
            batch_size, seq_length, embed_dim = X.shape
            if input_feed_forward:
                qkv = self.qkv_proj(X)
                qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
                qkv = qkv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
            else:
                qkv = X

            q, k, v = jnp.array_split(qkv, 3, axis=-1)
        else:
            batch_size, seq_length, embed_dim = q.shape
            q = (
                self.q_proj(q)
                .reshape(batch_size, seq_length, self.num_heads, -1)
                .transpose(1, 2)
            )
            k = (
                self.k_proj(k)
                .reshape(batch_size, seq_length, self.num_heads, -1)
                .transpose(1, 2)  # TODO this may be wrong
            )
            v = (
                self.v_proj(v)
                .reshape(batch_size, seq_length, self.num_heads, -1)
                .transpose(1, 2)
            )

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o, attention


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float

    def setup(self):
        self.mha = nn.MultiHeadDotProductAttention(
            # features=self.d_model,
            num_heads=self.num_heads,
        )
        # self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

        self.ffn = nn.Dense(features=self.d_ff)
        self.ffn_out = nn.Dense(features=self.d_model)

    def __call__(self, X=None, q=None, kv=None, mask=None, train=True):
        # MultiHead Attention
        if X is not None:
            attn_output = self.mha(inputs_q=X, inputs_kv=X, mask=mask)
            # x = self.norm1(X + self.dropout(attn_output, deterministic=not train))
            x = self.norm1(X + attn_output)

        else:
            attn_output = self.mha(inputs_q=q, inputs_kv=kv, mask=mask)
            # x = self.norm1(kv + self.dropout(attn_output, deterministic=not train))
            x = self.norm1(kv + attn_output)

        # Feedforward Neural Network
        ffn_output = self.ffn_out(nn.relu(self.ffn(x)))
        # x = self.norm2(x + self.dropout(ffn_output, deterministic=not train))
        x = self.norm2(x + ffn_output)

        return x


class TransformerEncoderLayer(nn.Module):
    d_model: int
    n_heads: int

    def setup(self):
        self.multi_head_attention = MultiheadAttention(self.d_model, self.n_heads)

        self.feed_forward = nn.Sequential(
            [
                nn.Dense(2 * self.d_model),
                nn.relu,
                nn.Dense(self.d_model),
            ]
        )

        self.layernorm1 = nn.LayerNorm(self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model)
        # self.initialize_parameters()
        # self.apply(initialize_parameters)

    def __call__(
        self,
        X: jnp.array = None,
        q: jnp.array = None,
        k: jnp.array = None,
        v: jnp.array = None,
        mask: jnp.array = None,
        input_feed_forward: bool = False,
    ):
        attention_output = self.multi_head_attention(
            X, q, k, v, mask, input_feed_forward
        )
        out = self.layernorm1(q + attention_output)
        feed_forward_out = self.feed_forward(out)
        out = self.layernorm2(out + feed_forward_out)
        return out


# %%
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
        cat_mask = dataset.cat_mask_token
        cat_mask_token = dataset.token_dict[cat_mask]
        tensor = tensor.at[bit_mask].set(cat_mask_token)
        # tensor[bit_mask] = model.cat_mask_token
    return tensor


# %%
class TabTransformer(nn.Module):
    dataset: TabularDS
    d_model: int = 64
    n_heads: int = 4

    def __post_init__(self):
        self.n_tokens = len(self.dataset.tokens)
        self.tokens = self.dataset.tokens
        self.token_dict = self.dataset.token_dict
        self.cat_mask_token = jnp.array(self.token_dict["[MASK]"])
        self.numeric_mask_token = jnp.array(self.token_dict["[NUMERIC_MASK]"])
        self.numeric_columns = self.dataset.numeric_columns
        self.col_tokens = self.dataset.category_columns + self.dataset.numeric_columns
        self.col_indices = jnp.array(
            [self.tokens.index(col) for col in self.col_tokens]
        )
        self.numeric_indices = jnp.array(
            [self.tokens.index(col) for col in self.dataset.numeric_columns]
        )
        super().__post_init__()

    def setup(self):
        dataset = self.dataset
        n_heads = self.n_heads
        d_model = self.d_model

        # self.decoder_dict = {v: k for k, v in self.token_dict.items()}
        # Masks
        # Embedding layers for categorical features
        self.embedding = nn.Embed(num_embeddings=self.n_tokens, features=self.d_model)
        self.n_numeric_cols = len(dataset.numeric_columns)
        self.n_cat_cols = len(dataset.category_columns)
        self.col_tokens = dataset.category_columns + dataset.numeric_columns
        self.n_columns = self.n_numeric_cols + self.n_cat_cols
        # self.numeric_embeddings = NumericEmbedding(d_model=self.d_model)
        self.col_indices = jnp.array(
            [self.tokens.index(col) for col in self.col_tokens]
        )

        self.transformer_block1 = TransformerBlock(
            d_model=d_model, num_heads=n_heads, d_ff=d_model * 4, dropout_rate=0.1
        )
        self.transformer_block2 = TransformerBlock(
            d_model=d_model, num_heads=n_heads, d_ff=d_model * 4, dropout_rate=0.1
        )

    def __call__(
        self,
        numeric_inputs: jnp.array,
        categorical_inputs: jnp.array,
    ):
        # Embed column indices
        repeated_col_indices = jnp.tile(self.col_indices, (numeric_inputs.shape[0], 1))
        col_embeddings = self.embedding(repeated_col_indices)
        cat_embeddings = self.embedding(categorical_inputs)

        # expanded_num_inputs = num_inputs  # jnp.tile(num_inputs, num_inputs.shape[0])
        # expanded_num_inputs = num_inputs.unsqueeze(2).repeat(1, 1, self.d_model)
        # TODO implement no grad here
        repeated_numeric_indices = jnp.tile(
            self.numeric_indices, (numeric_inputs.shape[0], 1)
        )

        # repeated_numeric_indices = self.numeric_indices.unsqueeze(0).repeat(
        #     num_inputs.size(0), 1
        # )
        numeric_col_embeddings = self.embedding(repeated_numeric_indices)
        nan_mask = jnp.isnan(numeric_inputs)
        assert nan_mask.shape == numeric_col_embeddings.shape[:2]
        base_numeric = jnp.zeros_like(numeric_col_embeddings)

        base_numeric = jnp.where(
            nan_mask[:, :, None],
            numeric_col_embeddings * numeric_inputs[:, :, None],
            base_numeric,
        )

        base_numeric = jnp.where(
            nan_mask[:, :, None], self.numeric_mask_token, base_numeric
        )

        query_embeddings = jnp.concatenate([cat_embeddings, base_numeric], axis=1)
        out = self.transformer_block1(q=col_embeddings, kv=query_embeddings)
        out = self.transformer_block2(X=out)

        return out


class MTM(nn.Module):
    dataset: TabularDS
    d_model: int = 64
    n_heads: int = 4

    def __post_init__(self):
        self.n_tokens = len(self.dataset.tokens)
        self.tokens = self.dataset.tokens
        self.token_dict = self.dataset.token_dict
        self.cat_mask_token = jnp.array(self.token_dict["[MASK]"])
        self.numeric_mask_token = jnp.array(self.token_dict["[NUMERIC_MASK]"])
        self.numeric_columns = self.dataset.numeric_columns
        self.col_tokens = self.dataset.category_columns + self.dataset.numeric_columns
        self.col_indices = jnp.array(
            [self.tokens.index(col) for col in self.col_tokens]
        )
        self.numeric_indices = jnp.array(
            [self.tokens.index(col) for col in self.dataset.numeric_columns]
        )
        super().__post_init__()

    @nn.compact
    def __call__(
        self,
        numeric_inputs: jnp.array,
        categorical_inputs: jnp.array,
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

    def __post_init__(self):
        self.n_tokens = len(self.dataset.tokens)
        self.tokens = self.dataset.tokens
        self.token_dict = self.dataset.token_dict
        self.cat_mask_token = jnp.array(self.token_dict["[MASK]"])
        self.numeric_mask_token = jnp.array(self.token_dict["[NUMERIC_MASK]"])
        self.numeric_columns = self.dataset.numeric_columns
        self.col_tokens = self.dataset.category_columns + self.dataset.numeric_columns
        self.col_indices = jnp.array(
            [self.tokens.index(col) for col in self.col_tokens]
        )
        self.numeric_indices = jnp.array(
            [self.tokens.index(col) for col in self.dataset.numeric_columns]
        )
        super().__post_init__()

    @nn.compact
    def __call__(
        self,
        numeric_inputs: jnp.array,
        categorical_inputs: jnp.array,
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
