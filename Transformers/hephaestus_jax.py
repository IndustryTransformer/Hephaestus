# %%
import time
from dataclasses import dataclass, field
from datetime import datetime as dt
from itertools import chain

import jax
import jax.numpy as jnp
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
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

        self.ffn = nn.Dense(features=self.d_ff)
        self.ffn_out = nn.Dense(features=self.d_model)

    def __call__(self, X=None, q=None, kv=None, mask=None, train=True):
        # MultiHead Attention
        if X is not None:
            attn_output = self.mha(inputs_q=X, inputs_kv=X, mask=mask)
            x = self.norm1(X + self.dropout(attn_output, deterministic=not train))

        else:
            attn_output = self.mha(inputs_q=q, inputs_kv=kv, mask=mask)
            x = self.norm1(kv + self.dropout(attn_output, deterministic=not train))

        # Feedforward Neural Network
        ffn_output = self.ffn_out(nn.relu(self.ffn(x)))
        x = self.norm2(x + self.dropout(ffn_output, deterministic=not train))

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
def mask_tensor(tensor, model, probability=0.8):
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
        tensor = tensor.at[bit_mask].set(float("-Inf"))
        # tensor[bit_mask] = jnp.array(float("-Inf"))
    else:
        tensor = tensor.at[bit_mask].set(model.cat_mask_token)
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
        super().__post_init__()

    def setup(self):
        dataset = self.dataset
        n_heads = self.n_heads
        d_model = self.d_model
        n_tokens = len(dataset.tokens)
        self.model_name = "Kai is Great"

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
        self.numeric_indices = jnp.array(
            [self.tokens.index(col) for col in dataset.numeric_columns]
        )
        self.transformer_block1 = TransformerBlock(
            d_model=d_model, num_heads=n_heads, d_ff=d_model * 4, dropout_rate=0.1
        )
        self.transformer_block2 = TransformerBlock(
            d_model=d_model, num_heads=n_heads, d_ff=d_model * 4, dropout_rate=0.1
        )
        # self.transformer_encoder1 = TransformerEncoderLayer(d_model, n_heads=n_heads)
        # self.transformer_encoder2 = TransformerEncoderLayer(d_model, n_heads=n_heads)
        self.regressor = nn.Sequential(
            [
                nn.Dense(d_model * 2),
                nn.relu,  # Error here
                nn.Dense(1),
            ]
            # nn.relu,
        )

        self.mlm_decoder = nn.Sequential(
            [nn.Dense(n_tokens)]
        )  # TODO try making more complex

        self.mnm_decoder = nn.Sequential(
            [
                nn.Dense(
                    self.n_columns * self.d_model, self.d_model * 4
                ),  # Try making more complex
                nn.relu,
                nn.Dense(self.n_numeric_cols),
            ]
        )

        self.flatten_layer = nn.Dense(len(self.col_tokens), 1)
        # self.apply(initialize_parameters)

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
        inf_mask = numeric_inputs == float("-inf")

        base_numeric = jnp.zeros_like(numeric_col_embeddings)

        num_embeddings = (
            numeric_col_embeddings[~inf_mask] * numeric_inputs[~inf_mask][:, None]
        )
        # Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)``
        base_numeric = base_numeric.at[~inf_mask].set(num_embeddings)
        base_numeric = base_numeric.at[inf_mask].set(
            self.embedding(self.numeric_mask_token)
        )

        query_embeddings = jnp.concatenate([cat_embeddings, base_numeric], axis=1)
        out = self.transformer_block1(q=col_embeddings, kv=query_embeddings)
        out = self.transformer_block2(X=out)
        # out = self.transformer_encoder1(
        #     q=col_embeddings,
        #     k=query_embeddings,
        #     v=query_embeddings
        #     # col_embeddings, query_embeddings, query_embeddings
        # )
        # out = self.transformer_encoder2(out, out, out)
        return out


class MTM(nn.Module):
    dataset: TabularDS
    d_model: int = 64
    n_heads: int = 4

    @nn.compact
    def __call__(self, numeric_inputs: jnp.array, categorical_inputs: jnp.array):
        out = TabTransformer(self.dataset, self.d_model, self.n_heads)(
            numeric_inputs, categorical_inputs
        )
        cat_out = self.mlm_decoder(out)

        numeric_out = jnp.reshape(out, (out.shape[0], -1))
        print(f"numeric_out shape: {numeric_out.shape}")
        numeric_out = self.mnm_decoder(numeric_out)
        print(f"numeric_out shape decoded: {numeric_out.shape}")
        return cat_out, numeric_out


# %%
def show_mask_pred(i, model, dataset, probability):
    numeric_values = dataset.X_train_numeric[i : i + 1, :]
    categorical_values = dataset.X_train_categorical[i : i + 1, :]
    numeric_masked = mask_tensor(numeric_values, model, probability=probability)
    categorical_masked = mask_tensor(categorical_values, model, probability=probability)
    # Predictions

    cat_preds, numeric_preds = model(numeric_masked, categorical_masked, task="mlm")
    # Get the predicted tokens from cat_preds
    cat_preds = cat_preds.argmax(dim=2)
    # Get the words from the tokens
    decoder_dict = dataset.token_decoder_dict
    cat_preds = [decoder_dict[i.item()] for i in cat_preds[0]]

    results_dict = {k: cat_preds[i] for i, k in enumerate(model.col_tokens)}
    for i, k in enumerate(model.col_tokens[model.n_cat_cols :]):
        results_dict[k] = numeric_preds[0][i].item()
    # Get the masked values
    categorical_masked = [decoder_dict[i.item()] for i in categorical_masked[0]]
    numeric_masked = numeric_masked[0].tolist()
    masked_values = categorical_masked + numeric_masked
    # zip the masked values with the column names
    masked_values = dict(zip(model.col_tokens, masked_values))
    # Get the original values
    categorical_values = [decoder_dict[i.item()] for i in categorical_values[0]]
    numeric_values = numeric_values[0].tolist()
    original_values = categorical_values + numeric_values
    # zip the original values with the column names
    original_values = dict(zip(model.col_tokens, original_values))
    # print(numeric_masked)
    # print(categorical_masked)
    result_dict = {
        "actual": original_values,
        "masked": masked_values,
        "pred": results_dict,
    }

    return result_dict


# %%
def mtm(model, dataset, model_name, epochs=100, batch_size=1000, lr=0.001):
    # Masked Tabular Modeling
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    numeric_loss_scaler = 15
    summary_writer = SummaryWriter("runs/" + model_name)
    early_stopping = EarlyStopping(patience=20, min_delta=0.0001)
    batch_count = 0
    model.train()
    # Tqdm for progress bar with loss and epochs displayed
    pbar = trange(epochs, desc="Epochs", leave=True)
    for epoch in pbar:  # trange(epochs, desc="Epochs", leave=True):
        for i in range(0, dataset.X_train_numeric.size(0), batch_size):
            numeric_values = dataset.X_train_numeric[i : i + batch_size, :]
            categorical_values = dataset.X_train_categorical[i : i + batch_size, :]
            numeric_masked = mask_tensor(numeric_values, model, probability=0.8)
            categorical_masked = mask_tensor(categorical_values, model, probability=0.8)
            optimizer.zero_grad()
            cat_preds, numeric_preds = model(
                numeric_masked, categorical_masked, task="mlm"
            )
            cat_targets = torch.cat(
                (
                    categorical_values,
                    model.numeric_indices.expand(categorical_values.size(0), -1),
                ),
                dim=1,
            )

            cat_preds = cat_preds.permute(0, 2, 1)  # TODO investigate as possible bug

            cat_loss = ce_loss(cat_preds, cat_targets)
            numeric_loss = (
                mse_loss(numeric_preds, numeric_values) * numeric_loss_scaler
            )  # Hyper param
            loss = cat_loss + numeric_loss  # TODO Look at scaling
            loss.backward()
            optimizer.step()
            batch_count += 1
            learning_rate = optimizer.param_groups[0]["lr"]
            summary_writer.add_scalar("LossTrain/agg_mask", loss.item(), batch_count)
            summary_writer.add_scalar(
                "LossTrain/mlm_loss", cat_loss.item(), batch_count
            )
            summary_writer.add_scalar(
                "LossTrain/mnm_loss", numeric_loss.item(), batch_count
            )
            summary_writer.add_scalar("Metrics/mtm_lr", learning_rate, batch_count)

        with torch.no_grad():
            numeric_values = dataset.X_test_numeric
            categorical_values = dataset.X_test_categorical
            numeric_masked = mask_tensor(numeric_values, model, probability=0.8)
            categorical_masked = mask_tensor(categorical_values, model, probability=0.8)
            optimizer.zero_grad()
            cat_preds, numeric_preds = model(
                numeric_masked, categorical_masked, task="mlm"
            )
            cat_targets = torch.cat(
                (
                    categorical_values,
                    model.numeric_indices.expand(categorical_values.size(0), -1),
                ),
                dim=1,
            )

            cat_preds = cat_preds.permute(0, 2, 1)

            test_cat_loss = ce_loss(cat_preds, cat_targets)
            test_numeric_loss = (
                mse_loss(numeric_preds, numeric_values) * numeric_loss_scaler
            )  # Hyper param
            test_loss = test_cat_loss + test_numeric_loss

        summary_writer.add_scalar("LossTest/agg_loss", test_loss.item(), batch_count)

        summary_writer.add_scalar(
            "LossTest/mlm_loss", test_cat_loss.item(), batch_count
        )
        summary_writer.add_scalar(
            "LossTest/mnm_loss", test_numeric_loss.item(), batch_count
        )
        early_stopping_status = early_stopping(model, test_loss)
        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} Loss: {loss.item():,.4f} "
            + f"Test Loss: {test_loss.item():,.4f}"
            + f" Early Stopping: {early_stopping.status}"
        )
        if early_stopping_status:
            break


# %%
def fine_tune_model(
    model, dataset, n_rows, model_name, epochs=100, lr=0.01, early_stop=True
):
    # Regression Model

    batch_size = 1000

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
    model_name = f"{model_name}_{n_rows}_{model_time}"
    if early_stop:
        early_stopping = EarlyStopping(patience=50, min_delta=0.0001)
    else:
        early_stopping = None
    summary_writer = SummaryWriter("runs/" + model_name)
    if n_rows is None:
        n_rows = dataset.X_train_numeric.size(0)
    if n_rows > dataset.X_train_numeric.size(0):
        raise ValueError(
            f"n_rows ({n_rows}) must be less than or equal to "
            + f"{dataset.X_train_numeric.size(0)}"
        )

    train_set_size = n_rows  # dataset.X_train_numeric.size(0)
    batch_count = 0
    model.train()
    pbar = trange(epochs, desc=f"Epochs, Model: {model_name}")
    if batch_size > train_set_size:
        batch_size = train_set_size
    for epoch in pbar:
        for i in range(0, train_set_size, batch_size):
            num_inputs = dataset.X_train_numeric[i : i + batch_size, :]
            cat_inputs = dataset.X_train_categorical[i : i + batch_size, :]
            optimizer.zero_grad()
            y_pred = model(num_inputs, cat_inputs)
            loss = loss_fn(y_pred, dataset.y_train[i : i + batch_size, :])
            loss.backward()
            optimizer.step()
            batch_count += 1
            learning_rate = optimizer.param_groups[0]["lr"]
            summary_writer.add_scalar(
                "LossTrain/regression_loss", loss.item(), batch_count
            )
            summary_writer.add_scalar(
                "Metrics/regression_lr", learning_rate, batch_count
            )

        # Test set
        with torch.no_grad():
            y_pred = model(dataset.X_test_numeric, dataset.X_test_categorical)
            loss_test = loss_fn(y_pred, dataset.y_test)
            summary_writer.add_scalar(
                "LossTest/regression_loss", loss_test.item(), batch_count
            )
        if early_stopping is not None:
            early_stopping_status = early_stopping(model, loss_test)
        else:
            early_stopping_status = False
        pbar.set_description(
            f"Epoch {epoch+1}/{epochs} "
            + f"n_rows: {n_rows:,} "
            + f"Loss: {loss.item():,.2f} "
            + f"Test Loss: {loss_test.item():,.2f} "
            + f"Early Stopping: {early_stopping_status}"
        )
        if early_stopping_status:
            break
    best_loss = early_stopping.best_loss if early_stopping is not None else loss_test
    return {"n_rows": n_rows, "test_loss": best_loss.item()}


# %%


def regression_actuals_preds(model, dataset):
    with torch.no_grad():
        y_pred = model(
            dataset.X_test_numeric, dataset.X_test_categorical, task="regression"
        )
        y = dataset.y_test
    df = pd.DataFrame({"actual": y.flatten(), "pred": y_pred.flatten()})
    df["pred"] = df["pred"].round()
    df["error"] = df["actual"] - df["pred"]
    return df
