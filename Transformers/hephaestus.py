# %%
import copy
from dataclasses import dataclass, field
from datetime import datetime as dt
from itertools import chain

# import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange

# Load and preprocess the dataset (assuming you have a CSV file)
df = pd.read_csv("../data/diamonds.csv")
df.head()


# %%
def initialize_parameters(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            init.constant_(module.bias, 0.000)
    elif isinstance(module, nn.Embedding):
        init.uniform_(module.weight, 0.0, 0.5)
    elif isinstance(module, nn.LayerNorm):
        init.normal_(module.weight, mean=0, std=1)
        init.constant_(module.bias, 0.01)


# %%


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.counter}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False


# %%
@dataclass
class TabularDS:
    df: pd.DataFrame = field(repr=False)
    target_column: str
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        X_train_numeric = self.X_train[self.numeric_columns]
        X_train_categorical = self.X_train[self.category_columns]
        X_test_numeric = self.X_test[self.numeric_columns]
        X_test_categorical = self.X_test[self.category_columns]

        self.X_train_numeric = torch.tensor(
            X_train_numeric.values, dtype=torch.float
        ).to(self.device)

        self.X_train_categorical = torch.tensor(
            X_train_categorical.values, dtype=torch.long
        ).to(self.device)

        self.X_test_numeric = torch.tensor(X_test_numeric.values, dtype=torch.float).to(
            self.device
        )

        self.X_test_categorical = torch.tensor(
            X_test_categorical.values, dtype=torch.long
        ).to(self.device)

        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float).to(
            self.device
        )

        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float).to(
            self.device
        )


# %%
# df = pd.read_csv("../data/diamonds.csv")
# df.head()
# # %%
# ds = TabularDS(df, "price")
# # %%
# ds
# # %%
# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # self.initialize_parameters()
        self.apply(initialize_parameters)

    def forward(self, q, k, v, mask=None, input_feed_forward=False):
        batch_size = q.size(0)

        if input_feed_forward:
            q = (
                self.q_linear(q)
                .view(batch_size, -1, self.n_heads, self.d_head)
                .transpose(1, 2)
            )
            k = (
                self.k_linear(k)
                .view(batch_size, -1, self.n_heads, self.d_head)
                .transpose(1, 2)
            )
            v = (
                self.v_linear(v)
                .view(batch_size, -1, self.n_heads, self.d_head)
                .transpose(1, 2)
            )

        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        out = self.out_linear(attn_output)
        return out

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        d_k = q.size(-1)
        scaled_attention_logits = matmul_qk / (d_k**0.5)

        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerEncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
            # nn.Linear(4 * d_model, d_model * 4),
            # nn.ReLU(),
            # nn.Linear(d_model * 4, d_model),
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        # self.initialize_parameters()
        self.apply(initialize_parameters)

    def forward(self, q, k, v, mask=None, input_feed_forward=False):
        attn_output = self.multi_head_attention(q, k, v, mask, input_feed_forward)
        out1 = self.layernorm1(q + attn_output)

        ff_output = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + ff_output)

        return out2


# %%
def mask_tensor(tensor, model, probability=0.8):
    if tensor.dtype == torch.float32:
        is_numeric = True
    elif tensor.dtype == torch.int32 or tensor.dtype == torch.int64:
        is_numeric = False
    else:
        raise ValueError(f"Task {tensor.dtype} not supported.")

    tensor = tensor.clone()
    bit_mask = torch.rand(tensor.shape) > probability
    if is_numeric:
        tensor[bit_mask] = torch.tensor(float("-Inf"))
    else:
        tensor[bit_mask] = model.cat_mask_token
    return tensor.to(model.device)


# %%
class TabTransformer(nn.Module):
    def __init__(
        self,
        dataset,
        d_model=64,
        n_heads=4,
    ):
        super(TabTransformer, self).__init__()
        self.device = dataset.device
        self.d_model = d_model
        self.tokens = dataset.tokens
        self.token_dict = dataset.token_dict
        # self.decoder_dict = {v: k for k, v in self.token_dict.items()}
        # Masks
        self.cat_mask_token = torch.tensor(self.token_dict["[MASK]"]).to(self.device)
        self.numeric_mask_token = torch.tensor(self.token_dict["[NUMERIC_MASK]"]).to(
            self.device
        )

        self.n_tokens = len(self.tokens)  # TODO Make this
        # Embedding layers for categorical features
        self.embeddings = nn.Embedding(self.n_tokens, self.d_model).to(self.device)
        self.n_numeric_cols = len(dataset.numeric_columns)
        self.n_cat_cols = len(dataset.category_columns)
        self.col_tokens = dataset.category_columns + dataset.numeric_columns
        self.n_columns = self.n_numeric_cols + self.n_cat_cols
        # self.numeric_embeddings = NumericEmbedding(d_model=self.d_model)
        self.col_indices = torch.tensor(
            [self.tokens.index(col) for col in self.col_tokens], dtype=torch.long
        ).to(self.device)
        self.numeric_indices = torch.tensor(
            [self.tokens.index(col) for col in dataset.numeric_columns],
            dtype=torch.long,
        ).to(self.device)
        self.transformer_encoder1 = TransformerEncoderLayer(
            d_model, n_heads=n_heads
        ).to(self.device)
        self.transformer_encoder2 = TransformerEncoderLayer(
            d_model, n_heads=n_heads
        ).to(self.device)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 1),
            # nn.ReLU(),
        ).to(self.device)

        self.mlm_decoder = nn.Sequential(nn.Linear(d_model, self.n_tokens)).to(
            self.device
        )  # TODO try making more complex

        self.mnm_decoder = nn.Sequential(
            nn.Linear(
                self.n_columns * self.d_model, self.d_model * 4
            ),  # Try making more complex
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.n_numeric_cols),
        ).to(self.device)

        self.flatten_layer = nn.Linear(len(self.col_tokens), 1).to(self.device)
        self.apply(initialize_parameters)

    def forward(self, num_inputs, cat_inputs, task="regression"):
        # Embed column indices
        repeated_col_indices = self.col_indices.unsqueeze(0).repeat(
            num_inputs.size(0), 1
        )
        col_embeddings = self.embeddings(repeated_col_indices)

        cat_embeddings = self.embeddings(cat_inputs)

        expanded_num_inputs = num_inputs.unsqueeze(2).repeat(1, 1, self.d_model)
        with torch.no_grad():
            repeated_numeric_indices = self.numeric_indices.unsqueeze(0).repeat(
                num_inputs.size(0), 1
            )
            numeric_col_embeddings = self.embeddings(repeated_numeric_indices)

            inf_mask = (expanded_num_inputs == float("-inf")).all(dim=2)

        base_numeric = torch.zeros_like(expanded_num_inputs)

        num_embeddings = (
            numeric_col_embeddings[~inf_mask] * expanded_num_inputs[~inf_mask]
        )
        base_numeric[~inf_mask] = num_embeddings
        base_numeric[inf_mask] = self.embeddings(self.numeric_mask_token)

        query_embeddings = torch.cat([cat_embeddings, base_numeric], dim=1)
        out = self.transformer_encoder1(
            col_embeddings,
            # query_embeddings,
            query_embeddings,
            query_embeddings
            # col_embeddings, query_embeddings, query_embeddings
        )
        out = self.transformer_encoder2(out, out, out)

        if task == "regression":
            out = self.regressor(out)
            out = self.flatten_layer(out.squeeze(-1))

            return out
        elif task == "mlm":
            cat_out = self.mlm_decoder(out)
            # print(f"Out shape: {out.shape}, cat_out shape: {cat_out.shape}")
            numeric_out = out.view(out.size(0), -1)
            # print(f"numeric_out shape: {numeric_out.shape}")
            numeric_out = self.mnm_decoder(numeric_out)
            return cat_out, numeric_out
        else:
            raise ValueError(f"Task {task} not supported.")


# %%
def show_mask_pred(i, model, dataset, probability):
    numeric_values = dataset.X_train_numeric[i : i + 1, :]
    categorical_values = dataset.X_train_categorical[i : i + 1, :]
    numeric_masked = mask_tensor(numeric_values, model, probability=probability)
    categorical_masked = mask_tensor(categorical_values, model, probability=probability)
    # Predictions
    with torch.no_grad():
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
