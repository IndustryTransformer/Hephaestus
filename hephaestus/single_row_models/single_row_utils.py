import copy
from datetime import datetime as dt

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange


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
    model,
    dataset,
    n_rows,
    model_name,
    epochs=100,
    lr=0.01,
    early_stop=True,
    return_model=False,
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
    if return_model:
        return model, {"n_rows": n_rows, "test_loss": best_loss.item(), "model": model}
    else:
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
