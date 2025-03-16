# %%
from datetime import datetime as dt

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange

import hephaestus.single_row_models as sr
from hephaestus.single_row_models.single_row_utils import EarlyStopping

# Load and preprocess the dataset (assuming you have a CSV file)
df = pd.read_csv("./data/diamonds.csv")
df.head()

# %%
dataset = sr.TabularDS(df, target_column="price")

# %%
print("Numeric Shape:", dataset.X_train_numeric[0:2].shape)
print("Categorical Shape:", dataset.X_train_categorical[0:2].shape)

# %%
model = sr.TabRegressor(dataset, n_heads=8).to(dataset.device)

batch_size = 3
test_num = dataset.X_train_numeric[0:batch_size, :]
test_num_mask = sr.mask_tensor(test_num, model)
test_cat = dataset.X_test_categorical[0:batch_size, :]
test_cat_mask = sr.mask_tensor(test_cat, model)
with torch.no_grad():
    x = model(
        test_num_mask,
        test_cat_mask,
    )
x
# %%
# sr.show_mask_pred(0, model, dataset, probability=0.8)

# %%
# Masked Tabualr Modeling
batch_size = 1000
model_name = "single_diamonds"
loss_fn = nn.MSELoss()
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

early_stopping = EarlyStopping(patience=50, min_delta=0.0001)
summary_writer = SummaryWriter(
    f"runs/{dt.now().strftime("%Y-%m-%dT%H:%M:%S")}_{model_name}"
)


batch_count = 0
model.train()
epochs = 100
pbar = trange(epochs, desc=f"Epochs, Model: {model_name}")
train_set_size = dataset.X_train_numeric.shape[0]
n_rows = train_set_size
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
        summary_writer.add_scalar("LossTrain/regression_loss", loss.item(), batch_count)
        summary_writer.add_scalar("Metrics/regression_lr", learning_rate, batch_count)

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


# %% Show results
model.eval()
mini_batch = 10
with torch.no_grad():
    y_pred = model(
        dataset.X_test_numeric[0:mini_batch, :],
        dataset.X_test_categorical[0:mini_batch, :],
    )
    print("Predictions:", y_pred)
    print("True Values:", dataset.y_test[0:mini_batch, :])

# %%
