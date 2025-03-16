# %%
from datetime import datetime as dt

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange

import hephaestus.single_row_models as sr
from hephaestus.single_row_models.single_row_utils import EarlyStopping

# Load and preprocess the dataset (assuming you have a CSV file)
df = pd.read_csv("./data/diamonds.csv")
df.head()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# %%
dataset = sr.TabularDS(df, target_column="price")
dataset_pt_train = sr.TabularDataset(train_df, dataset)
dataset_pt_test = sr.TabularDataset(test_df, dataset)
train_loader = torch.utils.data.DataLoader(
    dataset_pt_train, batch_size=100, shuffle=True
)
test_loader = torch.utils.data.DataLoader(dataset_pt_test, batch_size=100, shuffle=True)
# %%
model = sr.TabRegressor(dataset, n_heads=8).to(dataset.device)
counter = 0
for i in train_loader:
    test_num = i["X_numeric"]
    test_num_mask = sr.mask_tensor(test_num, model)
    test_cat = i["X_categorical"]
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
# Masked Tabular Modeling
batch_size = 1000
model_name = "single_diamonds_data_loader"
loss_fn = nn.MSELoss()
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

early_stopping = EarlyStopping(patience=50, min_delta=0.0001)
summary_writer = SummaryWriter(
    f"runs/{dt.now().strftime('%Y-%m-%dT%H:%M:%S')}_{model_name}"
)


batch_count = 0
model.train()
epochs = 10
pbar = trange(epochs, desc=f"Epochs, Model: {model_name}")

for epoch in pbar:
    for i in train_loader:
        num_inputs = i["X_numeric"]
        cat_inputs = i["X_categorical"]
        y = i["y"]
        optimizer.zero_grad()
        y_pred = model(num_inputs, cat_inputs)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        batch_count += 1
        learning_rate = optimizer.param_groups[0]["lr"]
        summary_writer.add_scalar("LossTrain/regression_loss", loss.item(), batch_count)
        summary_writer.add_scalar("Metrics/regression_lr", learning_rate, batch_count)

    # Test set
    with torch.no_grad():
        val_loss = 0
        for i in test_loader:
            num_inputs = i["X_numeric"]
            cat_inputs = i["X_categorical"]
            y = i["y"]
            y_pred = model(num_inputs, cat_inputs)
            val_loss += loss_fn(y_pred, y)
        loss_test = val_loss / len(test_loader)
        summary_writer.add_scalar(
            "LossTest/regression_loss", loss_test.item(), batch_count
        )
    if early_stopping is not None:
        early_stopping_status = early_stopping(model, loss_test)
    else:
        early_stopping_status = False
    pbar.set_description(
        f"Epoch {epoch+1}/{epochs} "
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
# Get a sample from the test loader
for batch in test_loader:
    test_numeric = batch["X_numeric"][:mini_batch]
    test_categorical = batch["X_categorical"][:mini_batch]
    test_y = batch["y"][:mini_batch]
    break

with torch.no_grad():
    y_pred = model(test_numeric, test_categorical)
    print("Predictions:", y_pred)
    print("True Values:", test_y)

# %%
