# ruff: noq: F402
# %%
import numpy as np
from IPython.display import Markdown  # noqa: F401

Markdown("""# Nox Prediction
This notebook is used to predict NOx emissions from a gas turbine using a transformer
model.

## Load Libraries and Prepare Data

""")
# %%
import glob
import os
from datetime import datetime as dt
from pathlib import Path

# ruff: noqa: E402
import pandas as pd
import pytorch_lightning as L  # noqa: N812
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hephaestus.single_row_models as sr
from hephaestus.single_row_models.plotting_utils import plot_prediction_analysis

os.environ["TOKENIZERS_PARALLELISM"] = "false"


Markdown("""## Hephaestus Parameters
We will use the following parameters for the Hephaestus model:
""")

D_MODEL = 128
N_HEADS = 4
LR = 0.0001
BATCH_SIZE = 64  # Smaller batch sizes lead to better predictions because outliers are
# better trained on.
name = "SmallBatch"
LOGGER_VARIANT_NAME = f"{name}_D{D_MODEL}_H{N_HEADS}_LR{LR}"


# Load and preprocess the train_dataset (assuming you have a CSV file)
csv_files = glob.glob("data/nox/*.csv")

# Read and combine all files
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    filename = Path(file).stem  # Get filename without extension
    temp_df["filename"] = filename
    df_list.append(temp_df)

# Concatenate all dataframes
df = pd.concat(df_list, ignore_index=True)
df.columns = df.columns.str.lower()

# Drop CO column if it exists
if "co" in df.columns:
    df = df.drop("co", axis=1)

df = df.rename(columns={"nox": "target"})
# scale the non-target numerical columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns
# Keep target unscaled for loss function
numeric_cols_to_scale = numeric_cols.drop("target")
df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
# df["target"] = df["target"] + 100
df["cat_column"] = "category"
df.head()

# %%
Markdown("""### Model Initialization

Initialize the model and create dataloaders for training and validation.
""")
# %%
single_row_config = sr.SingleRowConfig.generate(df, "target")
train_df, test_df = train_test_split(df.copy(), test_size=0.2, random_state=42)

train_dataset = sr.TabularDS(train_df, single_row_config)
test_dataset = sr.TabularDS(test_df, single_row_config)


model = sr.TabularRegressor(single_row_config, d_model=D_MODEL, n_heads=N_HEADS, lr=LR)
model.predict_step(train_dataset[0:10])

# %%
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=sr.training.tabular_collate_fn,
)
val_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=sr.training.tabular_collate_fn,
)


# %%
Markdown("""### Model Training
Using PyTorch Lightning, we will train the model using the training and validation
""")

logger_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
logger_name = f"{LOGGER_VARIANT_NAME}"
print(f"Using logger name: {logger_name}")
logger = TensorBoardLogger(
    "runs",
    name=logger_name,
)
model_summary = ModelSummary(max_depth=3)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, min_delta=0.001, mode="min"
)
progress_bar = TQDMProgressBar(leave=False)
trainer = L.Trainer(
    max_epochs=200,
    logger=logger,
    callbacks=[early_stopping, progress_bar, model_summary],
    log_every_n_steps=1,
)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

Markdown("""### Run inference on the entire inference dataloader""")
# %%
y = []
y_hat = []
for batch in train_dataloader:
    y.append(batch.target)
    preds = model.predict_step(batch)
    y_hat.append(preds)

y = torch.cat(y, dim=0).squeeze().numpy()
y_hat = torch.cat(y_hat, dim=0).squeeze().numpy()
print(y.shape, y_hat.shape)

# Calculate metrics on unscaled data
mse_train = mean_squared_error(y, y_hat)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y, y_hat)

print(f"Hephaestus Training MSE: {mse_train:.3f}")
print(f"Hephaestus Training RMSE: {rmse_train:.3f}")
print(f"Hephaestus Training MAE: {mae_train:.3f}")

res_df = pd.DataFrame({"Actual": y, "Predicted": y_hat})


Markdown("""### Plot the results""")

plot_prediction_analysis(
    df=res_df,
    name="Hephaestus",
    y_col="Actual",
    y_hat_col="Predicted",
)
# %%
Markdown("""## Compare with Scikit Learn Models
We will compare the Hephaestus model with the following Scikit Learn models:
- Linear Regression
- Random Forest

### Linear Regression
""")
# %%
if "target" in numeric_cols:
    X = df[numeric_cols.drop("target")]
else:
    X = df[numeric_cols]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse_linear)
Markdown(f"Linear Regression MSE: {mse_linear:.3f} | RMSE: {rmse:.3f}")
# %% Plot the results
res_df_lr = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_lr})
plot_prediction_analysis(
    df=res_df_lr,
    name="Linear Regression",
    y_col="Actual",
    y_hat_col="Predicted",
    it_color="scikit",
)
# %%
Markdown("""### Random Forest

Random Forest is a more complex model that can capture non-linear relationships. We will
see if it performs better than the Linear Regression model and the Hephaestus model.
""")
model_rf = RandomForestRegressor()

model_rf = RandomForestRegressor(random_state=0)
model_rf = model_rf.fit(X_train, y_train)

# %%

y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

Markdown(f"Random Forest MSE: {mse_rf:.3f}")

# %% PLot the results
res_df_rf = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_rf})

# %%
plot_prediction_analysis(
    df=res_df_rf,
    name="Random Forest",
    y_col="Actual",
    y_hat_col="Predicted",
    it_color="scikit",
)

# %%
y_val = []
y_hat_val = []
for batch in val_dataloader:
    y_val.append(batch.target)
    preds = model.predict_step(batch)
    y_hat_val.append(preds)

y_val = torch.cat(y_val, dim=0).squeeze().numpy()
y_hat_val = torch.cat(y_hat_val, dim=0).squeeze().numpy()

# Since target is already unscaled, calculate metrics directly
mse_val = mean_squared_error(y_val, y_hat_val)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(y_val, y_hat_val)

print(f"Hephaestus Validation MSE: {mse_val:.3f}")
print(f"Hephaestus Validation RMSE: {rmse_val:.3f}")
print(f"Hephaestus Validation MAE: {mae_val:.3f}")
