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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hephaestus.single_row_models as sr
from hephaestus.single_row_models.plotting_utils import plot_prediction_analysis

os.environ["TOKENIZERS_PARALLELISM"] = "false"


Markdown("""## Hephaestus Parameters
We will use the following parameters for the Hephaestus model:
""")

D_MODEL = 64
N_HEADS = 4
LR = 0.0001
BATCH_SIZE = 1024 * 8
name = "Stock_Skip_with_reg"
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

df = df.rename(columns={"nox": "target"})
# scale the non-target numerical columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns
# numeric_cols = numeric_cols.drop("target")
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
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
logger_name = f"{logger_time}_{LOGGER_VARIANT_NAME}"
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
# %%

# Markdown("""### Finetune the model

# We are having issues on the highest values of the target. We will finetune the model
# on the highest values of the target.""")
# fine_tune_model = sr.TabularRegressor(
#     single_row_config, d_model=D_MODEL, n_heads=N_HEADS, lr=0.00001
# )
# fine_tune_model.load_state_dict(model.state_dict())
# train_df_finetune_high = df.loc[df.target > 4]
# train_df_finetune_high = pd.concat(
#     [
#         train_df_finetune_high,
#         train_df.loc[df.target < 4].sample(len(train_df_finetune_high)),
#     ]
# )
# train_df_finetune_high = train_df_finetune_high.sample(frac=1).reset_index(drop=True)
# train_ds_finetune_high = sr.TabularDS(train_df_finetune_high, single_row_config)
# high_dataloader = torch.utils.data.DataLoader(
#     train_ds_finetune_high,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     collate_fn=sr.training.tabular_collate_fn,
# )
# finetune_logger = TensorBoardLogger(
#     "runs",
#     name=f"{logger_time}_{LOGGER_VARIANT_NAME}_finetune",
# )
# finetune_trainer = L.Trainer(
#     max_epochs=30,
#     logger=finetune_logger,
#     callbacks=[progress_bar, model_summary],
#     log_every_n_steps=1,
# )
# finetune_trainer.fit(
#     fine_tune_model,
#     train_dataloaders=high_dataloader,
#     val_dataloaders=val_dataloader,
# )

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
print(f"Mean Squared Error: {mean_squared_error(y, y_hat)}")

res_df = pd.DataFrame({"Actual": y, "Predicted": y_hat})


Markdown("""### Plot the results""")

plot_prediction_analysis(
    df=res_df,
    name="Hephaestus Finetune",
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
