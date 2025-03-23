# %% [markdown]
# # Turbine Model
#
# ## Load Libs
#
# %%
import glob
import os
from datetime import datetime as dt
from pathlib import Path

import numpy as np
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
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import hephaestus.single_row_models as sr
from hephaestus.single_row_models.plotting_utils import plot_prediction_analysis

D_MODEL = 64
N_HEADS = 4
LR = 0.0001
BATCH_SIZE = 1024 * 8
name = "SiluTransomed"
LOGGER_VARIANT_NAME = f"{name}_D{D_MODEL}_H{N_HEADS}_LR{LR}"


os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
logger_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
logger_name = f"{logger_time}_{LOGGER_VARIANT_NAME}"
print(f"Using logger name: {logger_name}")
logger = TensorBoardLogger(
    "runs",
    name=logger_name,
)
model_summary = ModelSummary(max_depth=3)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
progress_bar = TQDMProgressBar(leave=False)
trainer = L.Trainer(
    max_epochs=200,
    logger=logger,
    callbacks=[early_stopping, progress_bar, model_summary],
    log_every_n_steps=1,
)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# %% [markdown]
# ### Run inference on the entire inference dataloader
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


# %%
plot_df = res_df.sample(5000)

plot_prediction_analysis(
    df=res_df,
    name="Hephaestus",
    y_col="Actual",
    y_hat_col="Predicted",
)
# %%
# %% [markdown]
# ### Linear Regression

# %%

# prepare X and y
X = df[numeric_cols.drop("target")]
y = df["target"]


linear_model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse_linear)
print(f"Linear Regression MSE: {mse_linear}")
# %% Plot the results
res_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
plot_prediction_analysis(
    df=res_df,
    name="Linear Regression",
    y_col="Actual",
    y_hat_col="Predicted",
)
# %%
model = RandomForestRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(
    model,
    X_train,
    y_train,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    error_score="raise",
)
# report performance
print(
    "MAE on cross validation set : %.3f (%.3f)"
    % (abs(np.mean(n_scores)), np.std(n_scores))
)

# %% Plot the RandomForestRegressor results
# fit the model
model = RandomForestRegressor(random_state=0)
model = model.fit(X_train, y_train)

# %%

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("MSE on test set:", round(mse, 3))

# %% PLot the results
res_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
plot_df = res_df.sample(5000)

# %%
plot_prediction_analysis(
    df=res_df,
    name="Random Forest",
    y_col="Actual",
    y_hat_col="Predicted",
)

# %%
