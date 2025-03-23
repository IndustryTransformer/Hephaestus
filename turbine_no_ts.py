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

import altair as alt
import numpy as np
import pandas as pd
import pytorch_lightning as L  # noqa: N812
from sklearn.linear_model import LinearRegression

import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hephaestus.single_row_models as sr

D_MODEL = 128
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

# %%
# Run inference on the entire inference dataloader
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
plot_df = res_df.sample(5000)
# %%
# Create a scatter plot to compare predicted vs actual values using Altair

# Calculate metrics
mse = mean_squared_error(y, y_hat)
rmse = np.sqrt(mse)

scatter = (
    alt.Chart(plot_df)
    .mark_circle(opacity=0.5)
    .encode(
        x=alt.X("Actual", title="Actual Nox"),
        y=alt.Y("Predicted", title="Predicted Nox"),
        tooltip=["Actual", "Predicted"],
    )
    .properties(width=600, height=400, title="Actual vs Predicted Emissions")
)

# Create the diagonal line
min_val = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
max_val = max(plot_df["Actual"].max(), plot_df["Predicted"].max())
line_df = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})
line = alt.Chart(line_df).mark_line(color="red", strokeDash=[4, 4]).encode(x="x", y="y")

# Add text annotation for metrics
text_df = pd.DataFrame(
    [
        {
            "x": min_val + (max_val - min_val) * 0.05,
            "y": min_val + (max_val - min_val) * 0.95,
            "text": f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}",
        }
    ]
)
text = (
    alt.Chart(text_df)
    .mark_text(align="left", baseline="top")
    .encode(x="x", y="y", text="text")
)

# Combine the components
chart = (scatter + line + text).interactive()

# Display the chart
chart

# %%
#
# Create a line plot of df.index vs df.target (nox) using Altair
df_head_tail = pd.concat([df.head(1000), df.tail(1000)]).reset_index(drop=True)
line_chart = (
    alt.Chart(df_head_tail.reset_index())
    .mark_line()
    .encode(
        x=alt.X("index", title="Index"),
        y=alt.Y("target", title="Nox"),
        tooltip=["index", "target"],
    )
    .properties(width=800, height=400, title="Nox Emissions Over Index")
)

# Display the chart
line_chart
# %%
# %% [markdown]
# ## Linear Regression

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
plot_df = res_df.sample(5000)
scatter_linear = (
    alt.Chart(plot_df)
    .mark_circle(opacity=0.5)
    .encode(
        x=alt.X("Actual", title="Actual Nox"),
        y=alt.Y("Predicted", title="Predicted Nox"),
        tooltip=["Actual", "Predicted"],
    )
    .properties(
        width=600,
        height=400,
        title="Actual vs Predicted Emissions with Linear Regression",
    )
)
min_val = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
max_val = max(plot_df["Actual"].max(), plot_df["Predicted"].max())
line_df = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})
line = alt.Chart(line_df).mark_line(color="red", strokeDash=[4, 4]).encode(x="x", y="y")

# Add text annotation for metrics
text_df = pd.DataFrame(
    [
        {
            "x": min_val + (max_val - min_val) * 0.05,
            "y": min_val + (max_val - min_val) * 0.95,
            "text": f"MSE: {mse_linear:.2f}\nRMSE: {rmse:.2f}",
        }
    ]
)
text = (
    alt.Chart(text_df)
    .mark_text(align="left", baseline="top")
    .encode(x="x", y="y", text="text")
)

# Combine the components
chart_linear = (scatter_linear + line + text).interactive()

# Display the chart
chart_linear
# %%
