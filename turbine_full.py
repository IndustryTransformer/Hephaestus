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
# bettertrained on.
name = "SmallBatch"
LOGGER_VARIANT_NAME = f"{name}_D{D_MODEL}_H{N_HEADS}_LR{LR}"


# Load and preprocess the train_dataset (assuming you have a CSV file)
train_files = ["gt_2011.csv", "gt_2012.csv", "gt_2013.csv"]
test_files = ["gt_2014.csv", "gt_2015.csv"]

# Read train files
train_df_list = []
for file in train_files:
    file_path = f"data/nox/{file}"
    temp_df = pd.read_csv(file_path)
    filename = Path(file_path).stem  # Get filename without extension
    temp_df["filename"] = filename
    train_df_list.append(temp_df)

# Read test files
test_df_list = []
for file in test_files:
    file_path = f"data/nox/{file}"
    temp_df = pd.read_csv(file_path)
    filename = Path(file_path).stem  # Get filename without extension
    temp_df["filename"] = filename
    test_df_list.append(temp_df)

# Concatenate dataframes
train_df = pd.concat(train_df_list, ignore_index=True)
test_df = pd.concat(test_df_list, ignore_index=True)

# Process both dataframes
for df in [train_df, test_df]:
    df.columns = df.columns.str.lower()
    # Drop CO column if it exists
    if "co" in df.columns:
        df.drop("co", axis=1, inplace=True)
    df.rename(columns={"nox": "target"}, inplace=True)
    df["cat_column"] = "category"

# Store original data before scaling for analysis
train_df_orig = train_df.copy()
test_df_orig = test_df.copy()

# Fit scaler on train data and transform both sets
scaler = StandardScaler()
numeric_cols = train_df.select_dtypes(include=[float, int]).columns

# Important: Don't scale the target variable!
feature_cols = [col for col in numeric_cols if col != "target"]
print(f"\nScaling features: {feature_cols}")

train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# Check data distributions before combining
print("\n=== Data Distribution Analysis ===")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"\nTrain target stats:\n{train_df['target'].describe()}")
print(f"\nTest target stats:\n{test_df['target'].describe()}")

# Check for any significant differences in feature distributions
print("\n=== Feature Distribution Comparison ===")
for col in numeric_cols:
    if col != "target":
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        train_std = train_df[col].std()
        test_std = test_df[col].std()
        mean_diff = abs(train_mean - test_mean)
        print(
            f"{col}: Train μ={train_mean:.3f}±{train_std:.3f}, Test μ={test_mean:.3f}±{test_std:.3f}, Δμ={mean_diff:.3f}"
        )

# Combine for single df (for config generation)
df = pd.concat([train_df, test_df], ignore_index=True)
df.head()

# %%
Markdown("""### Model Initialization

Initialize the model and create dataloaders for training and validation.
""")
# %%
single_row_config = sr.SingleRowConfig.generate(df, "target")
# Use the pre-split train_df and test_df instead of random split

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
# Use feature_cols which excludes target
X_train = train_df[feature_cols]
X_test = test_df[feature_cols]
y_train = train_df["target"]
y_test = test_df["target"]
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
