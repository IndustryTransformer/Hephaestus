# ruff: noq: F402
# %% tags=["hide-input", "hide-output"]
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
import altair as alt


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

D_MODEL = 128
N_HEADS = 4
LR = 0.0001
BATCH_SIZE = 64  # Smaller batch sizes lead to better predictions because outliers are
# better trained on.
name = "MTM_Test"
LOGGER_VARIANT_NAME = f"{name}_D{D_MODEL}_H{N_HEADS}_LR{LR}"
LABEL_RATIO = 1.0

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
df["cat_column"] = "category"  # Dummy category for bugs in data loader
df.head()

# %%
Markdown("""### Model Initialization

Initialize the model and create dataloaders for training and validation.
""")
# %%
X_setup = df[df.columns.drop("target")]
# y = df["target"]

model_config_mtm = sr.SingleRowConfig.generate(X_setup)  # Full dataset - target
model_config_reg = sr.SingleRowConfig.generate(
    df, target="target"
)  # Full dataset with target
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

X_train_sub_set = df_train.drop(columns=["target"])
y_train_sub_set = df_train["target"]
X_test = df_test.drop(columns=["target"])
y_test = df_test["target"]
train_dataset = sr.TabularDS(X_train_sub_set, model_config_mtm)
test_dataset = sr.TabularDS(X_test, model_config_mtm)


mtm_model = sr.MaskedTabularModeling(
    model_config_mtm, d_model=D_MODEL, n_heads=N_HEADS, lr=LR
)
mtm_model.predict_step(train_dataset[0:10].inputs)

# %%
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=sr.training.masked_tabular_collate_fn,
)
val_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=sr.training.masked_tabular_collate_fn,
)


# %%
Markdown("""### Model Training
Using PyTorch Lightning, we will train the model using the training and validation
""")

retrain_model = False
pretrained_model_dir = Path("checkpoints/turbine_limited")
pre_trained_models = list(pretrained_model_dir.glob("*.ckpt"))
# Check if a model with the exact name exists or if retraining is forced
if retrain_model or not any(LOGGER_VARIANT_NAME in p.stem for p in pre_trained_models):
    print("Retraining model or specified model not found.")
    run_trainer = True
else:
    print("Attempting to load pre-trained model.")
    run_trainer = False


if run_trainer:
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
    trainer.fit(
        mtm_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.save_checkpoint(
        os.path.join(
            "checkpoints",
            "turbine_limited",
            f"{LOGGER_VARIANT_NAME}_{trainer.logger.name}.ckpt",
        )
    )

else:  # Find the checkpoint file matching the LOGGER_VARIANT_NAME prefix
    # Ensure the directory exists before searching
    pretrained_model_dir.mkdir(parents=True, exist_ok=True)

    found_checkpoints = list(pretrained_model_dir.glob(f"{LOGGER_VARIANT_NAME}*.ckpt"))

    if not found_checkpoints:
        # Handle the case where no matching checkpoint is found
        print(
            f"📭 No checkpoint found starting with {LOGGER_VARIANT_NAME} in {pretrained_model_dir}. Training model instead."
        )
        run_trainer = True  # Set to train if checkpoint not found
    elif len(found_checkpoints) > 1:
        # Handle ambiguity if multiple checkpoints match (e.g., load the latest)
        # For now, let's load the first one found as an example
        print(
            f"‼️ Warning: Found multiple checkpoints for {LOGGER_VARIANT_NAME}. Loading the first one: {found_checkpoints[0]}"
        )
        checkpoint_path = found_checkpoints[0]
        print(f"Loading checkpoint: {checkpoint_path}")
        mtm_model = sr.MaskedTabularModeling.load_from_checkpoint(
            checkpoint_path,
            model_config=model_config_mtm,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            lr=LR,
        )

    else:
        # Exactly one checkpoint found
        checkpoint_path = found_checkpoints[0]
        print(f"Loading checkpoint: {checkpoint_path}")
        mtm_model = sr.MaskedTabularModeling.load_from_checkpoint(checkpoint_path)

# %%

Markdown("### Init Regressor")
regressor = sr.TabularRegressor(
    model_config=model_config_reg, d_model=D_MODEL, n_heads=N_HEADS, lr=LR
)
regressor.model.tabular_encoder = mtm_model.model.tabular_encoder

# %%
reg_out = regressor.predict_step(train_dataset[0:10])
print(f"{reg_out=}")
# %%
Markdown("### Fine tuning the regressor")

train_df_sub_set = df_train.sample(frac=LABEL_RATIO, random_state=42)
X_train_sub_set = train_df_sub_set.drop(columns=["target"])
y_train_sub_set = train_df_sub_set["target"]

regressor_ds_subset = sr.TabularDS(train_df_sub_set, model_config_reg)
regressor_ds_val_full = sr.TabularDS(df_test, model_config_reg)

train_dataloader_10 = torch.utils.data.DataLoader(
    regressor_ds_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=sr.training.tabular_collate_fn,
)
test_data_loader_full = torch.utils.data.DataLoader(
    regressor_ds_val_full,
    batch_size=BATCH_SIZE,
    collate_fn=sr.training.tabular_collate_fn,
)

logger_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
logger_name = f"{logger_time}_Regressor_fine_tune_{LOGGER_VARIANT_NAME}"
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

trainer.fit(
    regressor,
    train_dataloaders=train_dataloader_10,
    val_dataloaders=test_data_loader_full,
)


# %%

Markdown("""### Run inference on the entire inference dataloader""")


# %%
y = []
y_hat = []
for batch in test_data_loader_full:
    y.append(batch.target)
    preds = regressor.predict_step(batch)
    y_hat.append(preds)

y = torch.cat(y, dim=0).squeeze().numpy()
y_hat = torch.cat(y_hat, dim=0).squeeze().numpy()
print(y.shape, y_hat.shape)
print(f"Mean Squared Error: {mean_squared_error(y, y_hat)}")

res_df = pd.DataFrame({"Actual": y, "Predicted": y_hat})

mse_hep = mean_squared_error(y, y_hat)
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
# if "target" in numeric_cols:
#     X = df[numeric_cols.drop("target")]
# else:
#     X = df[numeric_cols]
# y = df["target"]
# X_train_sub_set, X_test, y_train_sub_set, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# train_len = len(X_train_sub_set)
# train_len_10 = int(train_len * 0.1)
# X_train_sub_set = X_train_sub_set.iloc[:train_len_10]
# y_train_sub_set = y_train_sub_set.iloc[:train_len_10]
X_train_sub_set_skl = X_train_sub_set.select_dtypes(include=[np.number])
X_test_skl = X_test.select_dtypes(include=[np.number])
linear_model = LinearRegression()

linear_model.fit(X_train_sub_set_skl, y_train_sub_set)
y_pred_lr = linear_model.predict(X_test_skl)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
Markdown(f"Linear Regression MSE: {mse_lr:.3f} | RMSE: {rmse_lr:.3f}")
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
model_rf = model_rf.fit(X_train_sub_set_skl, y_train_sub_set)

# %%

y_pred_rf = model_rf.predict(X_test_skl)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
Markdown(f"Random Forest MSE: {mse_rf:.3f}, RMSE: {rmse_rf:.3f}")

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


# %%
# Prepare data for plotting
mse_data = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Hephaestus",
        ],
        "MSE": [
            mse_lr,
            mse_rf,
            mse_hep,
        ],
    }
)

# Create the Altair bar chart
mse_chart = (
    alt.Chart(mse_data)
    .mark_bar()
    .encode(
        x=alt.X("Model", title="Model", sort=mse_data["Model"].tolist()),
        y=alt.Y("MSE", title="Mean Squared Error"),
        color="Model",
    )
    .properties(
        title=f"MSE Comparison Across Models with only {(LABEL_RATIO*100)}% of data labled"
    )
)

mse_chart.show()
# %%
