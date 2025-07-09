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
from pathlib import Path

import altair as alt

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
from xgboost import XGBRegressor

import hephaestus.single_row_models as sr
from hephaestus.single_row_models.plotting_utils import plot_prediction_analysis

os.environ["TOKENIZERS_PARALLELISM"] = "false"


Markdown("""## Hephaestus Parameters
We will use the following parameters for the Hephaestus model:
""")

D_MODEL = 128
N_HEADS = 4
LR = 0.0001 * 8
BATCH_SIZE = (
    64 * 8
)  # Smaller batch sizes lead to better predictions because outliers are
# better trained on.

name = "TryLinearLayer"
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

# Remove CO column if it exists
if "co" in df.columns:
    df = df.drop(columns=["co"])
    print("Removed CO column from dataset")

df = df.rename(columns={"nox": "target"})
# Store original target values for unscaled metrics
original_target = df["target"].copy()

# scale the non-target numerical columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns
numeric_cols_without_target = numeric_cols.drop("target")
# Scale features but keep target separate for unscaled metrics
df[numeric_cols_without_target] = scaler.fit_transform(df[numeric_cols_without_target])

# Scale target separately and store the scaler
target_scaler = StandardScaler()
df["target"] = target_scaler.fit_transform(df[["target"]]).flatten()

# Add categorical column AFTER scaling to avoid type issues
df["cat_column"] = "category"  # Dummy category for bugs in data loader
# Ensure categorical column is properly typed
df["cat_column"] = df["cat_column"].astype("category")
print(f"Data types after processing: {df.dtypes}")
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
# print(f"Number of columns in model config: {model_config_mtm.n_columns}")
# print(f"Number of features after CO removal: {len(X_setup.columns)}")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

X_train_sub_set = df_train.drop(columns=["target"])
y_train_sub_set = df_train["target"]
X_test = df_test.drop(columns=["target"])
y_test = df_test["target"]
train_dataset = sr.TabularDS(X_train_sub_set, model_config_mtm)
test_dataset = sr.TabularDS(X_test, model_config_mtm)


mtm_model = sr.MaskedTabularModeling(
    model_config_mtm,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    lr=LR,
)
# mtm_model.predict_step(train_dataset[0:10].inputs)  # Skip test prediction before
# training

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

retrain_model = True
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
    logger_name = f"{LOGGER_VARIANT_NAME}"
    print(f"Using logger name: {logger_name}")
    logger = TensorBoardLogger(
        "runs",
        name=logger_name,
    )
    model_summary = ModelSummary(max_depth=3)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, min_delta=0.0001, mode="min"
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
            f"📭 No checkpoint found starting with {LOGGER_VARIANT_NAME} in",
            f"{pretrained_model_dir}. Training model instead.",
        )
        run_trainer = True  # Set to train if checkpoint not found
    elif len(found_checkpoints) > 1:
        # Handle ambiguity if multiple checkpoints match (e.g., load the latest)
        # For now, let's load the first one found as an example
        print(
            f"‼️ Warning: Found multiple checkpoints for {LOGGER_VARIANT_NAME}."
            f"Loading the first one: {found_checkpoints[0]}"
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
    model_config=model_config_reg,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    lr=LR,
)
regressor.model.tabular_encoder = mtm_model.model.tabular_encoder

# %%
# reg_out = regressor.predict_step(train_dataset[0:10])  # Skip test prediction before
# training
# print(f"{reg_out=}")
# %%
Markdown("""## Define Model Training Functions

We will define functions to train each model with different percentages of labeled data.
Each function will return model metrics for comparison.
""")


# %%
def train_hephaestus_model(
    df_train, df_test, model_config_mtm, model_config_reg, label_ratio=1.0
):
    """
    Train the Hephaestus model on a subset of labeled data.

    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        model_config_mtm: Model configuration for masked tabular modeling
        model_config_reg: Model configuration for regression
        label_ratio: Percentage of labeled data to use for training

    Returns:
        Dictionary containing the trained model and performance metrics
    """
    # Train on subset of data
    train_df_sub_set = df_train.sample(frac=label_ratio, random_state=42)
    # X_train_sub_set = train_df_sub_set.drop(columns=["target"])
    # y_train_sub_set = train_df_sub_set["target"]

    # Create datasets and dataloaders
    regressor_ds_subset = sr.TabularDS(train_df_sub_set, model_config_reg)
    regressor_ds_val_full = sr.TabularDS(df_test, model_config_reg)

    train_dataloader = torch.utils.data.DataLoader(
        regressor_ds_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=sr.training.tabular_collate_fn,
    )
    test_data_loader = torch.utils.data.DataLoader(
        regressor_ds_val_full,
        batch_size=BATCH_SIZE,
        collate_fn=sr.training.tabular_collate_fn,
    )

    # Initialize regressor with pre-trained encoder
    regressor = sr.TabularRegressor(
        model_config=model_config_reg,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        lr=LR,
    )
    regressor.model.tabular_encoder = mtm_model.model.tabular_encoder

    # Training configuration
    logger_name = f"Regressor_fine_tune_{LOGGER_VARIANT_NAME}_{label_ratio}"
    print(f"Using logger name: {logger_name}")
    logger = TensorBoardLogger("runs", name=logger_name)

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

    # Train the model
    trainer.fit(
        regressor,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_data_loader,
    )

    # Evaluate on test set
    y = []
    y_hat = []
    for batch in test_data_loader:
        y.append(batch.target)
        preds = regressor.predict_step(batch)
        y_hat.append(preds)

    y = torch.cat(y, dim=0).squeeze().numpy()
    y_hat = torch.cat(y_hat, dim=0).squeeze().numpy()

    # Convert scaled predictions back to original scale for metrics
    y_unscaled = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_hat_unscaled = target_scaler.inverse_transform(y_hat.reshape(-1, 1)).flatten()

    # Calculate metrics on unscaled values
    mse = mean_squared_error(y_unscaled, y_hat_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_unscaled, y_hat_unscaled)

    # Return model and metrics
    return {
        "model": regressor,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_unscaled,
        "y_pred": y_hat_unscaled,
        "label_ratio": label_ratio,
    }


# %%
def train_hephaestus_no_pretrain(df_train, df_test, model_config_reg, label_ratio=1.0):
    """
    Train a Hephaestus model from scratch without pre-trained weights.

    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        model_config_reg: Model configuration for regression
        label_ratio: Percentage of labeled data to use for training

    Returns:
        Dictionary containing the trained model and performance metrics
    """
    # Train on subset of data
    train_df_sub_set = df_train.sample(frac=label_ratio, random_state=42)

    # Create datasets and dataloaders
    regressor_ds_subset = sr.TabularDS(train_df_sub_set, model_config_reg)
    regressor_ds_val_full = sr.TabularDS(df_test, model_config_reg)

    train_dataloader = torch.utils.data.DataLoader(
        regressor_ds_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=sr.training.tabular_collate_fn,
    )
    test_data_loader = torch.utils.data.DataLoader(
        regressor_ds_val_full,
        batch_size=BATCH_SIZE,
        collate_fn=sr.training.tabular_collate_fn,
    )

    # Initialize regressor WITHOUT pre-trained encoder
    regressor = sr.TabularRegressor(
        model_config=model_config_reg,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        lr=LR,
    )
    # Note: NOT copying pre-trained weights - training from scratch

    # Training configuration
    logger_name = f"Regressor_no_pretrain_{LOGGER_VARIANT_NAME}_{label_ratio}"
    print(f"Using logger name: {logger_name}")
    logger = TensorBoardLogger("runs", name=logger_name)

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

    # Train the model
    trainer.fit(
        regressor,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_data_loader,
    )

    # Evaluate on test set
    y = []
    y_hat = []
    for batch in test_data_loader:
        y.append(batch.target)
        preds = regressor.predict_step(batch)
        y_hat.append(preds)

    y = torch.cat(y, dim=0).squeeze().numpy()
    y_hat = torch.cat(y_hat, dim=0).squeeze().numpy()

    # Convert scaled predictions back to original scale for metrics
    y_unscaled = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_hat_unscaled = target_scaler.inverse_transform(y_hat.reshape(-1, 1)).flatten()

    # Calculate metrics on unscaled values
    mse = mean_squared_error(y_unscaled, y_hat_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_unscaled, y_hat_unscaled)

    # Return model and metrics
    return {
        "model": regressor,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_unscaled,
        "y_pred": y_hat_unscaled,
        "label_ratio": label_ratio,
    }


# %%
def train_linear_regression(df_train, df_test, label_ratio=1.0):
    """
    Train a Linear Regression model on a subset of labeled data.

    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        label_ratio: Percentage of labeled data to use for training

    Returns:
        Dictionary containing the trained model and performance metrics
    """
    # Train on subset of data
    train_df_sub_set = df_train.sample(frac=label_ratio, random_state=42)
    X_train_sub_set = train_df_sub_set.drop(columns=["target"])
    y_train_sub_set = train_df_sub_set["target"]

    # Use only numeric columns for sklearn models
    X_train_sub_set_skl = X_train_sub_set.select_dtypes(include=[np.number])
    X_test_skl = df_test.drop(columns=["target"]).select_dtypes(include=[np.number])
    y_test = df_test["target"]

    # Train the model
    linear_model = LinearRegression()
    linear_model.fit(X_train_sub_set_skl, y_train_sub_set)

    # Evaluate
    y_pred = linear_model.predict(X_test_skl)

    # Convert scaled predictions back to original scale for metrics
    y_test_unscaled = target_scaler.inverse_transform(
        y_test.values.reshape(-1, 1)
    ).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate metrics on unscaled values
    mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)

    # Return model and metrics
    return {
        "model": linear_model,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_test_unscaled,
        "y_pred": y_pred_unscaled,
        "label_ratio": label_ratio,
    }


# %%
def train_random_forest(df_train, df_test, label_ratio=1.0):
    """
    Train a Random Forest model on a subset of labeled data.

    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        label_ratio: Percentage of labeled data to use for training

    Returns:
        Dictionary containing the trained model and performance metrics
    """
    # Train on subset of data
    train_df_sub_set = df_train.sample(frac=label_ratio, random_state=42)
    X_train_sub_set = train_df_sub_set.drop(columns=["target"])
    y_train_sub_set = train_df_sub_set["target"]

    # Use only numeric columns for sklearn models
    X_train_sub_set_skl = X_train_sub_set.select_dtypes(include=[np.number])
    X_test_skl = df_test.drop(columns=["target"]).select_dtypes(include=[np.number])
    y_test = df_test["target"]

    # Train the model
    rf_model = RandomForestRegressor(random_state=0)
    rf_model.fit(X_train_sub_set_skl, y_train_sub_set)

    # Evaluate
    y_pred = rf_model.predict(X_test_skl)

    # Convert scaled predictions back to original scale for metrics
    y_test_unscaled = target_scaler.inverse_transform(
        y_test.values.reshape(-1, 1)
    ).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate metrics on unscaled values
    mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)

    # Return model and metrics
    return {
        "model": rf_model,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_test_unscaled,
        "y_pred": y_pred_unscaled,
        "label_ratio": label_ratio,
    }


# %%
def train_xgboost(df_train, df_test, label_ratio=1.0):
    """
    Train an XGBoost model on a subset of labeled data.

    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        label_ratio: Percentage of labeled data to use for training

    Returns:
        Dictionary containing the trained model and performance metrics
    """
    # Train on subset of data
    train_df_sub_set = df_train.sample(frac=label_ratio, random_state=42)
    X_train_sub_set = train_df_sub_set.drop(columns=["target"])
    y_train_sub_set = train_df_sub_set["target"]

    # Use only numeric columns for sklearn models
    X_train_sub_set_skl = X_train_sub_set.select_dtypes(include=[np.number])
    X_test_skl = df_test.drop(columns=["target"]).select_dtypes(include=[np.number])
    y_test = df_test["target"]

    # Train the model
    xgb_model = XGBRegressor(random_state=0, verbosity=0)
    xgb_model.fit(X_train_sub_set_skl, y_train_sub_set)

    # Evaluate
    y_pred = xgb_model.predict(X_test_skl)

    # Convert scaled predictions back to original scale for metrics
    y_test_unscaled = target_scaler.inverse_transform(
        y_test.values.reshape(-1, 1)
    ).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate metrics on unscaled values
    mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)

    # Return model and metrics
    return {
        "model": xgb_model,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_test_unscaled,
        "y_pred": y_pred_unscaled,
        "label_ratio": label_ratio,
    }


# %%
Markdown("""## Evaluate Models on Different Data Fractions

We'll train and evaluate each model on different fractions of labeled data.
""")

# %%
# Define the data fractions to test
data_fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

# Lists to store results
hephaestus_results = []
hephaestus_no_pretrain_results = []
lr_results = []
rf_results = []
xgb_results = []

# Train and evaluate models on each data fraction
for fraction in data_fractions:
    print(f"\nTraining with {fraction * 100}% of labeled data:")

    # Train Hephaestus model (fine-tuned)
    print("Training Hephaestus model (fine-tuned)...")
    hep_result = train_hephaestus_model(
        df_train, df_test, model_config_mtm, model_config_reg, label_ratio=fraction
    )
    hephaestus_results.append(hep_result)

    # Train Hephaestus model (no pre-training)
    print("Training Hephaestus model (no pre-training)...")
    hep_no_pretrain_result = train_hephaestus_no_pretrain(
        df_train, df_test, model_config_reg, label_ratio=fraction
    )
    hephaestus_no_pretrain_results.append(hep_no_pretrain_result)

    # Train Linear Regression model
    print("Training Linear Regression model...")
    lr_result = train_linear_regression(df_train, df_test, label_ratio=fraction)
    lr_results.append(lr_result)

    # Train Random Forest model
    print("Training Random Forest model...")
    rf_result = train_random_forest(df_train, df_test, label_ratio=fraction)
    rf_results.append(rf_result)

    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_result = train_xgboost(df_train, df_test, label_ratio=fraction)
    xgb_results.append(xgb_result)

    print(f"Results with {fraction * 100}% of labeled data:")
    print(
        f"  Hephaestus (fine-tuned) MSE: {hep_result['mse']:.3f}, "
        f"RMSE: {hep_result['rmse']:.3f}, MAE: {hep_result['mae']:.3f}"
    )
    print(
        f"  Hephaestus (no pre-training) MSE: {hep_no_pretrain_result['mse']:.3f}, "
        f"RMSE: {hep_no_pretrain_result['rmse']:.3f}, "
        f"MAE: {hep_no_pretrain_result['mae']:.3f}"
    )
    print(
        f"  Linear Regression MSE: {lr_result['mse']:.3f}, "
        f"RMSE: {lr_result['rmse']:.3f}, MAE: {lr_result['mae']:.3f}"
    )
    print(
        f"  Random Forest MSE: {rf_result['mse']:.3f}, "
        f"RMSE: {rf_result['rmse']:.3f}, MAE: {rf_result['mae']:.3f}"
    )
    print(
        f"  XGBoost MSE: {xgb_result['mse']:.3f}, "
        f"RMSE: {xgb_result['rmse']:.3f}, MAE: {xgb_result['mae']:.3f}"
    )

# %%
Markdown("""## Visualize the Results

Create dataframes and visualizations to show model performance across different data
fractions.
""")

# %%
# Combine results into dataframes
results_rows = []

# Add Hephaestus results
for result in hephaestus_results:
    results_rows.append(
        {
            "Model": "Hephaestus (Fine-tuned)",
            "Label Ratio": result["label_ratio"],
            "MSE": result["mse"],
            "RMSE": result["rmse"],
            "MAE": result["mae"],
        }
    )

# Add Hephaestus no pre-training results
for result in hephaestus_no_pretrain_results:
    results_rows.append(
        {
            "Model": "Hephaestus (No Pre-training)",
            "Label Ratio": result["label_ratio"],
            "MSE": result["mse"],
            "RMSE": result["rmse"],
            "MAE": result["mae"],
        }
    )

# Add Linear Regression results
for result in lr_results:
    results_rows.append(
        {
            "Model": "Linear Regression",
            "Label Ratio": result["label_ratio"],
            "MSE": result["mse"],
            "RMSE": result["rmse"],
            "MAE": result["mae"],
        }
    )

# Add Random Forest results
for result in rf_results:
    results_rows.append(
        {
            "Model": "Random Forest",
            "Label Ratio": result["label_ratio"],
            "MSE": result["mse"],
            "RMSE": result["rmse"],
            "MAE": result["mae"],
        }
    )

# Add XGBoost results
for result in xgb_results:
    results_rows.append(
        {
            "Model": "XGBoost",
            "Label Ratio": result["label_ratio"],
            "MSE": result["mse"],
            "RMSE": result["rmse"],
            "MAE": result["mae"],
        }
    )

# Create dataframe
results_df = pd.DataFrame(results_rows)

# Print the results dataframe
print(results_df)

# %%
# Plot MSE vs. Label Ratio
mse_chart = (
    alt.Chart(results_df)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "Label Ratio:Q",
            title="Fraction of Labeled Data",
            scale=alt.Scale(type="log"),
        ),
        y=alt.Y("MSE:Q", title="Mean Squared Error"),
        color=alt.Color("Model:N"),
        tooltip=["Model", "Label Ratio", "MSE", "RMSE", "MAE"],
    )
    .properties(
        title="MSE Comparison Across Models with Different Amounts of Labeled Data",
        width=600,
        height=400,
    )
)

# Add a baseline rule at the minimum MSE value for each model
min_mse_by_model = results_df.groupby("Model")["MSE"].min().reset_index()

baseline_rules = (
    alt.Chart(min_mse_by_model)
    .mark_rule(strokeDash=[2, 2])
    .encode(y="MSE:Q", color=alt.Color("Model:N"), tooltip=["Model", "MSE"])
)

combined_chart = mse_chart + baseline_rules
# Save as PNG instead of HTML
combined_chart.save("result_images/model_performance_comparison.png")
combined_chart.show()

# %%
# Plot a bar chart showing MSE for specific label ratios (e.g., 0.1, 0.5, 1.0)
selected_ratios = [0.1, 0.5, 1.0]
filtered_df = results_df[results_df["Label Ratio"].isin(selected_ratios)]

# Calculate the min MSE for each label ratio to highlight the best model
best_models = filtered_df.loc[filtered_df.groupby("Label Ratio")["MSE"].idxmin()]
best_models["is_best"] = True
filtered_df = pd.merge(
    filtered_df,
    best_models[["Model", "Label Ratio", "is_best"]],
    on=["Model", "Label Ratio"],
    how="left",
)
filtered_df["is_best"] = filtered_df["is_best"].fillna(False)

# Create the bar chart with highlight for best model
# Create a base chart first, then apply faceting
base = alt.Chart(filtered_df).properties(
    title="MSE Comparison at Different Label Ratios", width=200
)

# Create the bar chart
bar_chart = base.mark_bar().encode(
    x=alt.X("Model:N", title="Model"),
    y=alt.Y("MSE:Q", title="Mean Squared Error"),
    color=alt.Color("Model:N"),
    tooltip=["Model", "Label Ratio", "MSE", "RMSE", "MAE"],
    opacity=alt.condition(alt.datum.is_best, alt.value(1), alt.value(0.6)),
    strokeWidth=alt.condition(alt.datum.is_best, alt.value(2), alt.value(0)),
    stroke=alt.condition(alt.datum.is_best, alt.value("black"), alt.value(None)),
)

# Add text labels for the best model
best_model_text = base.mark_text(
    align="center", baseline="top", dy=5, fontSize=10
).encode(
    x=alt.X("Model:N"),
    y=alt.Y("MSE:Q"),
    text=alt.Text("Model:N"),
    opacity=alt.condition(alt.datum.is_best, alt.value(1), alt.value(0)),
)

# Layer the visualizations before faceting
combined_layer = alt.layer(bar_chart, best_model_text)

# Apply faceting after layering
combined_bar_chart = combined_layer.facet(
    column=alt.Column("Label Ratio:N", title="Fraction of Labeled Data")
)

combined_bar_chart.save("result_images/model_performance_bars.png")
combined_bar_chart.show()

# %%
# Plot individual model visualizations for the 10% labeled data case
ratio_10_percent = 0.1
hep_result_10 = next(
    r for r in hephaestus_results if r["label_ratio"] == ratio_10_percent
)
hep_no_pretrain_result_10 = next(
    r for r in hephaestus_no_pretrain_results if r["label_ratio"] == ratio_10_percent
)
lr_result_10 = next(r for r in lr_results if r["label_ratio"] == ratio_10_percent)
rf_result_10 = next(r for r in rf_results if r["label_ratio"] == ratio_10_percent)
xgb_result_10 = next(r for r in xgb_results if r["label_ratio"] == ratio_10_percent)

# Create result dataframes
res_df = pd.DataFrame(
    {"Actual": hep_result_10["y_true"], "Predicted": hep_result_10["y_pred"]}
)
res_df_no_pretrain = pd.DataFrame(
    {
        "Actual": hep_no_pretrain_result_10["y_true"],
        "Predicted": hep_no_pretrain_result_10["y_pred"],
    }
)
res_df_lr = pd.DataFrame(
    {"Actual": lr_result_10["y_true"], "Predicted": lr_result_10["y_pred"]}
)
res_df_rf = pd.DataFrame(
    {"Actual": rf_result_10["y_true"], "Predicted": rf_result_10["y_pred"]}
)
res_df_xgb = pd.DataFrame(
    {"Actual": xgb_result_10["y_true"], "Predicted": xgb_result_10["y_pred"]}
)

# %% Plot individual model results
plot_prediction_analysis(
    df=res_df,
    name="Hephaestus Fine-tuned (10% data)",
    y_col="Actual",
    y_hat_col="Predicted",
)
# %%
plot_prediction_analysis(
    df=res_df_no_pretrain,
    name="Hephaestus No Pre-training (10% data)",
    y_col="Actual",
    y_hat_col="Predicted",
)
# %%
plot_prediction_analysis(
    df=res_df_lr,
    name="Linear Regression (10% data)",
    y_col="Actual",
    y_hat_col="Predicted",
    it_color="scikit",
)
# %%
plot_prediction_analysis(
    df=res_df_rf,
    name="Random Forest (10% data)",
    y_col="Actual",
    y_hat_col="Predicted",
    it_color="scikit",
)
# %%
plot_prediction_analysis(
    df=res_df_xgb,
    name="XGBoost (10% data)",
    y_col="Actual",
    y_hat_col="Predicted",
    it_color="scikit",
)

# %%
# Create a combined bar chart for the 10% case
mse_data_10 = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost",
            "Hephaestus (Fine-tuned)",
            "Hephaestus (No Pre-training)",
        ],
        "MSE": [
            lr_result_10["mse"],
            rf_result_10["mse"],
            xgb_result_10["mse"],
            hep_result_10["mse"],
            hep_no_pretrain_result_10["mse"],
        ],
    }
)

mse_chart_10 = (
    alt.Chart(mse_data_10)
    .mark_bar()
    .encode(
        x=alt.X("Model", title="Model", sort=mse_data_10["Model"].tolist()),
        y=alt.Y("MSE", title="Mean Squared Error"),
        color="Model",
    )
    .properties(
        title=f"MSE Comparison Across Models with only "
        f"{(ratio_10_percent * 100)}% of data labeled"
    )
)

mse_chart_10.save("result_images/model_performance_10.png")
mse_chart_10.show()

# %%
Markdown("""## Performance Improvement Analysis

Calculate how much better Hephaestus performs compared to traditional models at each 
data fraction.
""")

# Calculate performance improvement percentages
improvement_rows = []

for fraction in data_fractions:
    # Get results for this fraction
    hep_result = next(r for r in hephaestus_results if r["label_ratio"] == fraction)
    hep_no_pretrain_result = next(
        r for r in hephaestus_no_pretrain_results if r["label_ratio"] == fraction
    )
    lr_result = next(r for r in lr_results if r["label_ratio"] == fraction)
    rf_result = next(r for r in rf_results if r["label_ratio"] == fraction)
    xgb_result = next(r for r in xgb_results if r["label_ratio"] == fraction)

    # Calculate best traditional model MSE at this fraction
    trad_mses = [lr_result["mse"], rf_result["mse"], xgb_result["mse"]]
    trad_models = ["Linear Regression", "Random Forest", "XGBoost"]
    best_trad_idx = trad_mses.index(min(trad_mses))
    trad_mse = trad_mses[best_trad_idx]
    best_trad_model = trad_models[best_trad_idx]

    # Calculate improvement percentages for both Hephaestus models
    improvement_pct_finetuned = ((trad_mse - hep_result["mse"]) / trad_mse) * 100
    improvement_pct_no_pretrain = (
        (trad_mse - hep_no_pretrain_result["mse"]) / trad_mse
    ) * 100

    improvement_rows.append(
        {
            "Label Ratio": fraction,
            "Hephaestus (Fine-tuned) MSE": hep_result["mse"],
            "Hephaestus (No Pre-training) MSE": hep_no_pretrain_result["mse"],
            "Best Traditional Model": best_trad_model,
            "Best Traditional MSE": trad_mse,
            "Fine-tuned Improvement (%)": improvement_pct_finetuned,
            "No Pre-training Improvement (%)": improvement_pct_no_pretrain,
        }
    )

improvement_df = pd.DataFrame(improvement_rows)
print(improvement_df)

# %%
# Create a long-form dataframe for plotting both improvement lines
improvement_long = []
for _, row in improvement_df.iterrows():
    improvement_long.extend(
        [
            {
                "Label Ratio": row["Label Ratio"],
                "Improvement (%)": row["Fine-tuned Improvement (%)"],
                "Model Type": "Hephaestus (Fine-tuned)",
                "Best Traditional Model": row["Best Traditional Model"],
            },
            {
                "Label Ratio": row["Label Ratio"],
                "Improvement (%)": row["No Pre-training Improvement (%)"],
                "Model Type": "Hephaestus (No Pre-training)",
                "Best Traditional Model": row["Best Traditional Model"],
            },
        ]
    )

improvement_long_df = pd.DataFrame(improvement_long)

# Plot improvement percentage vs data fraction
improvement_chart = (
    alt.Chart(improvement_long_df)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "Label Ratio:Q",
            title="Fraction of Labeled Data",
            scale=alt.Scale(type="log"),
        ),
        y=alt.Y("Improvement (%):Q", title="Improvement (%)"),
        color=alt.Color("Model Type:N"),
        tooltip=[
            "Label Ratio",
            "Improvement (%)",
            "Model Type",
            "Best Traditional Model",
        ],
    )
    .properties(
        title="Hephaestus Performance Improvement over Traditional Models",
        width=600,
        height=400,
    )
)

# Add a zero line to show threshold where Hephaestus becomes better
zero_line = (
    alt.Chart(pd.DataFrame({"y": [0]}))
    .mark_rule(strokeDash=[3, 3], color="gray")
    .encode(y="y")
)

improvement_chart_with_line = improvement_chart + zero_line
improvement_chart_with_line.save("result_images/performance_improvement.png")
improvement_chart_with_line.show()
# %%
