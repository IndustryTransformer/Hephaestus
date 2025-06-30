# ruff: noqa: F402
# %%
import numpy as np
from IPython.display import Markdown
import os
from datetime import datetime as dt
from pathlib import Path

# ruff: noqa: E402
import pandas as pd
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hephaestus.single_row_models as sr
from hephaestus.single_row_models.plotting_utils import plot_prediction_analysis

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("# Playground Series S4E12 Insurance Premium Prediction")
print("Training Hephaestus model on insurance dataset")

# Model Parameters
D_MODEL = 128
N_HEADS = 4
LR = 0.0001
BATCH_SIZE = 64
name = "PlaygroundS4E12"
LOGGER_VARIANT_NAME = f"{name}_D{D_MODEL}_H{N_HEADS}_LR{LR}"

print(f"Model parameters: D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, LR={LR}, BATCH_SIZE={BATCH_SIZE}")

# Load and preprocess the dataset
print("\n## Loading and preprocessing data")
df = pd.read_csv('/home/ubuntu/Hephaestus/data/playground-series-s4e12/train.csv')
print(f"Dataset shape: {df.shape}")

# Drop id column as it's not useful for prediction
df = df.drop('id', axis=1)

# Rename target column to 'target' to match turbine_full.py pattern
df = df.rename(columns={'Premium Amount': 'target'})

# Handle missing values - fill with median for numeric, mode for categorical
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Missing values before cleaning: {df.isnull().sum().sum()}")

# Fill missing values
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')

print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# Scale ALL numeric columns including target for training
scaler_features = StandardScaler()
scaler_target = StandardScaler()

# Separate features and target
target_col = 'target'
feature_cols = [col for col in numeric_cols if col != target_col]

# Fit scalers
df_scaled = df.copy()
df_scaled[feature_cols] = scaler_features.fit_transform(df[feature_cols])
df_scaled[target_col] = scaler_target.fit_transform(df[[target_col]])

print(f"Target statistics before scaling: mean={df[target_col].mean():.2f}, std={df[target_col].std():.2f}")
print(f"Target statistics after scaling: mean={df_scaled[target_col].mean():.2f}, std={df_scaled[target_col].std():.2f}")

# Add a categorical column for the model (required by Hephaestus)
df_scaled["cat_column"] = "category"

print("\n## Model Initialization")
single_row_config = sr.SingleRowConfig.generate(df_scaled, target_col)
train_df, test_df = train_test_split(df_scaled.copy(), test_size=0.2, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

train_dataset = sr.TabularDS(train_df, single_row_config)
test_dataset = sr.TabularDS(test_df, single_row_config)

model = sr.TabularRegressor(single_row_config, d_model=D_MODEL, n_heads=N_HEADS, lr=LR)

# Test model with a small batch
test_batch = train_dataset[0:10]
test_predictions = model.predict_step(test_batch)
print(f"Test prediction shape: {test_predictions.shape}")

# Create data loaders
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

print("\n## Model Training")
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

print("Starting training...")
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

print("\n## Running inference on validation set")
y_scaled = []
y_hat_scaled = []

# Get predictions on validation set
for batch in val_dataloader:
    y_scaled.append(batch.target)
    preds = model.predict_step(batch)
    y_hat_scaled.append(preds)

y_scaled = torch.cat(y_scaled, dim=0).squeeze().numpy()
y_hat_scaled = torch.cat(y_hat_scaled, dim=0).squeeze().numpy()

print(f"Scaled predictions shape: {y_scaled.shape}, {y_hat_scaled.shape}")

# Unscale predictions and targets for proper evaluation
y_unscaled = scaler_target.inverse_transform(y_scaled.reshape(-1, 1)).squeeze()
y_hat_unscaled = scaler_target.inverse_transform(y_hat_scaled.reshape(-1, 1)).squeeze()

print("\n## Evaluation Metrics")
# Calculate metrics on unscaled data
mse = mean_squared_error(y_unscaled, y_hat_unscaled)
rmse = np.sqrt(mse)

# Calculate RMSLE (Root Mean Squared Logarithmic Error)
# Add small epsilon to avoid log(0)
epsilon = 1e-8
rmsle = np.sqrt(mean_squared_log_error(
    np.maximum(y_unscaled, epsilon),
    np.maximum(y_hat_unscaled, epsilon)
))

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Root Mean Squared Logarithmic Error (RMSLE): {rmsle:.4f}")

# Create results dataframe
res_df = pd.DataFrame({
    "Actual": y_unscaled,
    "Predicted": y_hat_unscaled
})

print(f"\nActual vs Predicted statistics:")
print(f"Actual - Mean: {y_unscaled.mean():.2f}, Std: {y_unscaled.std():.2f}")
print(f"Predicted - Mean: {y_hat_unscaled.mean():.2f}, Std: {y_hat_unscaled.std():.2f}")

# Plot results
print("\n## Plotting Results")
plot_prediction_analysis(
    df=res_df,
    name="Hephaestus Insurance Premium Prediction",
    y_col="Actual",
    y_hat_col="Predicted",
)

print(f"\nTraining completed successfully!")
print(f"Final metrics:")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  RMSLE: {rmsle:.4f}")