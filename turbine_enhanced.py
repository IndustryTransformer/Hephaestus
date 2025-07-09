# ruff: noqa: F402
# Enhanced Turbine NOx Prediction with Automated Feature Engineering
# %%
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as L
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

# %% Configuration
D_MODEL = 128
N_HEADS = 4
LR = 0.0001
BATCH_SIZE = 64
name = "EnhancedInteractionModel"
LOGGER_VARIANT_NAME = f"{name}_D{D_MODEL}_H{N_HEADS}_LR{LR}"

# %% Load and preprocess data
csv_files = glob.glob("data/nox/*.csv")

# Read and combine all files
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    filename = Path(file).stem
    temp_df["filename"] = filename
    df_list.append(temp_df)

# Concatenate all dataframes
df = pd.concat(df_list, ignore_index=True)
df.columns = df.columns.str.lower()
df = df.rename(columns={"nox": "target"})

# Add year column from filename for time-based split
df["year"] = df["filename"].str.extract(r"(\d{4})")
df.dropna(subset=["year"], inplace=True)
df["year"] = df["year"].astype(int)

# Scale the non-target numerical columns
scaler = StandardScaler()
target_scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns

# Store original target values for unscaled comparison
original_target = df["target"].copy()

# Exclude year from scaling
if "year" in numeric_cols:
    numeric_cols = numeric_cols.drop("year")

# Scale all numeric columns including target
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Fit separate scaler for target to enable inverse transformation
target_scaler.fit(original_target.values.reshape(-1, 1))

df["cat_column"] = "category"

print(f"Dataset shape: {df.shape}")
print(f"Numeric columns: {len(df.select_dtypes(include=['number']).columns)}")
print(f"Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")

# %% Create model configuration and datasets
single_row_config = sr.SingleRowConfig.generate(df.drop(columns=["year"]), "target")
# Time-based split
train_df = df[df["year"].isin([2011, 2012, 2013])].copy()
test_df = df[df["year"].isin([2014, 2015])].copy()

train_dataset = sr.TabularDS(train_df, single_row_config)
test_dataset = sr.TabularDS(test_df, single_row_config)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# %% Create enhanced model with automated feature engineering
print("Creating enhanced model with automated feature interactions...")

# Create the enhanced model
enhanced_model = sr.create_enhanced_model(
    model_config=single_row_config,
    train_df=train_df,
    target_col="target",
    d_model=D_MODEL,
    n_heads=N_HEADS,
    lr=LR,
    aggressive_interactions=True,  # Use more aggressive interaction settings
)

print(
    "Enhanced model created with",
    f"{sum(p.numel() for p in enhanced_model.parameters())}",
    "parameters",
)

# %% Create dataloaders
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

# %% Train the enhanced model
logger_name = f"{LOGGER_VARIANT_NAME}"
print(f"Training enhanced model with logger: {logger_name}")

logger = TensorBoardLogger("runs", name=logger_name)
model_summary = ModelSummary(max_depth=3)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, min_delta=0.001, mode="min"
)
progress_bar = TQDMProgressBar(leave=False)

trainer = L.Trainer(
    max_epochs=300,
    logger=logger,
    callbacks=[early_stopping, progress_bar, model_summary],
    log_every_n_steps=1,
    gradient_clip_val=1.0,  # Add gradient clipping for stability
)

trainer.fit(
    enhanced_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

# %% Evaluate enhanced model
print("Evaluating enhanced model...")

# Get predictions on training data
y_train = []
y_hat_train = []
for batch in train_dataloader:
    y_train.append(batch.target)
    preds = enhanced_model.predict_step(batch)
    y_hat_train.append(preds)

y_train = torch.cat(y_train, dim=0).squeeze().numpy()
y_hat_train = torch.cat(y_hat_train, dim=0).squeeze().numpy()

# Get predictions on test data
y_test = []
y_hat_test = []
for batch in val_dataloader:
    y_test.append(batch.target)
    preds = enhanced_model.predict_step(batch)
    y_hat_test.append(preds)

y_test = torch.cat(y_test, dim=0).squeeze().numpy()
y_hat_test = torch.cat(y_hat_test, dim=0).squeeze().numpy()

# Calculate metrics
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print("Enhanced Model Results:")
print(f"Training MSE: {train_mse:.4f} | RMSE: {train_rmse:.4f}")
print(f"Test MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f}")

# %% Plot results for enhanced model
res_df_enhanced = pd.DataFrame({"Actual": y_test, "Predicted": y_hat_test})

plot_prediction_analysis(
    df=res_df_enhanced,
    name="Enhanced Hephaestus (with Feature Interactions)",
    y_col="Actual",
    y_hat_col="Predicted",
)

# %% Compare with baseline model
print("Training baseline model for comparison...")

# Create baseline model (without interactions)
baseline_model = sr.TabularRegressor(
    single_row_config,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    lr=LR,
)

# Train baseline model
baseline_trainer = L.Trainer(
    max_epochs=300,
    logger=False,
    callbacks=[early_stopping, progress_bar],
    enable_checkpointing=False,
    enable_model_summary=False,
)

baseline_trainer.fit(
    baseline_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

# Evaluate baseline model
y_baseline = []
y_hat_baseline = []
for batch in val_dataloader:
    y_baseline.append(batch.target)
    preds = baseline_model.predict_step(batch)
    y_hat_baseline.append(preds)

y_baseline = torch.cat(y_baseline, dim=0).squeeze().numpy()
y_hat_baseline = torch.cat(y_hat_baseline, dim=0).squeeze().numpy()

baseline_mse = mean_squared_error(y_baseline, y_hat_baseline)
baseline_rmse = np.sqrt(baseline_mse)

print("Baseline Model Results:")
print(f"Test MSE: {baseline_mse:.4f} | RMSE: {baseline_rmse:.4f}")

# %% Compare with scikit-learn models
print("Comparing with scikit-learn models...")

# Prepare data for scikit-learn
if "target" in numeric_cols:
    X = df[numeric_cols.drop("target")]
else:
    X = df[numeric_cols]
y = df["target"]

# Time-based split for scikit-learn models
X_train_sk = X[df["year"].isin([2011, 2012, 2013])]
X_test_sk = X[df["year"].isin([2014, 2015])]
y_train_sk = y[df["year"].isin([2011, 2012, 2013])]
y_test_sk = y[df["year"].isin([2014, 2015])]

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_sk, y_train_sk)
y_pred_lr = linear_model.predict(X_test_sk)
mse_linear = mean_squared_error(y_test_sk, y_pred_lr)
rmse_linear = np.sqrt(mse_linear)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_sk, y_train_sk)
y_pred_rf = rf_model.predict(X_test_sk)
mse_rf = mean_squared_error(y_test_sk, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

# %% Calculate unscaled metrics for validation dataset
print("Computing unscaled metrics on validation dataset...")

# Get original test indices to align with unscaled data
test_indices = test_df.index
original_test_targets = original_target.iloc[test_indices].values

# Unscale predictions
y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_hat_test_unscaled = target_scaler.inverse_transform(
    y_hat_test.reshape(-1, 1)
).flatten()
y_baseline_unscaled = target_scaler.inverse_transform(
    y_baseline.reshape(-1, 1)
).flatten()
y_hat_baseline_unscaled = target_scaler.inverse_transform(
    y_hat_baseline.reshape(-1, 1)
).flatten()

# Calculate unscaled metrics
test_mse_unscaled = mean_squared_error(original_test_targets, y_hat_test_unscaled)
test_rmse_unscaled = np.sqrt(test_mse_unscaled)
baseline_mse_unscaled = mean_squared_error(
    original_test_targets, y_hat_baseline_unscaled
)
baseline_rmse_unscaled = np.sqrt(baseline_mse_unscaled)

# For scikit-learn models (already on original scale)
mse_linear_unscaled = mse_linear
rmse_linear_unscaled = rmse_linear
mse_rf_unscaled = mse_rf
rmse_rf_unscaled = rmse_rf

# %% Print comparison results
print("\n" + "=" * 70)
print("MODEL COMPARISON RESULTS (SCALED)")
print("=" * 70)
print(f"{'Model':<35} {'MSE':<10} {'RMSE':<10} {'Improvement':<15}")
print("-" * 70)
print(
    f"{'Linear Regression':<35} {mse_linear:<10.4f} {rmse_linear:<10.4f} "
    f"{'baseline':<15}"
)
rf_improvement = (mse_linear - mse_rf) / mse_linear * 100
rf_perf = f"{rf_improvement:>+6.1f}%"
print(f"{'Random Forest':<35} {mse_rf:<10.4f} {rmse_rf:<10.4f} {rf_perf:<15}")
baseline_improvement = (mse_linear - baseline_mse) / mse_linear * 100
base_perf = f"{baseline_improvement:>+6.1f}%"
print(
    f"{'Hephaestus Baseline':<35} {baseline_mse:<10.4f} "
    f"{baseline_rmse:<10.4f} {base_perf:<15}"
)
enhanced_improvement = (mse_linear - test_mse) / mse_linear * 100
enh_perf = f"{enhanced_improvement:>+6.1f}%"
print(
    f"{'Enhanced Hephaestus':<35} {test_mse:<10.4f} {test_rmse:<10.4f} "
    f"{enh_perf:<15}"
)
print("-" * 70)
vs_baseline_improvement = (baseline_mse - test_mse) / baseline_mse * 100
print(f"Enhanced vs Baseline improvement: {vs_baseline_improvement:+.1f}%")
print("=" * 70)

print("\n" + "=" * 70)
print("MODEL COMPARISON RESULTS (UNSCALED - ORIGINAL NOX VALUES)")
print("=" * 70)
print(f"{'Model':<35} {'MSE':<10} {'RMSE':<10} {'Improvement':<15}")
print("-" * 70)
print(
    f"{'Linear Regression':<35} {mse_linear_unscaled:<10.4f} "
    f"{rmse_linear_unscaled:<10.4f} {'baseline':<15}"
)
rf_improvement_unscaled = (
    (mse_linear_unscaled - mse_rf_unscaled) / mse_linear_unscaled * 100
)
rf_perf_unscaled = f"{rf_improvement_unscaled:>+6.1f}%"
print(
    f"{'Random Forest':<35} {mse_rf_unscaled:<10.4f} "
    f"{rmse_rf_unscaled:<10.4f} {rf_perf_unscaled:<15}"
)
baseline_improvement_unscaled = (
    (mse_linear_unscaled - baseline_mse_unscaled) / mse_linear_unscaled * 100
)
base_perf_unscaled = f"{baseline_improvement_unscaled:>+6.1f}%"
print(
    f"{'Hephaestus Baseline':<35} {baseline_mse_unscaled:<10.4f} "
    f"{baseline_rmse_unscaled:<10.4f} {base_perf_unscaled:<15}"
)
enhanced_improvement_unscaled = (
    (mse_linear_unscaled - test_mse_unscaled) / mse_linear_unscaled * 100
)
enh_perf_unscaled = f"{enhanced_improvement_unscaled:>+6.1f}%"
print(
    f"{'Enhanced Hephaestus':<35} {test_mse_unscaled:<10.4f} "
    f"{test_rmse_unscaled:<10.4f} {enh_perf_unscaled:<15}"
)
print("-" * 70)
vs_baseline_improvement_unscaled = (
    (baseline_mse_unscaled - test_mse_unscaled) / baseline_mse_unscaled * 100
)
print(f"Enhanced vs Baseline improvement: {vs_baseline_improvement_unscaled:+.1f}%")
print("=" * 70)

# %% Feature interaction analysis
print("\nFeature Interaction Analysis:")
lowrank_rank = enhanced_model.interaction_config["lowrank_rank"]
print(f"Model has {lowrank_rank} low-rank cross dimensions")
crossnet_layers = enhanced_model.interaction_config["crossnet_layers"]
print(f"Using {crossnet_layers} CrossNet layers")
use_cat_interactions = enhanced_model.interaction_config["use_cat_interactions"]
print(f"Categorical interactions: {use_cat_interactions}")
hash_size = enhanced_model.interaction_config["hash_size"]
print(f"Hash table size: {hash_size}")

# %% Show feature importance if available
try:
    adaptive_fe = sr.AdaptiveFeatureEngineering(single_row_config)
    important_pairs = adaptive_fe.analyze_features(train_df, "target", top_pairs=10)
    print("\nTop 10 important feature pairs:")
    numeric_feature_names = [
        col
        for col in train_df.select_dtypes(include=["number"]).columns
        if col != "target"
    ]
    for i, (idx1, idx2) in enumerate(important_pairs[:10]):
        feat1 = numeric_feature_names[idx1]
        feat2 = numeric_feature_names[idx2]
        print(f"{i + 1}. {feat1} × {feat2}")
except Exception as e:
    print(f"Could not analyze feature importance: {e}")

print("\nTraining completed! Check TensorBoard logs for detailed metrics.")
