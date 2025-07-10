# ruff: noqa: F402
# Enhanced Turbine NOx Prediction with Pre-training + Fine-tuning + Feature Interactions (100% data only)
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import hephaestus.single_row_models as sr

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% Configuration
D_MODEL = 128
N_HEADS = 4
LR = 0.0001 * 8  # Same as turbine_limited.py
BATCH_SIZE = 64 * 8  # Same as turbine_limited.py
name = "EnhancedWithPretraining"
LOGGER_VARIANT_NAME = f"{name}_D{D_MODEL}_H{N_HEADS}_LR{LR}"

# %% Load and preprocess data (same as turbine_limited.py)
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

# Remove CO column if it exists (same as turbine_limited.py)
if "co" in df.columns:
    df = df.drop(columns=["co"])
    print("Removed CO column from dataset")

df = df.rename(columns={"nox": "target"})
original_target = df["target"].copy()

# Scale the non-target numerical columns (same as turbine_limited.py)
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns
numeric_cols_without_target = numeric_cols.drop("target")
df[numeric_cols_without_target] = scaler.fit_transform(df[numeric_cols_without_target])

# Scale target separately and store the scaler
target_scaler = StandardScaler()
df["target"] = target_scaler.fit_transform(df[["target"]]).flatten()

# Add categorical column AFTER scaling
df["cat_column"] = "category"
df["cat_column"] = df["cat_column"].astype("category")

print(f"Dataset shape: {df.shape}")
print(f"Data types after processing: {df.dtypes}")

# %% Create model configurations
X_setup = df[df.columns.drop("target")]
model_config_mtm = sr.SingleRowConfig.generate(X_setup)  # For pre-training
model_config_reg = sr.SingleRowConfig.generate(df, target="target")  # For fine-tuning

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# %% Check for existing pre-trained model
pretrained_model_dir = Path("checkpoints/turbine_enhanced_pretrained")
pretrained_model_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = pretrained_model_dir / f"{LOGGER_VARIANT_NAME}_pretrain.ckpt"

if checkpoint_path.exists():
    print(f"Loading existing pre-trained model from {checkpoint_path}")
    # Create model first then load weights manually
    mtm_model = sr.MaskedTabularModeling(
        model_config_mtm,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        lr=LR,
    )
    # Load checkpoint and extract just the tabular encoder weights
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    encoder_state = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.tabular_encoder'):
            encoder_state[key] = value
    # Load only the encoder weights
    mtm_model.load_state_dict(encoder_state, strict=False)
else:
    print("No pre-trained model found. Please run the full script first to create pre-trained weights.")
    exit(1)

print("Using pre-trained model!")

# %% Training functions
def train_enhanced_hephaestus_model(df_train, df_test, model_config_reg, mtm_model, use_interactions=True):
    """Train enhanced Hephaestus model with pre-trained encoder and feature interactions."""
    
    # Create datasets (using 100% of data)
    train_dataset = sr.TabularDS(df_train, model_config_reg)
    test_dataset = sr.TabularDS(df_test, model_config_reg)
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=sr.training.tabular_collate_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=sr.training.tabular_collate_fn,
    )
    
    if use_interactions:
        # Create enhanced model with feature interactions
        enhanced_model = sr.create_enhanced_model(
            model_config=model_config_reg,
            train_df=df_train,
            target_col="target",
            d_model=D_MODEL,
            n_heads=N_HEADS,
            lr=LR,
            aggressive_interactions=False,  # Use conservative interactions
        )
        
        # Transfer pre-trained weights to the base encoder
        enhanced_model.base_encoder.load_state_dict(mtm_model.model.tabular_encoder.state_dict())
        
        model_type = "Enhanced"
        print(f"Created enhanced model with {sum(p.numel() for p in enhanced_model.parameters())} parameters")
        print(f"Feature interactions detected: {len(enhanced_model.important_pairs) if hasattr(enhanced_model, 'important_pairs') else 'N/A'}")
        
        # Print interaction config
        if hasattr(enhanced_model, 'interaction_config'):
            print(f"Interaction config: {enhanced_model.interaction_config}")
    else:
        # Create standard regressor
        enhanced_model = sr.TabularRegressor(
            model_config=model_config_reg,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            lr=LR,
        )
        
        # Transfer pre-trained weights
        enhanced_model.model.tabular_encoder.load_state_dict(mtm_model.model.tabular_encoder.state_dict())
        
        model_type = "Standard"
        print(f"Created standard model with {sum(p.numel() for p in enhanced_model.parameters())} parameters")
    
    # Training configuration
    logger_name = f"Fast_{model_type}_{LOGGER_VARIANT_NAME}_1.0"
    print(f"Fine-tuning {model_type} model with 100% data...")
    
    logger = TensorBoardLogger("runs", name=logger_name)
    model_summary = ModelSummary(max_depth=3)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, min_delta=0.001, mode="min"  # Reduced patience for speed
    )
    progress_bar = TQDMProgressBar(leave=False)
    
    trainer = L.Trainer(
        max_epochs=100,  # Reduced epochs for speed
        logger=logger,
        callbacks=[early_stopping, progress_bar, model_summary],
        log_every_n_steps=1,
    )
    
    # Fine-tune the model
    trainer.fit(
        enhanced_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )
    
    # Evaluate on test set
    y = []
    y_hat = []
    for batch in test_dataloader:
        y.append(batch.target)
        preds = enhanced_model.predict_step(batch)
        y_hat.append(preds)
    
    y = torch.cat(y, dim=0).squeeze().detach().numpy()
    y_hat = torch.cat(y_hat, dim=0).squeeze().detach().numpy()
    
    # Convert to original scale
    y_unscaled = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_hat_unscaled = target_scaler.inverse_transform(y_hat.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_unscaled, y_hat_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_unscaled, y_hat_unscaled)
    
    return {
        "model": enhanced_model,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_unscaled,
        "y_pred": y_hat_unscaled,
        "model_type": model_type,
    }

def train_baseline_model(df_train, df_test, model_type="LinearRegression"):
    """Train baseline models."""
    X_train = df_train.drop(columns=["target"]).select_dtypes(include=[np.number])
    y_train = df_train["target"]
    
    X_test = df_test.drop(columns=["target"]).select_dtypes(include=[np.number])
    y_test = df_test["target"]
    
    if model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "RandomForest":
        model = RandomForestRegressor(random_state=0)
    elif model_type == "XGBoost":
        model = XGBRegressor(random_state=0, verbosity=0)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Convert to original scale
    y_test_unscaled = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    
    return {
        "model": model,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_test_unscaled,
        "y_pred": y_pred_unscaled,
        "model_type": model_type,
    }

# %% Train and evaluate models
print("\n" + "="*80)
print("TRAINING AND EVALUATION PHASE (100% DATA)")
print("="*80)

# Train Enhanced Hephaestus (with feature interactions)
print("\n1. Training Enhanced Hephaestus (with feature interactions)...")
enhanced_result = train_enhanced_hephaestus_model(
    df_train, df_test, model_config_reg, mtm_model, use_interactions=True
)

# Train Standard Hephaestus (without feature interactions)
print("\n2. Training Standard Hephaestus (without feature interactions)...")
standard_result = train_enhanced_hephaestus_model(
    df_train, df_test, model_config_reg, mtm_model, use_interactions=False
)

# Train baseline models
print("\n3. Training baseline models...")
lr_result = train_baseline_model(df_train, df_test, "LinearRegression")
rf_result = train_baseline_model(df_train, df_test, "RandomForest")
xgb_result = train_baseline_model(df_train, df_test, "XGBoost")

# %% Print results
print("\n" + "="*80)
print("FINAL RESULTS COMPARISON (100% DATA)")
print("="*80)

results = [
    ("Enhanced Hephaestus (with interactions)", enhanced_result),
    ("Standard Hephaestus (no interactions)", standard_result),
    ("Linear Regression", lr_result),
    ("Random Forest", rf_result),
    ("XGBoost", xgb_result),
]

print(f"{'Model':<40} {'MSE':<10} {'RMSE':<10} {'MAE':<10}")
print("-" * 70)

for name, result in results:
    print(f"{name:<40} {result['mse']:<10.3f} {result['rmse']:<10.3f} {result['mae']:<10.3f}")

# %% Calculate improvements
enhanced_vs_standard = ((standard_result["mse"] - enhanced_result["mse"]) / standard_result["mse"]) * 100
enhanced_vs_best_baseline = min(lr_result["mse"], rf_result["mse"], xgb_result["mse"])
enhanced_vs_baseline_pct = ((enhanced_vs_best_baseline - enhanced_result["mse"]) / enhanced_vs_best_baseline) * 100

print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS")
print("="*80)
print(f"Enhanced vs Standard Hephaestus: {enhanced_vs_standard:+.1f}%")
print(f"Enhanced vs Best Baseline: {enhanced_vs_baseline_pct:+.1f}%")

if enhanced_vs_standard > 0:
    print("✅ Feature interactions provide meaningful improvement!")
else:
    print("❌ Feature interactions do not improve performance")

print("\n" + "="*80)
print("TRAINING COMPLETED!")
print("="*80)