# ruff: noqa: F402
# Enhanced Turbine NOx Prediction with Pre-training + Fine-tuning + Feature Interactions
# %%
import glob
import os
from pathlib import Path

import altair as alt
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

# %% Pre-training phase - Masked Tabular Modeling
print("Starting pre-training phase...")

X_train_pretrain = df_train.drop(columns=["target"])
train_dataset_pretrain = sr.TabularDS(X_train_pretrain, model_config_mtm)
X_test_pretrain = df_test.drop(columns=["target"])
test_dataset_pretrain = sr.TabularDS(X_test_pretrain, model_config_mtm)

# Create MTM model
mtm_model = sr.MaskedTabularModeling(
    model_config_mtm,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    lr=LR,
)

# Create dataloaders for pre-training
train_dataloader_pretrain = torch.utils.data.DataLoader(
    train_dataset_pretrain,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=sr.training.masked_tabular_collate_fn,
)
val_dataloader_pretrain = torch.utils.data.DataLoader(
    test_dataset_pretrain,
    batch_size=BATCH_SIZE,
    collate_fn=sr.training.masked_tabular_collate_fn,
)

# Train pre-training model
retrain_model = True
pretrained_model_dir = Path("checkpoints/turbine_enhanced_pretrained")
pretrained_model_dir.mkdir(parents=True, exist_ok=True)
pre_trained_models = list(pretrained_model_dir.glob("*.ckpt"))

if retrain_model or not any(LOGGER_VARIANT_NAME in p.stem for p in pre_trained_models):
    print("Pre-training model...")

    logger_pretrain = TensorBoardLogger("runs", name=f"{LOGGER_VARIANT_NAME}_pretrain")
    model_summary = ModelSummary(max_depth=3)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, min_delta=0.0001, mode="min"
    )
    progress_bar = TQDMProgressBar(leave=False)

    trainer_pretrain = L.Trainer(
        max_epochs=200,
        logger=logger_pretrain,
        callbacks=[early_stopping, progress_bar, model_summary],
        log_every_n_steps=1,
    )

    trainer_pretrain.fit(
        mtm_model,
        train_dataloaders=train_dataloader_pretrain,
        val_dataloaders=val_dataloader_pretrain,
    )

    # Save checkpoint
    checkpoint_path = pretrained_model_dir / f"{LOGGER_VARIANT_NAME}_pretrain.ckpt"
    trainer_pretrain.save_checkpoint(str(checkpoint_path))
    print(f"Saved pre-trained model to {checkpoint_path}")
else:
    print("Loading existing pre-trained model...")
    checkpoint_path = next(
        p for p in pre_trained_models if LOGGER_VARIANT_NAME in p.stem
    )
    mtm_model = sr.MaskedTabularModeling.load_from_checkpoint(str(checkpoint_path))

print("Pre-training phase completed!")


# %% Fine-tuning functions with enhanced feature interactions
def train_enhanced_hephaestus_model(
    df_train,
    df_test,
    model_config_reg,
    mtm_model,
    label_ratio=1.0,
    use_interactions=True,
):
    """
    Train enhanced Hephaestus model with pre-trained encoder and feature interactions.
    """
    # Sample training data based on label ratio
    train_df_subset = df_train.sample(frac=label_ratio, random_state=42)

    # Create datasets
    train_dataset = sr.TabularDS(train_df_subset, model_config_reg)
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
            train_df=train_df_subset,
            target_col="target",
            d_model=D_MODEL,
            n_heads=N_HEADS,
            lr=LR,
            aggressive_interactions=False,  # Use conservative interactions
        )

        # Transfer pre-trained weights to the base encoder
        enhanced_model.base_encoder.load_state_dict(
            mtm_model.model.tabular_encoder.state_dict()
        )

        model_type = "Enhanced"
        print(
            f"Created enhanced model with {sum(p.numel() for p in enhanced_model.parameters())} parameters"
        )
    else:
        # Create standard regressor
        enhanced_model = sr.TabularRegressor(
            model_config=model_config_reg,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            lr=LR,
        )

        # Transfer pre-trained weights
        enhanced_model.model.tabular_encoder.load_state_dict(
            mtm_model.model.tabular_encoder.state_dict()
        )

        model_type = "Standard"
        print(
            f"Created standard model with {sum(p.numel() for p in enhanced_model.parameters())} parameters"
        )

    # Training configuration
    logger_name = f"Enhanced_{model_type}_{LOGGER_VARIANT_NAME}_{label_ratio}"
    print(f"Fine-tuning {model_type} model with {label_ratio*100}% data...")

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

    # Fine-tune the model
    trainer.fit(
        enhanced_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )

    # Evaluate on test set using trainer.predict
    predictions = trainer.predict(enhanced_model, test_dataloader)
    y_hat = torch.cat(predictions, dim=0).squeeze().detach().numpy()

    # Get true values
    y = []
    for batch in test_dataloader:
        y.append(batch.target)
    y = torch.cat(y, dim=0).squeeze().detach().numpy()

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
        "label_ratio": label_ratio,
        "model_type": model_type,
    }


def train_basic_hephaestus_model(
    df_train,
    df_test,
    model_config_reg,
    mtm_model,
    label_ratio=1.0,
):
    """
    Train basic Hephaestus model with pre-trained encoder (no feature interactions).
    """
    # Sample training data based on label ratio
    train_df_subset = df_train.sample(frac=label_ratio, random_state=42)

    # Create datasets
    train_dataset = sr.TabularDS(train_df_subset, model_config_reg)
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

    # Create standard regressor
    regressor = sr.TabularRegressor(
        model_config=model_config_reg,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        lr=LR,
    )

    # Transfer pre-trained weights
    regressor.model.tabular_encoder.load_state_dict(
        mtm_model.model.tabular_encoder.state_dict()
    )

    model_type = "Basic"
    print(
        f"Created basic model with {sum(p.numel() for p in regressor.parameters())} parameters"
    )

    # Training configuration
    logger_name = f"Basic_{model_type}_{LOGGER_VARIANT_NAME}_{label_ratio}"
    print(f"Fine-tuning {model_type} model with {label_ratio*100}% data...")

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

    # Fine-tune the model
    trainer.fit(
        regressor,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )

    # Evaluate on test set using trainer.predict
    predictions = trainer.predict(regressor, test_dataloader)
    y_hat = torch.cat(predictions, dim=0).squeeze().detach().numpy()

    # Get true values
    y = []
    for batch in test_dataloader:
        y.append(batch.target)
    y = torch.cat(y, dim=0).squeeze().detach().numpy()

    # Convert to original scale
    y_unscaled = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_hat_unscaled = target_scaler.inverse_transform(y_hat.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_unscaled, y_hat_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_unscaled, y_hat_unscaled)

    return {
        "model": regressor,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_unscaled,
        "y_pred": y_hat_unscaled,
        "label_ratio": label_ratio,
        "model_type": model_type,
    }


# %% Baseline model functions (same as turbine_limited.py)
def train_linear_regression(df_train, df_test, label_ratio=1.0):
    """Train Linear Regression model."""
    train_df_subset = df_train.sample(frac=label_ratio, random_state=42)
    X_train = train_df_subset.drop(columns=["target"]).select_dtypes(
        include=[np.number]
    )
    y_train = train_df_subset["target"]

    X_test = df_test.drop(columns=["target"]).select_dtypes(include=[np.number])
    y_test = df_test["target"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert to original scale
    y_test_unscaled = target_scaler.inverse_transform(
        y_test.values.reshape(-1, 1)
    ).flatten()
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
        "label_ratio": label_ratio,
    }


def train_random_forest(df_train, df_test, label_ratio=1.0):
    """Train Random Forest model."""
    train_df_subset = df_train.sample(frac=label_ratio, random_state=42)
    X_train = train_df_subset.drop(columns=["target"]).select_dtypes(
        include=[np.number]
    )
    y_train = train_df_subset["target"]

    X_test = df_test.drop(columns=["target"]).select_dtypes(include=[np.number])
    y_test = df_test["target"]

    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert to original scale
    y_test_unscaled = target_scaler.inverse_transform(
        y_test.values.reshape(-1, 1)
    ).flatten()
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
        "label_ratio": label_ratio,
    }


def train_xgboost(df_train, df_test, label_ratio=1.0):
    """Train XGBoost model."""
    train_df_subset = df_train.sample(frac=label_ratio, random_state=42)
    X_train = train_df_subset.drop(columns=["target"]).select_dtypes(
        include=[np.number]
    )
    y_train = train_df_subset["target"]

    X_test = df_test.drop(columns=["target"]).select_dtypes(include=[np.number])
    y_test = df_test["target"]

    model = XGBRegressor(random_state=0, verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert to original scale
    y_test_unscaled = target_scaler.inverse_transform(
        y_test.values.reshape(-1, 1)
    ).flatten()
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
        "label_ratio": label_ratio,
    }


# %% Train and evaluate models on different data fractions
print("\n" + "=" * 80)
print("TRAINING AND EVALUATION PHASE")
print("=" * 80)

data_fractions = [1.0]

# Results storage
enhanced_results = []
standard_results = []
basic_results = []
lr_results = []
rf_results = []
xgb_results = []

for fraction in data_fractions:
    print(f"\nTraining with {fraction * 100}% of labeled data:")

    # Train Enhanced Hephaestus (with feature interactions)
    print("Training Enhanced Hephaestus (with feature interactions)...")
    enhanced_result = train_enhanced_hephaestus_model(
        df_train,
        df_test,
        model_config_reg,
        mtm_model,
        label_ratio=fraction,
        use_interactions=True,
    )
    enhanced_results.append(enhanced_result)

    # Train Standard Hephaestus (without feature interactions)
    print("Training Standard Hephaestus (without feature interactions)...")
    standard_result = train_enhanced_hephaestus_model(
        df_train,
        df_test,
        model_config_reg,
        mtm_model,
        label_ratio=fraction,
        use_interactions=False,
    )
    standard_results.append(standard_result)

    # Train Basic Hephaestus (basic pre-train + fine-tune)
    print("Training Basic Hephaestus (basic pre-train + fine-tune)...")
    basic_result = train_basic_hephaestus_model(
        df_train,
        df_test,
        model_config_reg,
        mtm_model,
        label_ratio=fraction,
    )
    basic_results.append(basic_result)

    # Train baseline models
    print("Training Linear Regression...")
    lr_result = train_linear_regression(df_train, df_test, label_ratio=fraction)
    lr_results.append(lr_result)

    print("Training Random Forest...")
    rf_result = train_random_forest(df_train, df_test, label_ratio=fraction)
    rf_results.append(rf_result)

    print("Training XGBoost...")
    xgb_result = train_xgboost(df_train, df_test, label_ratio=fraction)
    xgb_results.append(xgb_result)

    # Print results
    print(f"Results with {fraction * 100}% of labeled data:")
    print(
        f"  Enhanced Hephaestus MSE: {enhanced_result['mse']:.3f}, "
        f"RMSE: {enhanced_result['rmse']:.3f}, MAE: {enhanced_result['mae']:.3f}"
    )
    print(
        f"  Standard Hephaestus MSE: {standard_result['mse']:.3f}, "
        f"RMSE: {standard_result['rmse']:.3f}, MAE: {standard_result['mae']:.3f}"
    )
    print(
        f"  Basic Hephaestus MSE: {basic_result['mse']:.3f}, "
        f"RMSE: {basic_result['rmse']:.3f}, MAE: {basic_result['mae']:.3f}"
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

# %% Create results dataframe and visualizations
results_rows = []

# Add all results
for result in enhanced_results:
    results_rows.append(
        {
            "Model": "Enhanced Hephaestus",
            "Label Ratio": result["label_ratio"],
            "MSE": result["mse"],
            "RMSE": result["rmse"],
            "MAE": result["mae"],
        }
    )

for result in standard_results:
    results_rows.append(
        {
            "Model": "Standard Hephaestus",
            "Label Ratio": result["label_ratio"],
            "MSE": result["mse"],
            "RMSE": result["rmse"],
            "MAE": result["mae"],
        }
    )

for result in basic_results:
    results_rows.append(
        {
            "Model": "Basic Hephaestus",
            "Label Ratio": result["label_ratio"],
            "MSE": result["mse"],
            "RMSE": result["rmse"],
            "MAE": result["mae"],
        }
    )

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

results_df = pd.DataFrame(results_rows)

# %% Print final results
print("\n" + "=" * 80)
print("FINAL RESULTS COMPARISON")
print("=" * 80)
print(results_df)

# %% Create visualizations
# MSE comparison chart
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
        title="Enhanced vs Standard vs Basic Hephaestus: MSE Comparison with Different Data Fractions",
        width=700,
        height=400,
    )
)

# Save chart
os.makedirs("result_images", exist_ok=True)
mse_chart.save("result_images/enhanced_vs_standard_comparison.png")
mse_chart.show()

# %% Calculate improvement analysis
print("\n" + "=" * 80)
print("IMPROVEMENT ANALYSIS")
print("=" * 80)

improvement_rows = []
for fraction in data_fractions:
    enhanced_result = next(r for r in enhanced_results if r["label_ratio"] == fraction)
    standard_result = next(r for r in standard_results if r["label_ratio"] == fraction)
    basic_result = next(r for r in basic_results if r["label_ratio"] == fraction)

    # Calculate improvement of enhanced over standard
    improvement_pct_vs_standard = (
        (standard_result["mse"] - enhanced_result["mse"]) / standard_result["mse"]
    ) * 100

    # Calculate improvement of enhanced over basic
    improvement_pct_vs_basic = (
        (basic_result["mse"] - enhanced_result["mse"]) / basic_result["mse"]
    ) * 100

    improvement_rows.append(
        {
            "Label Ratio": fraction,
            "Enhanced MSE": enhanced_result["mse"],
            "Standard MSE": standard_result["mse"],
            "Basic MSE": basic_result["mse"],
            "Improvement vs Standard (%)": improvement_pct_vs_standard,
            "Improvement vs Basic (%)": improvement_pct_vs_basic,
        }
    )

improvement_df = pd.DataFrame(improvement_rows)
print(improvement_df)

print("\n" + "=" * 80)
print("TRAINING COMPLETED!")
print("=" * 80)
print(
    "Key findings will be in the results above and visualizations saved to result_images/"
)

# %%
