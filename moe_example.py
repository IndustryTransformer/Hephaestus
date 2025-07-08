"""
Comprehensive comparison of MoE vs Standard Hephaestus models for turbine NOx prediction.

This script:
1. Trains both MoE and standard models with pre-training and fine-tuning
2. Evaluates performance metrics (MSE, RMSE, MAE)
3. Saves comparison results using polars
4. Outputs results to outputs/ directory
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelSummary, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hephaestus.single_row_models as sr

# Set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_output_directory():
    """Create outputs directory if it doesn't exist."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def load_and_preprocess_data():
    """Load and preprocess the turbine NOx data."""
    print("📊 Loading and preprocessing data...")

    # Load data
    csv_files = glob.glob("data/nox/*.csv")
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/nox/ directory")

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

    # Remove CO column if it exists
    if "co" in df.columns:
        df = df.drop(columns=["co"])
        print("Removed CO column from dataset")

    df = df.rename(columns={"nox": "target"})
    original_target = df["target"].copy()

    # Scale features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    numeric_cols_without_target = numeric_cols.drop("target")
    df[numeric_cols_without_target] = scaler.fit_transform(
        df[numeric_cols_without_target]
    )

    # Scale target separately
    target_scaler = StandardScaler()
    df["target"] = target_scaler.fit_transform(df[["target"]]).flatten()

    # Add multiple categorical columns for better MoE demonstration
    df["cat_column"] = "category"  # Original dummy category
    df["power_level"] = pd.cut(
        df[numeric_cols_without_target[0]], bins=3, labels=["low", "medium", "high"]
    )
    df["efficiency_tier"] = pd.cut(
        df[numeric_cols_without_target[1]],
        bins=4,
        labels=["tier1", "tier2", "tier3", "tier4"],
    )

    # Convert categorical columns to proper type
    categorical_cols = ["cat_column", "power_level", "efficiency_tier"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    print(f"Dataset shape: {df.shape}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numeric columns: {len(numeric_cols_without_target)}")

    return df, target_scaler, original_target


def create_model_configs(df):
    """Create model configurations for standard and MoE models."""

    # Standard configuration (no MoE)
    model_config_standard = sr.SingleRowConfig.generate(
        df, target="target", use_moe=False
    )

    # MoE configuration
    model_config_moe = sr.SingleRowConfig.generate(
        df,
        target="target",
        use_moe=True,
        num_experts=8,
        expert_balance=[0.4, 0.3, 0.3],  # 40% numeric, 30% categorical, 30% interaction
        moe_top_k=2,
        adaptive_routing=True,
        categorical_aware=True,
    )

    return model_config_standard, model_config_moe


def pretrain_model(model_config, model_type="standard", max_epochs=100):
    """Pre-train a model using masked language modeling."""
    print(f"🔄 Pre-training {model_type} model...")

    # Model parameters
    D_MODEL = 128
    N_HEADS = 4
    LR = 0.0001 * 4
    BATCH_SIZE = 64 * 4

    # Create pre-training dataset (without target for MTM)
    df, _, _ = load_and_preprocess_data()
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    # Remove target for pre-training
    X_train = df_train.drop(columns=["target"])
    X_val = df_val.drop(columns=["target"])

    # Create MTM config (without target) but preserve MoE settings
    mtm_config = sr.SingleRowConfig.generate(
        X_train,
        use_moe=model_config.use_moe,
        num_experts=model_config.num_experts,
        expert_balance=model_config.expert_balance,
        moe_top_k=model_config.moe_top_k,
        adaptive_routing=model_config.adaptive_routing,
        categorical_aware=model_config.categorical_aware,
    )

    # Create datasets
    train_dataset = sr.TabularDS(X_train, mtm_config)
    val_dataset = sr.TabularDS(X_val, mtm_config)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=sr.training.masked_tabular_collate_fn,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=sr.training.masked_tabular_collate_fn,
    )

    # Create pre-training model
    mtm_model = sr.MaskedTabularModeling(
        mtm_config,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        lr=LR,
        use_linear_numeric_embedding=True,
        numeric_embedding_type="simple",
    )

    print(f"MTM model MoE enabled: {mtm_config.use_moe}")
    print(f"MTM model columns: {mtm_config.n_columns}")
    print(f"MTM model numeric cols: {mtm_config.n_numeric_cols}")
    print(f"MTM model categorical cols: {mtm_config.n_cat_cols}")

    # Training configuration
    logger_name = f"Pretrain_{model_type}_D{D_MODEL}_H{N_HEADS}"
    logger = TensorBoardLogger("runs", name=logger_name)
    model_summary = ModelSummary(max_depth=2)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, min_delta=0.001, mode="min"
    )
    progress_bar = TQDMProgressBar(leave=False)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stopping, progress_bar, model_summary],
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(
        mtm_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print(f"✅ Pre-training {model_type} completed!")
    return mtm_model


def finetune_model(
    pretrained_model, model_config, model_type="standard", max_epochs=200
):
    """Fine-tune a pre-trained model on the regression task."""
    print(f"🎯 Fine-tuning {model_type} model...")

    # Model parameters
    D_MODEL = 128
    N_HEADS = 4
    LR = 0.0001 * 2  # Lower learning rate for fine-tuning
    BATCH_SIZE = 64 * 4

    # Load data
    df, target_scaler, _ = load_and_preprocess_data()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = sr.TabularDS(df_train, model_config)
    test_dataset = sr.TabularDS(df_test, model_config)

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

    # Create regressor with pre-trained encoder
    regressor = sr.TabularRegressor(
        model_config=model_config,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        lr=LR,
        use_linear_numeric_embedding=True,
        numeric_embedding_type="simple",
    )

    # Transfer pre-trained weights
    regressor.model.tabular_encoder = pretrained_model.model.tabular_encoder

    # Training configuration
    logger_name = f"Finetune_{model_type}_D{D_MODEL}_H{N_HEADS}"
    logger = TensorBoardLogger("runs", name=logger_name)
    model_summary = ModelSummary(max_depth=2)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, min_delta=0.001, mode="min"
    )
    progress_bar = TQDMProgressBar(leave=False)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stopping, progress_bar, model_summary],
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(
        regressor,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )

    print(f"✅ Fine-tuning {model_type} completed!")
    return regressor, df_test, target_scaler


def evaluate_model(model, df_test, target_scaler, model_type="standard"):
    """Evaluate model performance and return metrics."""
    print(f"📊 Evaluating {model_type} model...")

    # Create test dataset and dataloader
    model_config = model.model.model_config
    test_dataset = sr.TabularDS(df_test, model_config)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        collate_fn=sr.training.tabular_collate_fn,
    )

    # Get predictions
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            y_true.append(batch.target)
            predictions = model.predict_step(batch)
            y_pred.append(predictions)

    # Concatenate all predictions
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    # Convert back to original scale
    y_true_unscaled = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)

    print(f"{model_type} Model Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")

    return {
        "model_type": model_type,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_true_unscaled,
        "y_pred": y_pred_unscaled,
    }


def train_and_evaluate_both_models():
    """Train and evaluate both standard and MoE models."""
    print("🚀 Starting comprehensive model comparison...")

    # Load data and create configs
    df, _, _ = load_and_preprocess_data()
    model_config_standard, model_config_moe = create_model_configs(df)

    # Results storage
    results = []

    # Train Standard Model
    print("\n" + "=" * 60)
    print("🔧 TRAINING STANDARD MODEL")
    print("=" * 60)

    # Pre-train standard model
    pretrained_standard = pretrain_model(model_config_standard, "simple", max_epochs=50)

    # Fine-tune standard model
    standard_model, df_test, target_scaler = finetune_model(
        pretrained_standard, model_config_standard, "simple", max_epochs=100
    )

    # Evaluate standard model
    standard_results = evaluate_model(standard_model, df_test, target_scaler, "simple")
    results.append(standard_results)

    # Train MoE Model
    print("\n" + "=" * 60)
    print("🧠 TRAINING MOE MODEL")
    print("=" * 60)

    # Pre-train MoE model
    pretrained_moe = pretrain_model(model_config_moe, "moe", max_epochs=50)

    # Fine-tune MoE model
    moe_model, _, _ = finetune_model(
        pretrained_moe, model_config_moe, "moe", max_epochs=100
    )

    # Evaluate MoE model
    moe_results = evaluate_model(moe_model, df_test, target_scaler, "moe")
    results.append(moe_results)

    return results, standard_model, moe_model


def save_comparison_results(results):
    """Save comparison results using polars."""
    print("\n📁 Saving comparison results...")

    # Create output directory
    output_dir = create_output_directory()

    # Create summary comparison dataframe
    summary_data = []
    for result in results:
        summary_data.append(
            {
                "model_type": result["model_type"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "mae": result["mae"],
            }
        )

    summary_df = pl.DataFrame(summary_data)

    # Calculate improvements
    standard_mse = summary_df.filter(pl.col("model_type") == "standard")["mse"][0]
    moe_mse = summary_df.filter(pl.col("model_type") == "moe")["mse"][0]
    mse_improvement = ((standard_mse - moe_mse) / standard_mse) * 100

    standard_rmse = summary_df.filter(pl.col("model_type") == "standard")["rmse"][0]
    moe_rmse = summary_df.filter(pl.col("model_type") == "moe")["rmse"][0]
    rmse_improvement = ((standard_rmse - moe_rmse) / standard_rmse) * 100

    standard_mae = summary_df.filter(pl.col("model_type") == "standard")["mae"][0]
    moe_mae = summary_df.filter(pl.col("model_type") == "moe")["mae"][0]
    mae_improvement = ((standard_mae - moe_mae) / standard_mae) * 100

    # Add improvement row
    improvement_data = {
        "model_type": "moe_improvement_%",
        "mse": mse_improvement,
        "rmse": rmse_improvement,
        "mae": mae_improvement,
    }

    summary_df = pl.concat([summary_df, pl.DataFrame([improvement_data])])

    # Save summary
    summary_path = output_dir / "model_comparison_summary.csv"
    summary_df.write_csv(summary_path)

    # Create detailed predictions comparison
    predictions_data = []
    for result in results:
        for i, (true_val, pred_val) in enumerate(
            zip(result["y_true"], result["y_pred"])
        ):
            predictions_data.append(
                {
                    "model_type": result["model_type"],
                    "sample_id": i,
                    "y_true": true_val,
                    "y_pred": pred_val,
                    "absolute_error": abs(true_val - pred_val),
                    "squared_error": (true_val - pred_val) ** 2,
                }
            )

    predictions_df = pl.DataFrame(predictions_data)
    predictions_path = output_dir / "model_predictions_comparison.csv"
    predictions_df.write_csv(predictions_path)

    # Print results
    print("\n" + "=" * 60)
    print("🎯 FINAL COMPARISON RESULTS")
    print("=" * 60)
    print(summary_df)

    print(f"\n📊 Performance Improvements (MoE vs Standard):")
    print(f"MSE improvement: {mse_improvement:+.2f}%")
    print(f"RMSE improvement: {rmse_improvement:+.2f}%")
    print(f"MAE improvement: {mae_improvement:+.2f}%")

    if mse_improvement > 0:
        print("🎉 MoE model outperforms standard model!")
    else:
        print("📈 Standard model performs better in this case.")

    print("\n💾 Results saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Predictions: {predictions_path}")

    return summary_df, predictions_df


def main():
    """Main execution function."""
    try:
        # Check if data directory exists
        if not os.path.exists("data/nox"):
            print("❌ Error: data/nox directory not found.")
            print("Please ensure your NOx data is in the data/nox/ directory.")
            return

        print("🚀 Starting comprehensive MoE vs Standard model comparison...")
        print("This will:")
        print("1. Pre-train both models using masked language modeling")
        print("2. Fine-tune both models on the regression task")
        print("3. Evaluate and compare performance metrics")
        print("4. Save results using polars to outputs/ directory")

        # Train and evaluate both models
        results, standard_model, moe_model = train_and_evaluate_both_models()

        # Save comparison results
        summary_df, predictions_df = save_comparison_results(results)

        print("\n✅ Comparison complete!")
        print("Check the outputs/ directory for detailed results.")
        print("Check TensorBoard logs in runs/ for training metrics.")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
