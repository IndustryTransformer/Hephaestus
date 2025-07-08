"""
Comprehensive three-way comparison: Standard vs Neural Feature Engineering vs Simple models.

This script compares:
1. Standard - Full transformer model 
2. Neural Feature Engineering - Enhanced with automatic feature discovery
3. Simple - Lightweight model that was performing well

Plus tests a simplified NFE version with reduced complexity.
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


def load_and_preprocess_real_data():
    """Load and preprocess the real turbine NOx data."""
    print("📊 Loading real turbine NOx data...")

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

    print(f"Dataset shape: {df.shape}")
    print(f"Target (NOx) stats: Mean={df['target'].mean():.2f}, Std={df['target'].std():.2f}")

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    numeric_cols_without_target = [col for col in numeric_cols if col != "target"]
    categorical_cols = df.select_dtypes(include=["object"]).columns
    categorical_cols = [col for col in categorical_cols if col != "filename"]

    print(f"Numeric features ({len(numeric_cols_without_target)}): {list(numeric_cols_without_target)}")

    # Scale numeric features
    if numeric_cols_without_target:
        scaler = StandardScaler()
        df[numeric_cols_without_target] = scaler.fit_transform(
            df[numeric_cols_without_target]
        )

    # Scale target separately
    target_scaler = StandardScaler()
    df["target"] = target_scaler.fit_transform(df[["target"]]).flatten()

    # Add categorical features if they don't exist
    if not categorical_cols:
        print("Creating load_level categories...")
        # Create load level categories based on the first numeric feature
        if numeric_cols_without_target:
            first_feature = numeric_cols_without_target[0]
            df["load_level"] = pd.cut(
                df[first_feature], bins=3, labels=["low", "medium", "high"]
            )
            categorical_cols = ["load_level"]

    # Convert categorical columns to proper type
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df, target_scaler


def create_output_directory():
    """Create outputs directory if it doesn't exist."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def train_model(config, model_type="standard", max_epochs=30):
    """Train a model with the given configuration."""
    print(f"🔄 Training {model_type} model...")

    # Model parameters - adjusted for faster training
    D_MODEL = 64 if model_type == "simple" else 128
    N_HEADS = 2 if model_type == "simple" else 4
    LR = 0.001
    BATCH_SIZE = 512  # Larger batch size for faster training

    # Load and prepare data
    df, target_scaler = load_and_preprocess_real_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = sr.TabularDS(train_df, config)
    val_dataset = sr.TabularDS(val_df, config)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=sr.training.tabular_collate_fn,
        num_workers=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=sr.training.tabular_collate_fn,
        num_workers=4,
    )

    # Create regressor model
    regressor = sr.TabularRegressor(
        model_config=config,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        lr=LR,
        use_linear_numeric_embedding=True,
        numeric_embedding_type="simple",
    )

    # Count parameters
    total_params = sum(p.numel() for p in regressor.parameters())
    trainable_params = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,}")

    # Training configuration
    logger_name = f"ThreeWay_{model_type}_D{D_MODEL}_H{N_HEADS}"
    logger = TensorBoardLogger("runs", name=logger_name)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, min_delta=0.001, mode="min"
    )
    progress_bar = TQDMProgressBar(leave=False)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stopping, progress_bar],
        log_every_n_steps=20,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    # Train the model
    trainer.fit(
        regressor,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print(f"✅ Training {model_type} completed!")
    return regressor, val_df, target_scaler


def evaluate_model(model, df_test, target_scaler, model_type="standard"):
    """Evaluate model performance and return metrics."""
    print(f"📊 Evaluating {model_type} model...")

    # Create test dataset and dataloader
    model_config = model.model.model_config
    test_dataset = sr.TabularDS(df_test, model_config)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1024,
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
    y_true = torch.cat(y_true, dim=0).cpu().numpy().flatten()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy().flatten()

    # Convert back to original scale
    y_true_unscaled = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate metrics on original scale
    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)

    # Calculate R²
    ss_res = np.sum((y_true_unscaled - y_pred_unscaled) ** 2)
    ss_tot = np.sum((y_true_unscaled - np.mean(y_true_unscaled)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calculate percentage errors
    mape = np.mean(np.abs((y_true_unscaled - y_pred_unscaled) / y_true_unscaled)) * 100

    print(f"{model_type} Model Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return {
        "model_type": model_type,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "parameters": sum(p.numel() for p in model.parameters()),
        "y_true": y_true_unscaled,
        "y_pred": y_pred_unscaled,
    }


def compare_all_models():
    """Train and compare Standard vs Simple vs NFE vs Simplified NFE models."""
    print("🚀 Four-Way Model Comparison on Real Turbine NOx Data")
    print("=" * 80)

    # Load data to create configurations
    df, _ = load_and_preprocess_real_data()

    # Model configurations
    configs = {
        'Simple': sr.SingleRowConfig.generate(
            df,
            target='target',
            use_moe=False,
            use_neural_feature_engineering=False
        ),
        'Standard': sr.SingleRowConfig.generate(
            df,
            target='target',
            use_moe=False,
            use_neural_feature_engineering=False
        ),
        'NFE_Full': sr.SingleRowConfig.generate(
            df,
            target='target',
            use_moe=False,
            use_neural_feature_engineering=True,
            nfe_max_interactions=15,
            nfe_max_ratios=10,
            nfe_polynomial_degree=3,
            nfe_fusion_strategy="learned"
        ),
        'NFE_Simple': sr.SingleRowConfig.generate(
            df,
            target='target',
            use_moe=False,
            use_neural_feature_engineering=True,
            nfe_max_interactions=5,  # Reduced from 15
            nfe_max_ratios=3,        # Reduced from 10
            nfe_polynomial_degree=2,  # Reduced from 3
            nfe_fusion_strategy="concat"  # Simpler fusion
        )
    }

    results = []
    models = {}

    # Train all models
    for model_name, config in configs.items():
        print(f"\n{'='*15} {model_name.upper()} MODEL {'='*15}")

        # Determine model type for parameter selection
        model_type = "simple" if model_name == "Simple" else "standard"
        
        # Train model
        model, test_df, target_scaler = train_model(config, model_type, max_epochs=20)
        models[model_name] = model

        # Evaluate model
        result = evaluate_model(model, test_df, target_scaler, model_name)
        results.append(result)

        # Show Neural Feature Engineering metrics
        if "NFE" in model_name and hasattr(model.model, 'get_neural_feature_engineering_metrics'):
            nfe_metrics = model.model.get_neural_feature_engineering_metrics()
            if nfe_metrics:
                print(f"🧠 NFE Insights:")
                if 'total_engineered_features' in nfe_metrics:
                    total_features = nfe_metrics['total_engineered_features']
                    print(f"  • Total engineered features: {total_features}")
                if 'enabled_components' in nfe_metrics:
                    enabled = nfe_metrics['enabled_components']
                    enabled_list = [k for k, v in enabled.items() if v]
                    print(f"  • Active components: {enabled_list}")

    # Save and display comparison results
    save_comparison_results(results)

    return results, models


def save_comparison_results(results):
    """Save comprehensive comparison results."""
    print(f"\n📁 Saving comparison results...")

    # Create output directory
    output_dir = create_output_directory()

    # Create summary comparison dataframe
    summary_data = []
    for result in results:
        summary_data.append({
            "model_type": result["model_type"],
            "mse": result["mse"],
            "rmse": result["rmse"],
            "mae": result["mae"],
            "r2": result["r2"],
            "mape": result["mape"],
            "parameters": result["parameters"],
        })

    summary_df = pl.DataFrame(summary_data)

    # Find the best model (lowest MSE)
    best_idx = summary_df["mse"].arg_min()
    best_model = summary_df["model_type"][best_idx]
    best_mse = summary_df["mse"][best_idx]

    # Calculate improvements relative to the best model
    improvements_data = []
    for i, result in enumerate(results):
        if result["model_type"] != best_model:
            mse_diff = ((result["mse"] - best_mse) / best_mse) * 100
            improvements_data.append({
                "model_type": f"{result['model_type']}_vs_Best",
                "mse_difference_pct": mse_diff,
                "r2_difference": result["r2"] - summary_df["r2"][best_idx],
                "parameter_ratio": result["parameters"] / summary_df["parameters"][best_idx],
            })

    if improvements_data:
        improvements_df = pl.DataFrame(improvements_data)
        summary_df = pl.concat([summary_df, improvements_df.select(["model_type"])])

    # Save summary
    summary_path = output_dir / "four_way_model_comparison.csv"
    summary_df.write_csv(summary_path)

    # Print results
    print(f"\n🎯 COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 80)
    print(summary_df)

    print(f"\n🏆 PERFORMANCE RANKING (by MSE):")
    sorted_results = sorted(results, key=lambda x: x["mse"])
    for i, result in enumerate(sorted_results, 1):
        params_millions = result["parameters"] / 1_000_000
        print(f"{i}. {result['model_type']:15} | MSE: {result['mse']:6.4f} | R²: {result['r2']:6.4f} | Params: {params_millions:.1f}M")

    print(f"\n📊 KEY INSIGHTS:")
    print(f"Best model: {best_model} (MSE: {best_mse:.4f})")
    
    # Parameter efficiency analysis
    simple_result = next((r for r in results if r["model_type"] == "Simple"), None)
    if simple_result:
        print(f"Simple model efficiency: {simple_result['r2']:.4f} R² with {simple_result['parameters']/1000:.0f}K params")

    print(f"\n💾 Results saved to: {summary_path}")

    return summary_df


def main():
    """Main execution function."""
    try:
        # Check if data directory exists
        if not os.path.exists("data/nox"):
            print("❌ Error: data/nox directory not found.")
            print("Please ensure your NOx data is in the data/nox/ directory.")
            return

        print("🚀 Starting Four-Way Model Comparison")
        print("Models: Simple | Standard | NFE_Full | NFE_Simple")

        # Run comparison
        results, models = compare_all_models()

        print("\n✅ Comparison complete!")
        print("Check the outputs/ directory for detailed results.")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()