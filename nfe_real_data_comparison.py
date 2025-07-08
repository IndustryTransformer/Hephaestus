"""
Training and evaluation script for Neural Feature Engineering vs Standard models using real turbine NOx data.

This script trains both models using your actual turbine data and provides
detailed comparison metrics for turbine NOx prediction.
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
    original_target = df["target"].copy()

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target (NOx) stats:")
    print(f"  Mean: {df['target'].mean():.2f}")
    print(f"  Std: {df['target'].std():.2f}")
    print(f"  Min: {df['target'].min():.2f}")
    print(f"  Max: {df['target'].max():.2f}")

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    numeric_cols_without_target = [col for col in numeric_cols if col != "target"]
    categorical_cols = df.select_dtypes(include=["object"]).columns
    categorical_cols = [col for col in categorical_cols if col != "filename"]

    print(f"Numeric features ({len(numeric_cols_without_target)}): {list(numeric_cols_without_target)}")
    print(f"Categorical features ({len(categorical_cols)}): {list(categorical_cols)}")

    # Scale numeric features
    if numeric_cols_without_target:
        scaler = StandardScaler()
        df[numeric_cols_without_target] = scaler.fit_transform(
            df[numeric_cols_without_target]
        )

    # Scale target separately
    target_scaler = StandardScaler()
    df["target"] = target_scaler.fit_transform(df[["target"]]).flatten()

    # Add categorical features if they exist, otherwise create some
    if not categorical_cols:
        print("No categorical columns found, creating load_level categories...")
        # Create load level categories based on the first numeric feature
        if numeric_cols_without_target:
            first_feature = numeric_cols_without_target[0]
            df["load_level"] = pd.cut(
                df[first_feature], bins=3, labels=["low", "medium", "high"]
            )
            categorical_cols = ["load_level"]
        else:
            # If no numeric features either, create a dummy category
            df["operation_mode"] = "normal"
            categorical_cols = ["operation_mode"]

    # Convert categorical columns to proper type
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    print(f"Final dataset shape: {df.shape}")
    print(f"Final numeric features: {len(numeric_cols_without_target)}")
    print(f"Final categorical features: {len(categorical_cols)}")

    return df, target_scaler, original_target


def create_output_directory():
    """Create outputs directory if it doesn't exist."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def train_model(config, model_type="standard", max_epochs=50):
    """Train a model with the given configuration."""
    print(f"🔄 Training {model_type} model...")

    # Model parameters
    D_MODEL = 128
    N_HEADS = 4
    LR = 0.0005
    BATCH_SIZE = 256

    # Load and prepare data
    df, target_scaler, _ = load_and_preprocess_real_data()
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
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=sr.training.tabular_collate_fn,
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
    print(f"Model parameters: {trainable_params:,} ({total_params:,} total)")

    # Training configuration
    logger_name = f"NFE_RealData_{model_type}_D{D_MODEL}_H{N_HEADS}"
    logger = TensorBoardLogger("runs", name=logger_name)
    model_summary = ModelSummary(max_depth=2)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=8, min_delta=0.001, mode="min"
    )
    progress_bar = TQDMProgressBar(leave=False)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stopping, progress_bar, model_summary],
        log_every_n_steps=10,
        enable_checkpointing=False,  # Disable for this comparison
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
        batch_size=512,
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

    print(f"{model_type} Model Metrics (Original Scale):")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  MAPE: {mape:.2f}%")

    return {
        "model_type": model_type,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "y_true": y_true_unscaled,
        "y_pred": y_pred_unscaled,
    }


def compare_models():
    """Train and compare Standard vs Neural Feature Engineering models."""
    print("🚀 Training and Comparing Models on Real Turbine NOx Data")
    print("=" * 80)

    # Load data to create configurations
    df, _, _ = load_and_preprocess_real_data()

    # Model configurations
    configs = {
        'Standard': sr.SingleRowConfig.generate(
            df,
            target='target',
            use_moe=False,
            use_neural_feature_engineering=False
        ),
        'Neural_Feature_Engineering': sr.SingleRowConfig.generate(
            df,
            target='target',
            use_moe=False,
            use_neural_feature_engineering=True,
            nfe_max_interactions=15,
            nfe_max_ratios=10,
            nfe_polynomial_degree=3,
            nfe_fusion_strategy="learned"
        )
    }

    results = []
    models = {}

    # Train both models
    for model_name, config in configs.items():
        print(f"\n{'='*20} {model_name.upper()} MODEL {'='*20}")

        # Train model
        model, test_df, target_scaler = train_model(config, model_name.lower(), max_epochs=50)
        models[model_name] = model

        # Evaluate model
        result = evaluate_model(model, test_df, target_scaler, model_name)
        results.append(result)

        # Show Neural Feature Engineering metrics
        if hasattr(model.model, 'get_neural_feature_engineering_metrics'):
            nfe_metrics = model.model.get_neural_feature_engineering_metrics()
            if nfe_metrics:
                print(f"\n🧠 Neural Feature Engineering Insights:")
                if 'total_engineered_features' in nfe_metrics:
                    total_features = nfe_metrics['total_engineered_features']
                    print(f"  • Total engineered features: {total_features}")
                if 'enabled_components' in nfe_metrics:
                    enabled = nfe_metrics['enabled_components']
                    enabled_list = [k for k, v in enabled.items() if v]
                    print(f"  • Active components: {enabled_list}")

                # Show feature importance summary if available
                if hasattr(model.model, 'get_feature_engineering_summary'):
                    summary = model.model.get_feature_engineering_summary()
                    if summary:
                        print(f"  • Feature engineering summary available")

    # Save and display comparison results
    save_comparison_results(results)

    return results, models


def save_comparison_results(results):
    """Save comparison results using polars."""
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
        })

    summary_df = pl.DataFrame(summary_data)

    # Calculate improvements
    standard_metrics = summary_df.filter(pl.col("model_type") == "Standard")
    nfe_metrics = summary_df.filter(pl.col("model_type") == "Neural_Feature_Engineering")

    if len(standard_metrics) > 0 and len(nfe_metrics) > 0:
        std_mse = standard_metrics["mse"][0]
        nfe_mse = nfe_metrics["mse"][0]
        mse_improvement = ((std_mse - nfe_mse) / std_mse) * 100

        std_r2 = standard_metrics["r2"][0]
        nfe_r2 = nfe_metrics["r2"][0]
        r2_improvement = nfe_r2 - std_r2

        std_mape = standard_metrics["mape"][0]
        nfe_mape = nfe_metrics["mape"][0]
        mape_improvement = ((std_mape - nfe_mape) / std_mape) * 100

        # Add improvement row
        improvement_data = {
            "model_type": "NFE_vs_Standard_improvement",
            "mse": mse_improvement,
            "rmse": np.sqrt(nfe_mse) - np.sqrt(std_mse),
            "mae": nfe_metrics["mae"][0] - standard_metrics["mae"][0],
            "r2": r2_improvement,
            "mape": mape_improvement,
        }

        summary_df = pl.concat([summary_df, pl.DataFrame([improvement_data])])

    # Save summary
    summary_path = output_dir / "nfe_real_data_comparison.csv"
    summary_df.write_csv(summary_path)

    # Create detailed predictions comparison
    predictions_data = []
    for result in results:
        for i, (true_val, pred_val) in enumerate(
            zip(result["y_true"], result["y_pred"])
        ):
            predictions_data.append({
                "model_type": result["model_type"],
                "sample_id": i,
                "y_true": true_val,
                "y_pred": pred_val,
                "absolute_error": abs(true_val - pred_val),
                "squared_error": (true_val - pred_val) ** 2,
                "percentage_error": abs((true_val - pred_val) / true_val) * 100 if true_val != 0 else 0,
            })

    predictions_df = pl.DataFrame(predictions_data)
    predictions_path = output_dir / "nfe_real_predictions_comparison.csv"
    predictions_df.write_csv(predictions_path)

    # Print results
    print(f"\n🎯 FINAL COMPARISON RESULTS")
    print("=" * 80)
    print(summary_df)

    if len(standard_metrics) > 0 and len(nfe_metrics) > 0:
        print(f"\n📊 Performance Summary:")
        print(f"MSE improvement: {mse_improvement:+.2f}%")
        print(f"R² improvement: {r2_improvement:+.4f}")
        print(f"MAPE improvement: {mape_improvement:+.2f}%")

        if mse_improvement > 0 and r2_improvement > 0:
            print("🎉 Neural Feature Engineering outperforms standard model!")
        elif mse_improvement > 0 or r2_improvement > 0:
            print("📊 Neural Feature Engineering shows mixed improvements.")
        else:
            print("📈 Standard model performs better in this case.")

    print(f"\n💾 Results saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Detailed predictions: {predictions_path}")

    return summary_df


def main():
    """Main execution function."""
    try:
        # Check if data directory exists
        if not os.path.exists("data/nox"):
            print("❌ Error: data/nox directory not found.")
            print("Please ensure your NOx data is in the data/nox/ directory.")
            return

        print("🚀 Starting Neural Feature Engineering vs Standard Model Training")
        print("This will train both models on your real turbine NOx data")

        # Run comparison
        results, models = compare_models()

        print("\n✅ Comparison complete!")
        print("Check the outputs/ directory for detailed results.")
        print("Check TensorBoard logs in runs/ for training metrics:")
        print("  tensorboard --logdir=runs")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()