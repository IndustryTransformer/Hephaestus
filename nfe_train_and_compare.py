"""
Training and evaluation script for Neural Feature Engineering vs Standard models.

This script trains both models, evaluates their performance, and provides
detailed comparison metrics for turbine NOx prediction.
"""

import os
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
from pathlib import Path

import hephaestus.single_row_models as sr

# Set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_turbine_data():
    """Create realistic turbine NOx data with known interactions."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic turbine data
    n_samples = 5000  # Larger dataset for better training
    
    # Numeric features
    ambient_temp = torch.randn(n_samples) * 10 + 20  # Temperature (C)
    pressure = torch.randn(n_samples) * 5 + 100      # Pressure (kPa)  
    humidity = torch.rand(n_samples) * 100           # Humidity %
    power_output = torch.randn(n_samples) * 50 + 200 # Power MW
    fuel_flow = torch.randn(n_samples) * 10 + 50     # Fuel flow (kg/s)
    air_flow = torch.randn(n_samples) * 20 + 150     # Air flow (kg/s)
    
    # Create meaningful interactions for NOx prediction
    fuel_efficiency = power_output / fuel_flow
    temp_pressure_interaction = ambient_temp * pressure / 1000
    air_fuel_ratio = air_flow / fuel_flow
    combustion_intensity = power_output * fuel_flow / 1000
    
    # Target NOx with realistic dependencies (NOx increases with certain conditions)
    nox = (
        0.08 * ambient_temp +                    # Higher temp = more NOx
        0.06 * pressure +                        # Higher pressure = more NOx
        0.12 * humidity +                        # Higher humidity = less NOx
        0.10 * fuel_flow +                       # More fuel = more NOx
        0.25 * temp_pressure_interaction +       # Key interaction
        -0.15 * fuel_efficiency +                # Efficient combustion = less NOx
        0.20 * air_fuel_ratio +                  # Lean burn affects NOx
        0.18 * combustion_intensity +            # Intense combustion = more NOx
        torch.randn(n_samples) * 1.5 + 15       # Noise + baseline
    )
    
    # Categorical features
    turbine_type = torch.randint(0, 4, (n_samples,))     # 4 turbine types
    fuel_type = torch.randint(0, 3, (n_samples,))        # 3 fuel types
    maintenance_status = torch.randint(0, 4, (n_samples,)) # 4 maintenance levels
    load_condition = torch.randint(0, 3, (n_samples,))   # 3 load conditions
    
    # Create categorical mappings
    turbine_types = ["GT_Frame_E", "GT_Frame_F", "GT_Aeroderivative", "GT_Industrial"]
    fuel_types = ["Natural_Gas", "Diesel", "Dual_Fuel"]
    maintenance_statuses = ["Excellent", "Good", "Fair", "Poor"]
    load_conditions = ["Base_Load", "Peak_Load", "Cycling"]
    
    # Create DataFrame
    df = pd.DataFrame({
        'ambient_temp': ambient_temp.numpy(),
        'pressure': pressure.numpy(),
        'humidity': humidity.numpy(),
        'power_output': power_output.numpy(),
        'fuel_flow': fuel_flow.numpy(),
        'air_flow': air_flow.numpy(),
        'turbine_type': [turbine_types[turbine_type[i]] for i in range(n_samples)],
        'fuel_type': [fuel_types[fuel_type[i]] for i in range(n_samples)],
        'maintenance_status': [maintenance_statuses[maintenance_status[i]] for i in range(n_samples)],
        'load_condition': [load_conditions[load_condition[i]] for i in range(n_samples)],
        'nox': nox.numpy()
    })
    
    return df


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
    df = create_turbine_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Scale numeric features
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32']).columns
    numeric_cols = [col for col in numeric_cols if col != 'nox']
    
    if numeric_cols:
        scaler = StandardScaler()
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    
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
    logger_name = f"NFE_Comparison_{model_type}_D{D_MODEL}_H{N_HEADS}"
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
    return regressor, val_df, scaler if numeric_cols else None


def evaluate_model(model, df_test, scaler, model_type="standard"):
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
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"{model_type} Model Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    
    return {
        "model_type": model_type,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def compare_models():
    """Train and compare Standard vs Neural Feature Engineering models."""
    print("🚀 Training and Comparing Standard vs Neural Feature Engineering Models")
    print("=" * 80)
    
    # Create sample data for configuration
    df = create_turbine_data()
    
    # Model configurations
    configs = {
        'Standard': sr.SingleRowConfig.generate(
            df, 
            target='nox',
            use_moe=False,
            use_neural_feature_engineering=False
        ),
        'Neural_Feature_Engineering': sr.SingleRowConfig.generate(
            df,
            target='nox',
            use_moe=False,
            use_neural_feature_engineering=True,
            nfe_max_interactions=12,
            nfe_max_ratios=8,
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
        model, test_df, scaler = train_model(config, model_name.lower(), max_epochs=50)
        models[model_name] = model
        
        # Evaluate model
        result = evaluate_model(model, test_df, scaler, model_name)
        results.append(result)
        
        # Show Neural Feature Engineering metrics
        if hasattr(model.model, 'get_neural_feature_engineering_metrics'):
            nfe_metrics = model.model.get_neural_feature_engineering_metrics()
            if nfe_metrics:
                print("\n🧠 Neural Feature Engineering Insights:")
                if 'total_engineered_features' in nfe_metrics:
                    print(f"  • Total engineered features: {nfe_metrics['total_engineered_features']}")
                if 'enabled_components' in nfe_metrics:
                    enabled = nfe_metrics['enabled_components']
                    enabled_list = [k for k, v in enabled.items() if v]
                    print(f"  • Active components: {enabled_list}")
    
    # Save and display comparison results
    save_comparison_results(results)
    
    return results, models


def save_comparison_results(results):
    """Save comparison results using polars."""
    print("\n📁 Saving comparison results...")
    
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
        
        # Add improvement row
        improvement_data = {
            "model_type": "NFE_vs_Standard_improvement",
            "mse": mse_improvement,
            "rmse": np.sqrt(nfe_mse) - np.sqrt(std_mse),
            "mae": nfe_metrics["mae"][0] - standard_metrics["mae"][0],
            "r2": r2_improvement,
        }
        
        summary_df = pl.concat([summary_df, pl.DataFrame([improvement_data])])
    
    # Save summary
    summary_path = output_dir / "nfe_model_comparison.csv"
    summary_df.write_csv(summary_path)
    
    # Print results
    print("\n🎯 FINAL COMPARISON RESULTS")
    print("=" * 80)
    print(summary_df)
    
    if len(standard_metrics) > 0 and len(nfe_metrics) > 0:
        print("\n📊 Performance Summary:")
        print(f"MSE improvement: {mse_improvement:+.2f}%")
        print(f"R² improvement: {r2_improvement:+.4f}")
        
        if mse_improvement > 0:
            print("🎉 Neural Feature Engineering outperforms standard model!")
        else:
            print("📈 Standard model performs better in this case.")
    
    print(f"\n💾 Results saved to: {summary_path}")
    
    return summary_df


def main():
    """Main execution function."""
    try:
        print("🚀 Starting Neural Feature Engineering vs Standard Model Training Comparison")
        print("This will train both models and compare their performance on turbine NOx prediction")
        
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