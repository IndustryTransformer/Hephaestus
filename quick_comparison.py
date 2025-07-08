"""
Quick comparison script - minimal training for fast results.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hephaestus.single_row_models as sr

# Set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data_quick():
    """Quick data loading."""
    csv_files = glob.glob("data/nox/*.csv")
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/nox/ directory")

    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.lower()
    
    if "co" in df.columns:
        df = df.drop(columns=["co"])
    
    df = df.rename(columns={"nox": "target"})
    
    # Quick preprocessing
    numeric_cols = [col for col in df.select_dtypes(include=[float, int]).columns if col != "target"]
    
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    target_scaler = StandardScaler()
    df["target"] = target_scaler.fit_transform(df[["target"]]).flatten()
    
    # Add simple categorical
    df["load_level"] = pd.cut(df[numeric_cols[0]], bins=3, labels=["low", "medium", "high"])
    df["load_level"] = df["load_level"].astype(str)  # Convert to string first
    
    return df, target_scaler


def quick_train_eval(config, model_name):
    """Quick training and evaluation - just 5 epochs."""
    print(f"⚡ Quick training {model_name}...")
    
    df, target_scaler = load_data_quick()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Small model for speed
    D_MODEL = 32 if "simple" in model_name.lower() else 64
    N_HEADS = 2
    LR = 0.005  # Higher LR for fast convergence
    BATCH_SIZE = 1024  # Large batches
    
    # Create datasets
    train_dataset = sr.TabularDS(train_df, config)
    val_dataset = sr.TabularDS(val_df, config)
    
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                          collate_fn=sr.training.tabular_collate_fn, num_workers=4)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                        collate_fn=sr.training.tabular_collate_fn, num_workers=4)
    
    # Create model
    regressor = sr.TabularRegressor(
        model_config=config,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        lr=LR,
        use_linear_numeric_embedding=True,
        numeric_embedding_type="simple",
    )
    
    params = sum(p.numel() for p in regressor.parameters())
    print(f"  Parameters: {params:,}")
    
    # Quick training
    trainer = L.Trainer(
        max_epochs=5,  # Very few epochs
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, min_delta=0.01)]
    )
    
    trainer.fit(regressor, train_dl, val_dl)
    
    # Quick evaluation
    test_dataset = sr.TabularDS(val_df, config)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=1024, 
                                         collate_fn=sr.training.tabular_collate_fn)
    
    y_true, y_pred = [], []
    regressor.eval()
    with torch.no_grad():
        for batch in test_dl:
            y_true.append(batch.target)
            y_pred.append(regressor.predict_step(batch))
    
    y_true = torch.cat(y_true).cpu().numpy().flatten()
    y_pred = torch.cat(y_pred).cpu().numpy().flatten()
    
    # Convert to original scale
    y_true_orig = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Metrics
    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = 1 - np.sum((y_true_orig - y_pred_orig)**2) / np.sum((y_true_orig - np.mean(y_true_orig))**2)
    
    print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # NFE insights
    nfe_features = 0
    if hasattr(regressor.model, 'get_neural_feature_engineering_metrics'):
        nfe_metrics = regressor.model.get_neural_feature_engineering_metrics()
        if nfe_metrics and 'total_engineered_features' in nfe_metrics:
            nfe_features = nfe_metrics['total_engineered_features']
            print(f"  NFE Features: {nfe_features}")
    
    return {
        "model": model_name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "params": params,
        "nfe_features": nfe_features
    }


def main():
    """Quick comparison of all models."""
    print("⚡ QUICK MODEL COMPARISON (5 epochs each)")
    print("=" * 50)
    
    # Load sample data for config
    df, _ = load_data_quick()
    
    # Model configs
    configs = {
        "Simple": sr.SingleRowConfig.generate(df, target='target', use_moe=False, use_neural_feature_engineering=False),
        "Standard": sr.SingleRowConfig.generate(df, target='target', use_moe=False, use_neural_feature_engineering=False),
        "NFE_Full": sr.SingleRowConfig.generate(df, target='target', use_moe=False, use_neural_feature_engineering=True,
                                               nfe_max_interactions=15, nfe_max_ratios=10, nfe_polynomial_degree=3),
        "NFE_Light": sr.SingleRowConfig.generate(df, target='target', use_moe=False, use_neural_feature_engineering=True,
                                                nfe_max_interactions=5, nfe_max_ratios=3, nfe_polynomial_degree=2),
    }
    
    results = []
    for name, config in configs.items():
        try:
            result = quick_train_eval(config, name)
            results.append(result)
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
    
    # Results summary
    print(f"\n🎯 QUICK RESULTS SUMMARY")
    print("=" * 50)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mse')
    
    for _, row in results_df.iterrows():
        nfe_info = f" ({row['nfe_features']} NFE features)" if row['nfe_features'] > 0 else ""
        print(f"{row['model']:12} | MSE: {row['mse']:6.4f} | R²: {row['r2']:6.4f} | Params: {row['params']/1000:5.0f}K{nfe_info}")
    
    # Save quick results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "quick_comparison.csv", index=False)
    
    print(f"\n💡 Key Insights:")
    best = results_df.iloc[0]
    print(f"Best model: {best['model']} (MSE: {best['mse']:.4f})")
    
    simple_result = results_df[results_df['model'] == 'Simple']
    if not simple_result.empty:
        simple_r2 = simple_result.iloc[0]['r2']
        simple_params = simple_result.iloc[0]['params']
        print(f"Simple model: R² {simple_r2:.4f} with only {simple_params/1000:.0f}K parameters")
    
    print(f"Results saved to outputs/quick_comparison.csv")


if __name__ == "__main__":
    main()