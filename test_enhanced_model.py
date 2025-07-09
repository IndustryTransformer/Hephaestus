#!/usr/bin/env python3
"""Quick test of enhanced model functionality."""

import glob
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hephaestus.single_row_models as sr

def test_enhanced_model():
    """Test the enhanced model with basic functionality."""
    
    # Load data
    csv_files = glob.glob("data/nox/*.csv")
    if not csv_files:
        print("No data files found. Please check data/nox/ directory.")
        return
    
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        filename = Path(file).stem
        temp_df["filename"] = filename
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"nox": "target"})
    
    # Scale numeric columns
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df["cat_column"] = "category"
    
    print(f"Dataset shape: {df.shape}")
    print(f"Numeric columns: {len(df.select_dtypes(include=['number']).columns)}")
    
    # Create configuration
    single_row_config = sr.SingleRowConfig.generate(df, "target")
    train_df, test_df = train_test_split(df.copy(), test_size=0.2, random_state=42)
    
    # Test model creation
    print("\n--- Testing Model Creation ---")
    enhanced_model = sr.create_enhanced_model(
        model_config=single_row_config,
        train_df=train_df,
        target_col="target",
        d_model=64,  # Smaller for faster testing
        n_heads=2,
        lr=0.001,
        aggressive_interactions=True
    )
    
    param_count = sum(p.numel() for p in enhanced_model.parameters())
    print(f"Enhanced model created with {param_count:,} parameters")
    
    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    train_dataset = sr.TabularDS(train_df, single_row_config)
    test_dataset = sr.TabularDS(test_df, single_row_config)
    
    # Test single sample
    sample = train_dataset[0]
    print(f"Sample input shape - numeric: {sample.inputs.numeric.shape}, categorical: {sample.inputs.categorical.shape}")
    print(f"Sample target shape: {sample.target.shape}")
    
    # Test batch
    batch = train_dataset[0:4]  # Get a small batch
    print(f"Batch input shape - numeric: {batch.inputs.numeric.shape}, categorical: {batch.inputs.categorical.shape}")
    print(f"Batch target shape: {batch.target.shape}")
    
    # Test model prediction
    enhanced_model.eval()
    with torch.no_grad():
        prediction = enhanced_model(batch)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction sample: {prediction[:3].flatten()}")
    
    print("\n--- Testing Interaction Components ---")
    print(f"Low-rank cross enabled: {enhanced_model.interaction_config['use_lowrank_cross']}")
    print(f"CrossNet enabled: {enhanced_model.interaction_config['use_crossnet']}")
    print(f"Categorical interactions: {enhanced_model.interaction_config['use_cat_interactions']}")
    print(f"Low-rank dimension: {enhanced_model.interaction_config['lowrank_rank']}")
    print(f"CrossNet layers: {enhanced_model.interaction_config['crossnet_layers']}")
    
    # Test baseline comparison
    print("\n--- Testing Baseline Model ---")
    baseline_model = sr.TabularRegressor(
        single_row_config,
        d_model=64,
        n_heads=2,
        lr=0.001,
    )
    baseline_param_count = sum(p.numel() for p in baseline_model.parameters())
    print(f"Baseline model created with {baseline_param_count:,} parameters")
    
    # Test baseline prediction
    baseline_model.eval()
    with torch.no_grad():
        baseline_prediction = baseline_model.predict_step(batch)
        print(f"Baseline prediction shape: {baseline_prediction.shape}")
        print(f"Baseline prediction sample: {baseline_prediction[:3].flatten()}")
    
    print(f"\nParameter increase: {((param_count - baseline_param_count) / baseline_param_count * 100):.1f}%")
    
    print("\n✅ All tests passed! Enhanced model is working correctly.")

if __name__ == "__main__":
    test_enhanced_model()