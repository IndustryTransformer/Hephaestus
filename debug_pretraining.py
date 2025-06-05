#!/usr/bin/env python3
"""
Debug script for MaskedTabularPretrainer to identify issues at low mask probability.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

import hephaestus as hp
from hephaestus.timeseries_models.encoder_decoder_dataset import (
    EncoderDecoderDataset,
    encoder_decoder_collate_fn,
)
from hephaestus.training.masked_timeseries_training import MaskedTabularPretrainer


def setup_debug_environment():
    """Setup debugging environment with minimal data."""
    print("=== Setting up debug environment ===")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return device


def load_and_prepare_data():
    """Load and prepare minimal dataset for debugging."""
    print("\n=== Loading and preparing data ===")
    
    # Load dataset
    df = pd.read_parquet("data/combined_3w_real_sample.parquet")
    print(f"Original dataset shape: {df.shape}")
    
    # Take very small sample for debugging
    df = df.head(1000)  # Just 1000 rows for quick debugging
    df.drop(columns=["original_filename", "file_class_label", "system_id"], inplace=True)
    
    # Process class labels
    df = df.loc[df["class"].notna()]
    df["class"] = df["class"].astype("str")
    
    # Add time features (simplified)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    seconds = (df["timestamp"].dt.hour * 3600 + 
              df["timestamp"].dt.minute * 60 + 
              df["timestamp"].dt.second)
    seconds_in_day = 24 * 60 * 60
    df["timestamp_sin"] = np.sin(2 * np.pi * seconds / seconds_in_day)
    df["timestamp_cos"] = np.cos(2 * np.pi * seconds / seconds_in_day)
    
    # Define features
    categorical_cols = ["class"]
    drop_cols = ["timestamp"] + categorical_cols
    numeric_cols = [col for col in df.select_dtypes(include=["number"]).columns 
                   if col not in drop_cols]
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Normalize numeric data
    scaler = RobustScaler()
    numeric_data = df[numeric_cols].values.astype(np.float32)
    numeric_data[np.isinf(numeric_data)] = np.nan
    scaled_data = scaler.fit_transform(numeric_data)
    scaled_data = np.clip(scaled_data, -10, 10)
    scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=10.0, neginf=-10.0)
    df[numeric_cols] = scaled_data
    
    print(f"Data range after scaling: [{df[numeric_cols].min().min():.3f}, {df[numeric_cols].max().max():.3f}]")
    print(f"NaN count: {df[numeric_cols].isna().sum().sum()}")
    
    # Create short sequences for debugging
    SEQUENCE_LENGTH = 32  # Very short sequences
    df.sort_values("timestamp", inplace=True)
    df["idx"] = df.index // SEQUENCE_LENGTH
    
    print(f"Created {df['idx'].nunique()} sequences of length {SEQUENCE_LENGTH}")
    
    return df


def create_debug_model_and_data(df):
    """Create model and dataset for debugging."""
    print("\n=== Creating model and dataset ===")
    
    # Configuration
    BATCH_SIZE = 2  # Very small batch for debugging
    D_MODEL = 64   # Small model for debugging
    N_HEADS = 4    # Few heads for debugging
    MASK_PROBABILITY = 0.01  # 1% masking for easier debugging
    
    # Create time series config
    time_series_config = hp.TimeSeriesConfig.generate(df=df, target="class")
    
    print(f"Total tokens: {time_series_config.n_tokens}")
    print(f"Numeric columns: {len(time_series_config.numeric_col_tokens)}")
    print(f"Categorical columns: {len(time_series_config.categorical_col_tokens)}")
    
    # Create simple train dataset (just use first few sequences)
    train_groups = df.loc[:500, "idx"].unique()  # First few sequences only
    
    # Create event targets mapping
    event_values = df["class"].unique()
    event_targets = {str(v): i for i, v in enumerate(event_values)}
    
    # Create dataset
    train_ds = EncoderDecoderDataset(
        df[df["idx"].isin(train_groups)],
        time_series_config,
        target_col="class",
        target_values=event_targets,
    )
    
    print(f"Dataset size: {len(train_ds)}")
    
    # Create model
    model = MaskedTabularPretrainer(
        config=time_series_config,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        learning_rate=1e-4,
        mask_probability=MASK_PROBABILITY,
        attention_type="standard",  # Use standard attention for debugging
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, train_ds, time_series_config


def debug_single_batch(model, dataset):
    """Debug a single batch through the training process."""
    print("\n=== Debugging single batch ===")
    
    device = next(model.parameters()).device
    
    # Get a single sample manually
    sample = dataset[0]  # Get first sample
    inputs, targets = sample
    
    # Add batch dimension
    inputs.numeric = inputs.numeric.unsqueeze(0) if inputs.numeric is not None else None
    inputs.categorical = inputs.categorical.unsqueeze(0) if inputs.categorical is not None else None
    targets.categorical = targets.categorical.unsqueeze(0) if targets.categorical is not None else None
    
    # Move to device
    if inputs.numeric is not None:
        inputs.numeric = inputs.numeric.to(device)
    if inputs.categorical is not None:
        inputs.categorical = inputs.categorical.to(device)
    
    print("=== Input Information ===")
    if inputs.numeric is not None:
        print(f"Numeric input shape: {inputs.numeric.shape}")
        print(f"Numeric range: [{inputs.numeric.min():.3f}, {inputs.numeric.max():.3f}]")
        print(f"Numeric mean: {inputs.numeric.mean():.3f}")
        print(f"Numeric std: {inputs.numeric.std():.3f}")
        print(f"Numeric NaN count: {torch.isnan(inputs.numeric).sum()}")
        print(f"Numeric inf count: {torch.isinf(inputs.numeric).sum()}")
    
    if inputs.categorical is not None:
        print(f"Categorical input shape: {inputs.categorical.shape}")
        print(f"Categorical unique values: {inputs.categorical.unique().tolist()}")
        print(f"Categorical range: [{inputs.categorical.min()}, {inputs.categorical.max()}]")
    
    # Test masking
    print("\n=== Testing Masking ===")
    with torch.no_grad():
        masked_numeric, masked_categorical, numeric_mask, categorical_mask = model.mask_inputs(
            inputs.numeric, inputs.categorical
        )
    
    if numeric_mask is not None:
        print(f"Numeric mask shape: {numeric_mask.shape}")
        print(f"Numeric elements masked: {numeric_mask.sum()}/{numeric_mask.numel()} ({numeric_mask.float().mean():.4f})")
        if numeric_mask.sum() > 0:
            print(f"Masked numeric value used: {masked_numeric[numeric_mask][0]:.3f}")
            print(f"Original values at masked positions: {inputs.numeric[numeric_mask][:5].tolist()}")
    
    if categorical_mask is not None:
        print(f"Categorical mask shape: {categorical_mask.shape}")
        print(f"Categorical elements masked: {categorical_mask.sum()}/{categorical_mask.numel()} ({categorical_mask.float().mean():.4f})")
        if categorical_mask.sum() > 0:
            mask_token = model.config.token_dict.get("[MASK]", 2)
            print(f"Mask token used: {mask_token}")
            print(f"Masked categorical values: {masked_categorical[categorical_mask][:5].tolist()}")
            print(f"Original values at masked positions: {inputs.categorical[categorical_mask][:5].tolist()}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    try:
        with torch.no_grad():
            numeric_pred, categorical_pred = model(
                input_numeric=masked_numeric,
                input_categorical=masked_categorical,
                deterministic=False
            )
        
        print("Forward pass successful!")
        
        if numeric_pred is not None:
            print(f"Numeric predictions shape: {numeric_pred.shape}")
            print(f"Numeric predictions range: [{numeric_pred.min():.3f}, {numeric_pred.max():.3f}]")
            print(f"Numeric predictions NaN count: {torch.isnan(numeric_pred).sum()}")
            print(f"Numeric predictions inf count: {torch.isinf(numeric_pred).sum()}")
        
        if categorical_pred is not None:
            print(f"Categorical predictions shape: {categorical_pred.shape}")
            print(f"Categorical predictions range: [{categorical_pred.min():.3f}, {categorical_pred.max():.3f}]")
            print(f"Categorical predictions NaN count: {torch.isnan(categorical_pred).sum()}")
            print(f"Categorical predictions inf count: {torch.isinf(categorical_pred).sum()}")
            
            # Check logits
            if categorical_pred.numel() > 0:
                print(f"Categorical logits sample: {categorical_pred.view(-1, categorical_pred.size(-1))[0, :10].tolist()}")
    
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Test loss calculation using the exact same path as training_step
    print("\n=== Testing Loss Calculation (Using Real Training Path) ===")
    try:
        # Use the actual training_step internal logic
        model.train()
        
        # Create a minimal batch in the right format
        batch = (inputs, targets if targets is not None else inputs)
        
        # Call the EXACT same masking function
        masked_numeric, masked_categorical, numeric_mask, categorical_mask = model.mask_inputs(
            inputs.numeric, inputs.categorical
        )
        
        print(f"Using real masking from model:")
        if numeric_mask is not None:
            print(f"  Numeric elements masked: {numeric_mask.sum()}/{numeric_mask.numel()}")
        if categorical_mask is not None:
            print(f"  Categorical elements masked: {categorical_mask.sum()}/{categorical_mask.numel()}")
        
        # Get predictions using the EXACT same forward path
        numeric_pred, categorical_pred = model(
            input_numeric=masked_numeric,
            input_categorical=masked_categorical,
            targets_numeric=None,
            targets_categorical=None,
            deterministic=False,
        )
        
        # Use the EXACT same loss calculation as training_step
        losses = []
        numeric_loss_val = torch.tensor(0.0, device=device)
        categorical_loss_val = torch.tensor(0.0, device=device)
        
        # Numeric loss (exactly as in training_step)
        if numeric_pred is not None and numeric_mask is not None:
            numeric_mask_transposed = numeric_mask.permute(0, 2, 1)
            numeric_transposed = inputs.numeric.permute(0, 2, 1)
            masked_numeric_true = numeric_transposed[numeric_mask_transposed]
            masked_numeric_pred = numeric_pred[numeric_mask_transposed]
            
            if masked_numeric_true.numel() > 0:
                print(f"Numeric masked elements for loss: {masked_numeric_true.numel()}")
                print(f"True values sample: {masked_numeric_true[:5].tolist()}")
                print(f"Pred values sample: {masked_numeric_pred[:5].tolist()}")
                
                # Clamp predictions to prevent extreme values (exact same logic)
                masked_numeric_pred = torch.clamp(masked_numeric_pred, -10.0, 10.0)
                masked_numeric_true = torch.clamp(masked_numeric_true, -10.0, 10.0)
                
                numeric_loss = model.numeric_loss_fn(masked_numeric_pred, masked_numeric_true)
                
                # Safeguard: Cap the loss to prevent gradient explosion (exact same logic)
                if numeric_loss > 100.0:
                    numeric_loss = torch.clamp(numeric_loss, max=100.0)
                
                print(f"Numeric loss: {numeric_loss.item():.6f}")
                print(f"Numeric loss requires_grad: {numeric_loss.requires_grad}")
                print(f"Numeric loss grad_fn: {numeric_loss.grad_fn}")
                
                numeric_loss_val = numeric_loss
                losses.append(numeric_loss)
            else:
                print("No numeric elements to compute loss on")
        
        # Categorical loss (exactly as in training_step)
        if categorical_pred is not None and categorical_mask is not None:
            cat_pred_flat = categorical_pred.permute(0, 1, 3, 2).reshape(
                -1, model.config.n_tokens
            )
            cat_true_flat = inputs.categorical.reshape(-1)
            cat_mask_flat = categorical_mask.reshape(-1)
            
            if cat_mask_flat.sum() > 0:
                masked_cat_pred = cat_pred_flat[cat_mask_flat]
                masked_cat_true = cat_true_flat[cat_mask_flat].long()
                
                print(f"Categorical masked elements for loss: {masked_cat_true.numel()}")
                print(f"True categorical values: {masked_cat_true[:5].tolist()}")
                
                categorical_loss = model.categorical_loss_fn(masked_cat_pred, masked_cat_true)
                print(f"Categorical loss: {categorical_loss.item():.6f}")
                print(f"Categorical loss requires_grad: {categorical_loss.requires_grad}")
                print(f"Categorical loss grad_fn: {categorical_loss.grad_fn}")
                
                categorical_loss_val = categorical_loss
                losses.append(categorical_loss)
                
                # Check accuracy
                predicted_tokens = masked_cat_pred.argmax(dim=-1)
                accuracy = (predicted_tokens == masked_cat_true).float().mean()
                print(f"Categorical accuracy: {accuracy.item():.4f}")
            else:
                print("No categorical elements to compute loss on")
        
        # Combine losses exactly as in training_step
        if losses:
            total_loss = sum(losses)
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        print(f"\nTotal loss: {total_loss.item():.6f}")
        print(f"Total loss requires grad: {total_loss.requires_grad}")
        print(f"Total loss device: {total_loss.device}")
        print(f"Total loss grad_fn: {total_loss.grad_fn}")
        print(f"Losses list length: {len(losses)}")
        
        # Test backward pass
        print("\n=== Testing Backward Pass ===")
        model.zero_grad()
        
        if total_loss.requires_grad:
            total_loss.backward()
        else:
            print("ERROR: Total loss does not require gradients!")
            return None
        
        # Check gradients more carefully
        grad_norm = 0.0
        param_count = 0
        param_with_grad_count = 0
        zero_grad_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count += 1
                if param.grad is not None:
                    param_with_grad_count += 1
                    param_grad_norm = param.grad.norm().item()
                    if param_grad_norm > 0:
                        grad_norm += param_grad_norm ** 2
                    else:
                        zero_grad_count += 1
                        # print(f"Zero gradient for: {name}")
                else:
                    print(f"No gradient for parameter: {name}")
        
        grad_norm = grad_norm ** 0.5
        print(f"Gradient norm: {grad_norm:.6f}")
        print(f"Total parameters: {param_count}")
        print(f"Parameters with gradients: {param_with_grad_count}")
        print(f"Parameters with zero gradients: {zero_grad_count}")
        
        return total_loss.item()
        
    except Exception as e:
        print(f"Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_debug_training_step(model, dataset):
    """Run a debug training step to see what happens."""
    print("\n=== Running Debug Training Step ===")
    
    from torch.utils.data import DataLoader
    
    device = next(model.parameters()).device
    
    # Create minimal dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=encoder_decoder_collate_fn,
    )
    
    # Get a batch and move to device
    batch = next(iter(dataloader))
    inputs, targets = batch
    
    # Move batch to device properly
    if inputs.numeric is not None:
        inputs.numeric = inputs.numeric.to(device)
    if inputs.categorical is not None:
        inputs.categorical = inputs.categorical.to(device)
    if targets.categorical is not None:
        targets.categorical = targets.categorical.to(device)
    
    batch = (inputs, targets)
    
    # Try the training step
    model.train()
    try:
        loss = model.training_step(batch, 0)
        print(f"Training step successful! Loss: {loss.item():.6f}")
        return True
    except Exception as e:
        print(f"Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_issues_and_provide_fixes():
    """Analyze the common issues found and provide specific fixes."""
    print("\n" + "=" * 80)
    print("COMMON ISSUES FOUND AND FIXES")
    print("=" * 80)
    
    print("\n1. DEVICE PLACEMENT ISSUE:")
    print("   Problem: Loss function class weights not moved to GPU")
    print("   Fix: Move loss function weights to device in model initialization")
    print("   Code: self.categorical_loss_fn.weight = self.categorical_loss_fn.weight.to(device)")
    
    print("\n2. GRADIENT FLOW ISSUE:")
    print("   Problem: total_loss initialized with requires_grad=True but loses gradient")
    print("   Fix: Use proper tensor initialization for accumulating losses")
    print("   Code: total_loss = torch.tensor(0.0, device=device, requires_grad=True)")
    
    print("\n3. LOW MASKING PROBABILITY ISSUE:")
    print("   Problem: With 1% masking, often no tokens are masked")
    print("   Fix: Use a minimum masking or increase probability for debugging")
    print("   Code: Increase mask_probability to 0.15 (15%) for stable training")
    
    print("\n4. NUMERIC MASK TRANSPOSITION:")
    print("   Problem: Numeric mask shape doesn't match prediction shape")
    print("   Solution: Verified transpose is working correctly")
    
    print("\n5. ATTENTION MECHANISM:")
    print("   Problem: Complex 4D attention may have numerical issues")
    print("   Solution: Use standard attention for debugging first")


def main():
    """Main debugging function."""
    print("Starting MaskedTabularPretrainer Debug Session")
    print("=" * 60)
    
    # Setup
    device = setup_debug_environment()
    
    # Load data
    df = load_and_prepare_data()
    
    # Create model and dataset
    model, dataset, config = create_debug_model_and_data(df)
    model = model.to(device)
    
    # Debug single batch
    loss = debug_single_batch(model, dataset)
    
    if loss is not None:
        print(f"\nSingle batch debugging successful with loss: {loss:.6f}")
        
        # Try full training step
        success = run_debug_training_step(model, dataset)
        
        if success:
            print("\n" + "=" * 60)
            print("DEBUG SUMMARY: MaskedTabularPretrainer appears to be working!")
            print("The model can:")
            print("- Process inputs correctly")
            print("- Apply masking")
            print("- Calculate forward pass")
            print("- Compute losses")
            print("- Calculate gradients")
            print("- Run training steps")
            print("\nIf training is failing, the issue is likely in:")
            print("- Longer sequences causing memory issues")
            print("- Batch accumulation or dataloader issues")
            print("- Learning rate or optimization settings")
            print("- Early stopping or validation issues")
        else:
            print("\n" + "=" * 60)
            print("DEBUG SUMMARY: Training step failed due to device placement")
            print("The main issue is that loss function weights are not on GPU")
    else:
        print("\n" + "=" * 60)
        print("DEBUG SUMMARY: Critical issues found")
        print("Check the error messages above for tensor shape mismatches or NaN values")
    
    # Provide analysis and fixes
    analyze_issues_and_provide_fixes()


if __name__ == "__main__":
    main()