#!/usr/bin/env python3
"""Quick test of the improved masked attention implementation"""

import torch
import numpy as np
import pandas as pd

# Test the basic functionality without full training
def test_mask_functions():
    print("=== Testing Mask Functions ===")
    
    # Import our updated class
    from hephaestus.training.masked_timeseries_training import MaskedTabularPretrainer
    import hephaestus as hp
    
    # Create a minimal config for testing
    test_df = pd.DataFrame({
        'idx': list(range(100)),
        'col1': [1, 2, 3, 4, 5] * 20,
        'col2': [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
        'category': ['A', 'B', 'C', 'A', 'B'] * 20
    })
    
    config = hp.TimeSeriesConfig.generate(df=test_df, target='category')
    
    # Create model
    model = MaskedTabularPretrainer(
        config=config,
        d_model=64,
        n_heads=4,
        mask_probability=0.15
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test mask creation
    batch_size, n_features, seq_len = 2, 3, 10
    test_numeric = torch.randn(batch_size, n_features, seq_len)
    test_categorical = torch.randint(0, 10, (batch_size, n_features, seq_len))
    
    # Test numeric masking
    numeric_mask = model.create_masks(test_numeric, mask_probability=0.3)
    masked_numeric, _ = model.bert_style_masking(test_numeric, numeric_mask)
    
    print(f"Numeric mask shape: {numeric_mask.shape}")
    print(f"Masked positions: {numeric_mask.sum().item()}/{numeric_mask.numel()}")
    print(f"NaN values in masked input: {torch.isnan(masked_numeric).sum().item()}")
    
    # Test categorical masking  
    categorical_mask = model.create_masks(test_categorical, mask_probability=0.3)
    masked_categorical, _ = model.bert_style_masking(test_categorical, categorical_mask)
    
    print(f"Categorical mask shape: {categorical_mask.shape}")
    print(f"Masked positions: {categorical_mask.sum().item()}/{categorical_mask.numel()}")
    
    # Test loss calculation
    dummy_predictions = torch.randn(batch_size, seq_len, n_features)
    dummy_targets = test_numeric.permute(0, 2, 1)
    mask_transposed = numeric_mask.permute(0, 2, 1)
    
    loss = model.calculate_reconstruction_loss(dummy_predictions, dummy_targets, mask_transposed)
    print(f"Loss calculation successful: {loss.item():.4f}")
    
    # Test progressive masking
    for epoch in [0, 10, 25, 50]:
        mask_prob = model.get_mask_probability(epoch, 50)
        print(f"Epoch {epoch}: mask probability = {mask_prob:.3f}")
    
    print("=== All tests passed! ===")

if __name__ == "__main__":
    test_mask_functions()