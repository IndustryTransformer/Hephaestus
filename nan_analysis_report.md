# NaN Loss Analysis: pretrain_only_efficient.py vs planets.py

## Executive Summary

The `pretrain_only_efficient.py` script produces NaN losses during training while `planets.py` trains successfully. This analysis identifies the key architectural and data handling differences that cause this divergence.

## Key Differences

### 1. Model Architecture

**pretrain_only_efficient.py:**
- Uses `MaskedTabularPretrainer` - a masked language model approach
- Implements reconstruction loss on ALL positions (both masked and unmasked)
- Very low mask probability (0.00001) means almost no masking occurs
- Uses `EncoderDecoderDataset` and `encoder_decoder_collate_fn`

**planets.py:**
- Uses `TabularDecoder` - standard next-token prediction
- Traditional autoregressive training approach
- Uses `TimeSeriesDS` and `tabular_collate_fn`

### 2. Loss Calculation Strategy

**pretrain_only_efficient.py (lines 200-220):**
```python
# Calculate numeric loss on ALL positions (reconstruction task)
numeric_transposed = inputs.numeric.permute(0, 2, 1)
numeric_loss = self.numeric_loss_fn(numeric_predictions, numeric_transposed)
```

**planets.py:**
- Standard next-token prediction loss
- Built-in causal masking prevents information leakage
- More stable gradient flow

### 3. Data Preprocessing

**pretrain_only_efficient.py:**
- Uses StandardScaler normalization
- Processes 2M rows with sequence length 1024
- Complex time feature engineering (sin/cos transforms)
- Multiple industrial sensor readings

**planets.py:**
- No data scaling/normalization
- Clean physics simulation data
- Simple planetary position coordinates
- Natural data distribution

### 4. Training Configuration

**pretrain_only_efficient.py:**
- Batch size: 14 (small due to long sequences)
- Sequence length: 1024 (very long)
- Learning rate: 5e-5
- Model dimension: 128
- Flash attention for efficiency

**planets.py:**
- Batch size: 32 (larger)
- Natural sequence lengths (varies)
- Default learning rate
- Model dimension: 512
- Flash attention

## Root Cause Analysis

### Primary Issue: Masked Reconstruction Task Design

The core problem lies in the masked reconstruction approach with extremely low mask probability (0.00001):

1. **Nearly No Masking**: With 0.00001 mask probability, virtually no tokens are masked
2. **Identity Reconstruction**: Model tries to reconstruct inputs it can already see
3. **Gradient Instability**: MSE loss on identity reconstruction creates unstable gradients
4. **Information Leakage**: Model has access to targets during "reconstruction"

### Secondary Issues

1. **Data Scale Mismatch**: Industrial sensor data may have extreme values despite scaling
2. **Sequence Length**: 1024-length sequences increase computational complexity
3. **Loss Function**: MSE loss on high-dimensional outputs can explode with bad predictions

## Technical Deep Dive

### Why planets.py Works

1. **Causal Masking**: Proper autoregressive training prevents information leakage
2. **Natural Data**: Physics simulation data has well-behaved distributions
3. **Proven Architecture**: Standard transformer decoder approach
4. **Appropriate Scale**: Model dimension (512) matches task complexity

### Why pretrain_only_efficient.py Fails

1. **Reconstruction Paradox**: Asking model to reconstruct what it can see
2. **Gradient Explosion**: MSE loss on near-identity mapping creates unstable gradients
3. **Scale Issues**: Industrial data may contain edge cases despite normalization
4. **Architecture Mismatch**: Masked reconstruction better suited for higher mask ratios

## Recommendations

### Immediate Fixes

1. **Increase Mask Probability**: Set to 0.15-0.3 for meaningful reconstruction task
2. **Add Gradient Clipping**: Implement stronger gradient clipping (current: 1.0)
3. **Reduce Sequence Length**: Start with shorter sequences (256-512)
4. **Learning Rate**: Reduce to 1e-5 or 2e-5

### Architectural Improvements

1. **Switch to Autoregressive**: Use `TabularDecoder` like planets.py for stability
2. **Proper Masking Strategy**: If keeping masked approach, mask significant portions
3. **Loss Function**: Consider using Huber loss or other robust alternatives to MSE
4. **Warmup Schedule**: Implement learning rate warmup for masked training

### Data Handling

1. **Robust Scaling**: Consider clipping extreme values before scaling
2. **NaN Strategy**: Leverage the model's built-in NaN handling via mask tokens
3. **Batch Size**: Increase if possible to stabilize training

## How to Fix Masked Attention Implementation

### Current Issues with Masked Attention

The current implementation in `MaskedTabularPretrainer` has several fundamental problems:

1. **No Actual Masking**: With 0.00001 mask probability, almost no tokens are masked
2. **Identity Reconstruction**: Model sees inputs and tries to reconstruct the same values
3. **Loss on ALL Positions**: Computing loss on unmasked positions creates information leakage
4. **No Attention Masking**: The attention mechanism doesn't respect masked positions

### Proper Masked Language Model Implementation

#### 1. Implement True Token Masking

```python
def create_masks(self, inputs, mask_probability=0.15):
    """Create proper BERT-style masks for reconstruction"""
    batch_size, n_features, seq_len = inputs.shape
    
    # Create random mask (True = masked)
    mask = torch.rand(batch_size, n_features, seq_len) < mask_probability
    
    # Don't mask special tokens (e.g., class tokens, padding)
    mask = mask & (inputs != self.config.pad_token_id)
    
    # Create masked inputs
    masked_inputs = inputs.clone()
    masked_inputs[mask] = self.config.mask_token_id
    
    return masked_inputs, mask
```

#### 2. Fix Loss Calculation - Only on Masked Positions

```python
def calculate_reconstruction_loss(self, predictions, targets, mask):
    """Calculate loss ONLY on masked positions"""
    if predictions is None:
        return torch.tensor(0.0, device=self.device)
    
    # Only compute loss where mask is True
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]
    
    if masked_predictions.numel() == 0:
        return torch.tensor(0.0, device=self.device)
    
    return self.loss_fn(masked_predictions, masked_targets)
```

#### 3. Implement Attention Masking

```python
def create_attention_mask(self, input_mask, causal=True):
    """Create attention mask that respects both causal and padding constraints"""
    batch_size, seq_len = input_mask.shape
    
    # Create causal mask (lower triangular)
    if causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_mask.device)
    else:
        causal_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=input_mask.device)
    
    # Combine with padding mask
    # input_mask: True for valid positions, False for padding
    # attention expects: False for attend, True for mask
    padding_mask = ~input_mask.unsqueeze(1) | ~input_mask.unsqueeze(2)
    
    # Combine masks
    combined_mask = causal_mask.unsqueeze(0) | padding_mask
    
    return combined_mask
```

#### 4. Proper Training Step Implementation

```python
def training_step(self, batch, batch_idx):
    inputs, _ = batch
    
    # Create masks for reconstruction task
    masked_numeric, numeric_mask = self.create_masks(
        inputs.numeric, self.mask_probability
    )
    masked_categorical, categorical_mask = self.create_masks(
        inputs.categorical, self.mask_probability
    )
    
    # Create attention mask
    attention_mask = self.create_attention_mask(
        inputs.numeric.any(dim=1)  # Valid positions
    )
    
    # Forward pass with masked inputs
    numeric_pred, categorical_pred = self.model(
        numeric_inputs=masked_numeric,
        categorical_inputs=masked_categorical,
        attention_mask=attention_mask
    )
    
    # Calculate loss ONLY on masked positions
    numeric_loss = self.calculate_reconstruction_loss(
        numeric_pred, inputs.numeric, numeric_mask
    )
    categorical_loss = self.calculate_reconstruction_loss(
        categorical_pred, inputs.categorical, categorical_mask
    )
    
    total_loss = numeric_loss + categorical_loss
    return total_loss
```

#### 5. Enhanced Masking Strategies

```python
def bert_style_masking(self, inputs, mask_prob=0.15):
    """BERT-style masking: 80% [MASK], 10% random, 10% unchanged"""
    mask = torch.rand_like(inputs.float()) < mask_prob
    
    # 80% of masked tokens become [MASK]
    mask_token_positions = mask & (torch.rand_like(inputs.float()) < 0.8)
    
    # 10% become random tokens
    random_positions = mask & ~mask_token_positions & (torch.rand_like(inputs.float()) < 0.5)
    
    # 10% stay unchanged (but still predicted)
    unchanged_positions = mask & ~mask_token_positions & ~random_positions
    
    masked_inputs = inputs.clone()
    masked_inputs[mask_token_positions] = self.config.mask_token_id
    masked_inputs[random_positions] = torch.randint_like(
        masked_inputs[random_positions], 0, self.config.vocab_size
    )
    
    return masked_inputs, mask
```

#### 6. Progressive Masking Schedule

```python
def get_mask_probability(self, epoch, total_epochs):
    """Progressive masking: start low, increase over training"""
    min_prob = 0.05
    max_prob = 0.25
    
    # Linear schedule
    progress = epoch / total_epochs
    return min_prob + (max_prob - min_prob) * progress
```

### Key Implementation Changes Needed

1. **Replace current masking logic** in `MaskedTabularPretrainer.training_step()`
2. **Modify loss calculation** to only compute on masked positions
3. **Add attention masking** to the transformer forward pass
4. **Implement proper mask token handling** in the embedding layers
5. **Add masking schedule** for progressive difficulty

### Expected Benefits

- **Stable Training**: Only predicting masked tokens eliminates information leakage
- **Meaningful Task**: Model learns to reconstruct missing information
- **Better Representations**: Forces model to understand context and relationships
- **Robust to Missing Data**: Natural handling of NaN values through mask tokens

This approach transforms the broken "identity reconstruction" into a proper masked language modeling task similar to BERT, where the model learns to predict missing tokens based on context.

## Conclusion

The NaN loss issue stems from a fundamental architectural mismatch: asking a model to reconstruct inputs it can already see, with virtually no masking. The planets.py approach succeeds because it uses a proven autoregressive architecture with proper causal masking and well-behaved data.

The solution is either to fix the masked reconstruction approach (higher mask probability, better loss functions) or switch to the proven autoregressive approach used in planets.py.