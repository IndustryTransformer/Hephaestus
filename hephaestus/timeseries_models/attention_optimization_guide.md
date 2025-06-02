# Attention Optimization Guide for Longer Context Windows

This guide explains the optimizations available for handling longer sequences (1024+) in the Hephaestus framework.

## Overview of Bottlenecks

The current 4D attention implementation has several bottlenecks:

1. **Memory Complexity**: O(n_columns × seq_len²) memory usage
2. **Computation Complexity**: Full attention matrix computation for all feature pairs
3. **No Sparsity**: Dense attention patterns even when not necessary
4. **Materialization**: Full attention matrices are materialized in memory

## Available Optimizations

### 1. Local Windowed Attention
- **Memory**: O(n_columns × seq_len × window_size)
- **Use Case**: When local context is sufficient (e.g., time series with local patterns)
- **Configuration**:
```python
attention_kwargs = {"window_size": 256}
```

### 2. Sparse Attention
- **Memory**: O(n_columns × seq_len × (local_window + seq_len/stride))
- **Use Case**: When both local and global context are needed
- **Configuration**:
```python
attention_kwargs = {"stride": 4, "local_window": 32}
```

### 3. Feature-wise Attention
- **Memory**: O(seq_len²) - processes each feature independently
- **Use Case**: When cross-feature attention is not critical
- **Configuration**:
```python
attention_kwargs = {"share_weights": True}  # or False for per-feature weights
```

### 4. Chunked Attention
- **Memory**: O(n_columns × chunk_size²)
- **Use Case**: Very long sequences that need full attention
- **Configuration**:
```python
attention_kwargs = {"chunk_size": 256, "overlap": 32}
```

### 5. Flash Attention
- **Memory**: Efficient kernel-fused implementation
- **Use Case**: When GPU with Flash Attention support is available
- **Requirements**: PyTorch 2.0+, CUDA GPU
- **Configuration**: No additional parameters needed

## Integration Examples

### Using with TimeSeriesDecoder

```python
from hephaestus.timeseries_models.efficient_transformer import EfficientTimeSeriesTransformer
from hephaestus.timeseries_models.models import TimeSeriesDecoder

class EfficientTimeSeriesDecoder(TimeSeriesDecoder):
    def __init__(
        self,
        config,
        d_model=64,
        n_heads=4,
        attention_type="local",
        attention_kwargs=None,
    ):
        super().__init__(config, d_model, n_heads)
        
        # Replace the standard transformer with efficient version
        self.time_series_transformer = EfficientTimeSeriesTransformer(
            config=config,
            d_model=d_model,
            n_heads=n_heads,
            attention_type=attention_type,
            attention_kwargs=attention_kwargs or {"window_size": 256}
        )
```

### Using with TabularEncoderDecoder

```python
from hephaestus.timeseries_models.encoder_decoder import TabularEncoderDecoder
from hephaestus.timeseries_models.efficient_transformer import EfficientTimeSeriesTransformer

class EfficientTabularEncoderDecoder(TabularEncoderDecoder):
    def __init__(
        self,
        config,
        d_model=512,
        n_heads=32,
        learning_rate=1e-4,
        classification_values=None,
        pretrained_encoder=None,
        attention_type="sparse",
        attention_kwargs=None,
    ):
        super().__init__(
            config, d_model, n_heads, learning_rate, 
            classification_values, pretrained_encoder
        )
        
        # Replace encoder if not using pretrained
        if pretrained_encoder is None:
            self.encoder = EfficientTimeSeriesTransformer(
                config=config,
                d_model=d_model,
                n_heads=n_heads,
                attention_type=attention_type,
                attention_kwargs=attention_kwargs or {"stride": 4, "local_window": 32}
            )
```

## Choosing the Right Optimization

### Decision Tree:

1. **Sequence Length < 512**: Use standard attention (no optimization needed)

2. **Sequence Length 512-2048**:
   - Local patterns dominant → Local Windowed Attention
   - Need global context → Sparse Attention
   - GPU available → Flash Attention

3. **Sequence Length > 2048**:
   - Feature independence → Feature-wise Attention
   - Need full attention → Chunked Attention
   - Latest GPU → Flash Attention

4. **Memory Constrained**:
   - First choice: Feature-wise Attention
   - Second choice: Local Windowed Attention with small window
   - Third choice: Chunked Attention with small chunks

## Benchmarking Script

```python
from hephaestus.timeseries_models.efficient_transformer import benchmark_attention_mechanisms

# Run benchmarks
results = benchmark_attention_mechanisms(
    seq_lengths=[256, 512, 1024, 2048, 4096],
    n_columns=20,  # Your number of features
    d_model=64,
    n_heads=8,
    batch_size=16,
    device="cuda"
)

# Analyze results
for seq_len, attn_results in results.items():
    print(f"\nSequence Length: {seq_len}")
    best_time = float('inf')
    best_method = None
    
    for method, metrics in attn_results.items():
        if metrics["successful"] and metrics["time_ms"] < best_time:
            best_time = metrics["time_ms"]
            best_method = method
    
    print(f"Best method: {best_method} ({best_time:.2f}ms)")
```

## Memory Estimation

### Standard Attention
```
Memory (MB) ≈ (batch_size × n_heads × n_columns × seq_len²) × 4 / (1024²)
```

### Local Windowed Attention
```
Memory (MB) ≈ (batch_size × n_heads × n_columns × seq_len × window_size) × 4 / (1024²)
```

### Feature-wise Attention
```
Memory (MB) ≈ (batch_size × n_heads × n_columns × seq_len²) × 4 / (1024² × n_columns)
```

## Performance Tips

1. **Start with Flash Attention** if you have a compatible GPU
2. **Profile your specific use case** - the best method depends on your data
3. **Consider hybrid approaches** - use different attention in different layers
4. **Adjust hyperparameters** based on your sequence characteristics:
   - Time series with strong local patterns: smaller window sizes
   - Tabular data with global dependencies: larger windows or sparse attention
5. **Monitor GPU memory** during training to find the optimal configuration

## Example: Anomaly Detection with Long Sequences

```python
# For anomaly detection with sequences of length 2048
encoder = EfficientTimeSeriesTransformer(
    config=config,
    d_model=512,
    n_heads=32,
    attention_type="sparse",  # Good for anomaly patterns
    attention_kwargs={
        "stride": 8,  # Check every 8th position globally
        "local_window": 64  # Detailed local context
    }
)

# Or use Flash Attention for best performance
encoder = EfficientTimeSeriesTransformer(
    config=config,
    d_model=512,
    n_heads=32,
    attention_type="flash"  # Automatic optimization
)
```