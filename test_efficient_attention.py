#!/usr/bin/env python3
"""
Test script for efficient attention mechanisms.

This script demonstrates how to use the efficient attention implementations
and compares their performance on different sequence lengths.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from hephaestus.timeseries_models.efficient_attention import (
    create_efficient_attention,
    LocalWindowedAttention4D,
    SparseAttention4D,
    FeaturewiseAttention4D,
    ChunkedAttention4D,
    FlashAttention4D,
)
from hephaestus.timeseries_models.multihead_attention import MultiHeadAttention4D


@dataclass
class AttentionTestConfig:
    """Configuration for attention testing."""

    batch_size: int = 4
    n_columns: int = 10
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def test_attention_forward_pass(
    attention_module: nn.Module, config: AttentionTestConfig, seq_len: int
):
    """Test forward pass of an attention module."""
    # Create dummy input
    x = torch.randn(
        config.batch_size,
        config.n_columns,
        seq_len,
        config.d_model,
        device=config.device,
    )

    # Forward pass
    attention_module = attention_module.to(config.device)
    output, _ = attention_module(x, x, x)

    # Verify output shape
    expected_shape = (config.batch_size, config.n_columns, seq_len, config.d_model)
    assert output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output.shape}"
    )

    # Check for NaN values
    assert not torch.isnan(output).any(), "Output contains NaN values"

    return output


def compare_attention_outputs(config: AttentionTestConfig, seq_len: int = 128):
    """Compare outputs of different attention mechanisms."""
    # Create input
    torch.manual_seed(42)
    x = torch.randn(
        config.batch_size,
        config.n_columns,
        seq_len,
        config.d_model,
        device=config.device,
    )

    # Standard attention (reference)
    standard_attn = MultiHeadAttention4D(config.d_model, config.n_heads, config.dropout)
    standard_attn = standard_attn.to(config.device)
    standard_attn.eval()

    with torch.no_grad():
        standard_output, _ = standard_attn(x, x, x)

    # Test different attention types
    attention_types = {
        "local": {"window_size": 64},
        "sparse": {"stride": 4, "local_window": 16},
        "featurewise": {"share_weights": True},
        "chunked": {"chunk_size": 64, "overlap": 16},
    }

    if config.device == "cuda":
        attention_types["flash"] = {}

    results = {}

    for attn_type, kwargs in attention_types.items():
        print(f"\nTesting {attn_type} attention...")

        # Create attention module
        attn_module = create_efficient_attention(
            attn_type, config.d_model, config.n_heads, config.dropout, **kwargs
        )
        attn_module = attn_module.to(config.device)
        attn_module.eval()

        # Forward pass
        with torch.no_grad():
            output, _ = attn_module(x, x, x)

        # Compare with standard attention
        diff = torch.abs(output - standard_output).mean().item()
        max_diff = torch.abs(output - standard_output).max().item()

        results[attn_type] = {
            "mean_diff": diff,
            "max_diff": max_diff,
            "passes": max_diff < 1.0,  # Reasonable threshold for differences
        }

        print(f"  Mean difference: {diff:.6f}")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Test passed: {results[attn_type]['passes']}")

    return results


def benchmark_memory_usage(config: AttentionTestConfig, seq_lengths: list[int]):
    """Benchmark memory usage for different sequence lengths."""
    print("\nMemory Usage Benchmark")
    print("=" * 60)

    attention_configs = {
        "standard": {},
        "local": {"window_size": 256},
        "sparse": {"stride": 4, "local_window": 32},
        "featurewise": {"share_weights": True},
        "chunked": {"chunk_size": 256, "overlap": 32},
    }

    if config.device == "cuda":
        attention_configs["flash"] = {}

    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        for attn_type, kwargs in attention_configs.items():
            try:
                # Clear cache
                if config.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                # Create attention module
                attn_module = create_efficient_attention(
                    attn_type, config.d_model, config.n_heads, config.dropout, **kwargs
                )
                attn_module = attn_module.to(config.device)

                # Create input
                x = torch.randn(
                    config.batch_size,
                    config.n_columns,
                    seq_len,
                    config.d_model,
                    device=config.device,
                )

                # Forward pass
                _ = attn_module(x, x, x)

                # Get memory usage
                if config.device == "cuda":
                    torch.cuda.synchronize()
                    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                else:
                    memory_mb = 0  # CPU memory tracking is complex

                print(f"{attn_type:15} | Memory: {memory_mb:8.2f} MB | Status: ✓")

            except Exception as e:
                print(
                    f"{attn_type:15} | Memory:      N/A MB | Status: ✗ ({str(e)[:30]}...)"
                )


def test_gradient_flow(config: AttentionTestConfig, seq_len: int = 128):
    """Test gradient flow through different attention mechanisms."""
    print("\nGradient Flow Test")
    print("=" * 60)

    attention_types = {
        "standard": {},
        "local": {"window_size": 64},
        "sparse": {"stride": 4, "local_window": 16},
        "featurewise": {"share_weights": True},
        "chunked": {"chunk_size": 64, "overlap": 16},
    }

    if config.device == "cuda":
        attention_types["flash"] = {}

    for attn_type, kwargs in attention_types.items():
        print(f"\nTesting {attn_type} attention gradient flow...")

        # Create attention module
        if attn_type == "standard":
            attn_module = MultiHeadAttention4D(
                config.d_model, config.n_heads, config.dropout
            )
        else:
            attn_module = create_efficient_attention(
                attn_type, config.d_model, config.n_heads, config.dropout, **kwargs
            )

        attn_module = attn_module.to(config.device)
        attn_module.train()

        # Create input with requires_grad
        x = torch.randn(
            config.batch_size,
            config.n_columns,
            seq_len,
            config.d_model,
            device=config.device,
            requires_grad=True,
        )

        # Forward pass
        output, _ = attn_module(x, x, x)

        # Create dummy loss
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check gradients
        grad_norm = x.grad.norm().item()
        has_nan = torch.isnan(x.grad).any().item()

        print(f"  Gradient norm: {grad_norm:.6f}")
        print(f"  Contains NaN: {has_nan}")
        print(f"  Test passed: {not has_nan and grad_norm > 0}")


def main():
    """Run all tests."""
    config = AttentionTestConfig()

    print("Efficient Attention Test Suite")
    print("=" * 60)
    print("Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Number of columns: {config.n_columns}")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Number of heads: {config.n_heads}")

    # Test 1: Basic forward pass
    print("\n\n1. Testing Forward Pass")
    print("=" * 60)

    test_seq_len = 256
    attention_modules = {
        "Standard": MultiHeadAttention4D(
            config.d_model, config.n_heads, config.dropout
        ),
        "Local Windowed": LocalWindowedAttention4D(
            config.d_model, config.n_heads, window_size=64
        ),
        "Sparse": SparseAttention4D(
            config.d_model, config.n_heads, stride=4, local_window=16
        ),
        "Feature-wise": FeaturewiseAttention4D(config.d_model, config.n_heads),
        "Chunked": ChunkedAttention4D(config.d_model, config.n_heads, chunk_size=64),
    }

    if config.device == "cuda":
        attention_modules["Flash"] = FlashAttention4D(config.d_model, config.n_heads)

    for name, module in attention_modules.items():
        try:
            test_attention_forward_pass(module, config, test_seq_len)
            print(f"{name:20} | Forward pass: ✓")
        except Exception as e:
            print(f"{name:20} | Forward pass: ✗ ({str(e)})")

    # Test 2: Compare outputs
    print("\n\n2. Comparing Outputs with Standard Attention")
    print("=" * 60)
    compare_attention_outputs(config, seq_len=128)

    # Test 3: Memory usage
    print("\n\n3. Memory Usage Benchmark")
    benchmark_memory_usage(config, seq_lengths=[128, 256, 512, 1024])

    # Test 4: Gradient flow
    print("\n\n4. Gradient Flow Test")
    test_gradient_flow(config, seq_len=128)

    print("\n\nAll tests completed!")


if __name__ == "__main__":
    main()
