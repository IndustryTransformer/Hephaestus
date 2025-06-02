"""
Efficient attention mechanisms for longer context windows.

This module provides various optimized attention implementations that reduce
memory usage and computational complexity for long sequences.
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalWindowedAttention4D(nn.Module):
    """
    Local windowed attention for 4D tensors.

    Each position only attends to a local window of positions, reducing
    complexity from O(n²) to O(n*w) where w is window size.

    Args:
        embed_dim: Dimensionality of the embedding
        num_heads: Number of attention heads
        window_size: Size of the local attention window
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )

        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, query, key, value, mask=None):
        """Forward pass with local windowed attention."""
        batch_size, n_columns, seq_len, _ = query.shape

        # Project inputs
        query_flat = query.reshape(-1, self.embed_dim)
        key_flat = key.reshape(-1, self.embed_dim)
        value_flat = value.reshape(-1, self.embed_dim)

        q = self.q_proj(query_flat)
        k = self.k_proj(key_flat)
        v = self.v_proj(value_flat)

        # Reshape and split heads
        q = q.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.permute(0, 3, 1, 2, 4)  # [batch, heads, cols, seq, dim]
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)

        # Compute attention with sliding window
        context = self._windowed_attention(q, k, v, self.window_size)

        # Reshape and concat heads
        context = context.permute(0, 2, 3, 1, 4)
        context = context.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        # Output projection
        context_flat = context.reshape(-1, self.embed_dim)
        output = self.out_proj(context_flat)
        output = output.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        return output, None  # Return None for attention weights to save memory

    def _windowed_attention(self, q, k, v, window_size):
        """Compute attention with sliding window."""
        batch_size, num_heads, n_columns, seq_len, head_dim = q.shape

        # Create output tensor
        output = torch.zeros_like(v)

        # Process each position with its local window
        for i in range(seq_len):
            # Define window boundaries
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1  # Causal: can only attend to past

            # Extract local keys and values
            local_k = k[:, :, :, start_idx:end_idx, :]
            local_v = v[:, :, :, start_idx:end_idx, :]
            local_q = q[:, :, :, i : i + 1, :]

            # Compute local attention scores
            scores = torch.matmul(local_q, local_k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )

            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            context = torch.matmul(attn_weights, local_v)
            output[:, :, :, i : i + 1, :] = context

        return output


class SparseAttention4D(nn.Module):
    """
    Sparse attention that only attends to specific positions.

    Implements strided sparse attention pattern where each position
    attends to every n-th position, reducing complexity.

    Args:
        embed_dim: Dimensionality of the embedding
        num_heads: Number of attention heads
        stride: Stride for sparse attention pattern
        local_window: Size of local window around each position
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        stride: int = 4,
        local_window: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.stride = stride
        self.local_window = local_window
        self.head_dim = embed_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, query, key, value, mask=None):
        """Forward pass with sparse attention pattern."""
        batch_size, n_columns, seq_len, _ = query.shape

        # Project inputs
        query_flat = query.reshape(-1, self.embed_dim)
        key_flat = key.reshape(-1, self.embed_dim)
        value_flat = value.reshape(-1, self.embed_dim)

        q = self.q_proj(query_flat)
        k = self.k_proj(key_flat)
        v = self.v_proj(value_flat)

        # Reshape and split heads
        q = q.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)

        # Transpose
        q = q.permute(0, 3, 1, 2, 4)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)

        # Create sparse attention mask
        sparse_mask = self._create_sparse_mask(seq_len, device=q.device)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply sparse mask
        scores = scores.masked_fill(
            ~sparse_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device) * float("-inf"),
            diagonal=1,
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, v)

        # Reshape output
        context = context.permute(0, 2, 3, 1, 4)
        context = context.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        # Output projection
        context_flat = context.reshape(-1, self.embed_dim)
        output = self.out_proj(context_flat)
        output = output.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        return output, attn_weights

    def _create_sparse_mask(self, seq_len, device):
        """Create sparse attention mask."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        for i in range(seq_len):
            # Local window
            start = max(0, i - self.local_window)
            mask[i, start : i + 1] = True

            # Strided positions
            for j in range(0, i, self.stride):
                mask[i, j] = True

        return mask


class FeaturewiseAttention4D(nn.Module):
    """
    Feature-wise attention that computes attention separately for each feature.

    Instead of full 4D attention, this computes attention per feature (column),
    reducing memory usage from O(n_cols * seq_len²) to O(seq_len²).

    Args:
        embed_dim: Dimensionality of the embedding
        num_heads: Number of attention heads
        dropout: Dropout rate
        share_weights: Whether to share attention weights across features
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        share_weights: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.share_weights = share_weights

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )

        # Projections - can be shared or separate per feature
        if share_weights:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
        else:
            # We'll create these dynamically based on n_columns
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """Forward pass with feature-wise attention."""
        batch_size, n_columns, seq_len, embed_dim = query.shape

        # Initialize per-feature projections if needed
        if not self.share_weights and self.q_proj is None:
            self._init_per_feature_projections(n_columns, query.device)

        outputs = []

        # Process each feature separately
        for col_idx in range(n_columns):
            # Extract feature
            q_col = query[:, col_idx, :, :]  # [batch, seq_len, embed_dim]
            k_col = key[:, col_idx, :, :]
            v_col = value[:, col_idx, :, :]

            # Project
            if self.share_weights:
                q = self.q_proj(q_col)
                k = self.k_proj(k_col)
                v = self.v_proj(v_col)
            else:
                q = self.q_projs[col_idx](q_col)
                k = self.k_projs[col_idx](k_col)
                v = self.v_projs[col_idx](v_col)

            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )

            # Compute attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device) * float("-inf"),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention
            context = torch.matmul(attn_weights, v)

            # Reshape
            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, self.embed_dim)
            )

            # Output projection
            if self.share_weights:
                output = self.out_proj(context)
            else:
                output = self.out_projs[col_idx](context)

            outputs.append(output.unsqueeze(1))

        # Concatenate all features
        output = torch.cat(outputs, dim=1)

        return output, None

    def _init_per_feature_projections(self, n_columns, device):
        """Initialize per-feature projection layers."""
        self.q_projs = nn.ModuleList(
            [
                nn.Linear(self.embed_dim, self.embed_dim).to(device)
                for _ in range(n_columns)
            ]
        )
        self.k_projs = nn.ModuleList(
            [
                nn.Linear(self.embed_dim, self.embed_dim).to(device)
                for _ in range(n_columns)
            ]
        )
        self.v_projs = nn.ModuleList(
            [
                nn.Linear(self.embed_dim, self.embed_dim).to(device)
                for _ in range(n_columns)
            ]
        )
        self.out_projs = nn.ModuleList(
            [
                nn.Linear(self.embed_dim, self.embed_dim).to(device)
                for _ in range(n_columns)
            ]
        )

        # Initialize weights
        for projs in [self.q_projs, self.k_projs, self.v_projs, self.out_projs]:
            for proj in projs:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, 0.0)


class ChunkedAttention4D(nn.Module):
    """
    Chunked attention that processes sequences in chunks to reduce memory usage.

    This implementation divides the sequence into chunks and computes attention
    within each chunk, with optional overlap for context preservation.

    Args:
        embed_dim: Dimensionality of the embedding
        num_heads: Number of attention heads
        chunk_size: Size of each chunk
        overlap: Number of positions to overlap between chunks
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        chunk_size: int = 256,
        overlap: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.head_dim = embed_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, query, key, value, mask=None):
        """Forward pass with chunked attention."""
        batch_size, n_columns, seq_len, _ = query.shape

        # Project inputs
        query_flat = query.reshape(-1, self.embed_dim)
        key_flat = key.reshape(-1, self.embed_dim)
        value_flat = value.reshape(-1, self.embed_dim)

        q = self.q_proj(query_flat)
        k = self.k_proj(key_flat)
        v = self.v_proj(value_flat)

        # Reshape and split heads
        q = q.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)

        # Transpose
        q = q.permute(0, 3, 1, 2, 4)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)

        # Process in chunks
        output_chunks = []
        stride = self.chunk_size - self.overlap

        for start_idx in range(0, seq_len, stride):
            end_idx = min(start_idx + self.chunk_size, seq_len)

            # Extract chunk
            q_chunk = q[:, :, :, start_idx:end_idx, :]

            # For causal attention, we need all previous keys and values
            k_chunk = k[:, :, :, :end_idx, :]
            v_chunk = v[:, :, :, :end_idx, :]

            # Compute attention for chunk
            chunk_output = self._compute_chunk_attention(
                q_chunk, k_chunk, v_chunk, start_idx
            )

            # Store non-overlapping part
            if start_idx == 0:
                output_chunks.append(chunk_output)
            else:
                # Skip overlap region
                output_chunks.append(chunk_output[:, :, :, self.overlap :, :])

        # Concatenate chunks
        output = torch.cat(output_chunks, dim=3)

        # Reshape output
        output = output.permute(0, 2, 3, 1, 4)
        output = output.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        # Output projection
        output_flat = output.reshape(-1, self.embed_dim)
        output = self.out_proj(output_flat)
        output = output.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        return output, None

    def _compute_chunk_attention(self, q_chunk, k_chunk, v_chunk, offset):
        """Compute attention for a single chunk."""
        chunk_len = q_chunk.size(3)
        context_len = k_chunk.size(3)

        # Compute scores
        scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        # Create causal mask for chunk
        mask = torch.ones(chunk_len, context_len, device=scores.device) * float("-inf")
        for i in range(chunk_len):
            mask[i, : offset + i + 1] = 0

        scores = scores + mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        output = torch.matmul(attn_weights, v_chunk)

        return output


class FlashAttention4D(nn.Module):
    """
    Flash Attention implementation for 4D tensors.

    This uses PyTorch's scaled_dot_product_attention which automatically
    uses Flash Attention when available (requires PyTorch 2.0+).

    Args:
        embed_dim: Dimensionality of the embedding
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = (
            dropout  # Note: dropout is handled by scaled_dot_product_attention
        )

        # Initialize
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, query, key, value, mask=None):
        """Forward pass using Flash Attention."""
        batch_size, n_columns, seq_len, _ = query.shape

        # Project inputs
        query_flat = query.reshape(-1, self.embed_dim)
        key_flat = key.reshape(-1, self.embed_dim)
        value_flat = value.reshape(-1, self.embed_dim)

        q = self.q_proj(query_flat)
        k = self.k_proj(key_flat)
        v = self.v_proj(value_flat)

        # Reshape for Flash Attention
        # Flash Attention expects [batch, seq_len, num_heads, head_dim]
        q = q.reshape(batch_size * n_columns, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size * n_columns, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size * n_columns, seq_len, self.num_heads, self.head_dim)

        # Use PyTorch's scaled_dot_product_attention
        # This will use Flash Attention when available
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            # Apply attention with causal mask
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,  # We'll use is_causal instead
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        # Reshape back
        output = output.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        # Output projection
        output_flat = output.reshape(-1, self.embed_dim)
        output = self.out_proj(output_flat)
        output = output.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        return output, None


def create_efficient_attention(
    attention_type: Literal[
        "standard", "local", "sparse", "featurewise", "chunked", "flash"
    ],
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create different attention mechanisms.

    Args:
        attention_type: Type of attention to create
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        **kwargs: Additional arguments for specific attention types

    Returns:
        Attention module
    """
    if attention_type == "standard":
        from hephaestus.timeseries_models.multihead_attention import (
            MultiHeadAttention4D,
        )

        return MultiHeadAttention4D(embed_dim, num_heads, dropout)

    elif attention_type == "local":
        window_size = kwargs.get("window_size", 256)
        return LocalWindowedAttention4D(embed_dim, num_heads, window_size, dropout)

    elif attention_type == "sparse":
        stride = kwargs.get("stride", 4)
        local_window = kwargs.get("local_window", 32)
        return SparseAttention4D(embed_dim, num_heads, stride, local_window, dropout)

    elif attention_type == "featurewise":
        share_weights = kwargs.get("share_weights", False)
        return FeaturewiseAttention4D(embed_dim, num_heads, dropout, share_weights)

    elif attention_type == "chunked":
        chunk_size = kwargs.get("chunk_size", 256)
        overlap = kwargs.get("overlap", 32)
        return ChunkedAttention4D(embed_dim, num_heads, chunk_size, overlap, dropout)

    elif attention_type == "flash":
        return FlashAttention4D(embed_dim, num_heads, dropout)

    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
