import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention4D(nn.Module):
    """
    Multi-head attention module for 4D tensors.

    This implementation handles 4D inputs in the form (batch_size, n_columns, seq_len, d_model),
    which is different from the standard 3D implementation (batch_size, seq_len, d_model).

    Args:
        embed_dim (int): Dimensionality of the embedding.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Check if embed_dim is divisible by num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )

        # Define projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights to avoid NaN issues
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, n_columns, seq_len, d_model)
            key: Key tensor of shape (batch_size, n_columns, seq_len, d_model)
            value: Value tensor of shape (batch_size, n_columns, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            tuple: Output tensor and attention weights
        """

        batch_size, n_columns, seq_len, _ = query.shape

        # Reshape to apply linear projections
        query_flat = query.reshape(-1, self.embed_dim)
        key_flat = key.reshape(-1, self.embed_dim)
        value_flat = value.reshape(-1, self.embed_dim)

        # Apply projections
        q = self.q_proj(query_flat)
        k = self.k_proj(key_flat)
        v = self.v_proj(value_flat)

        # Reshape and split heads
        q = q.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.permute(
            0, 3, 1, 2, 4
        )  # (batch_size, num_heads, n_columns, seq_len, head_dim)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            # Ensure mask preserves information about previous values
            # The current implementation might be too aggressive in masking
            if mask.dim() == 2:
                # Expand mask for batch size and heads
                expanded_mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # Expand mask for heads
                expanded_mask = mask.unsqueeze(1).unsqueeze(1)

            # Add logging for mask shape and values in debugging
            # if self.training:
            #     non_masked = (
            #         (~expanded_mask.bool()).sum().item()
            #         if expanded_mask.dtype == torch.bool
            #         else 0
            #     )
            #     # print(f"Attention mask allows {non_masked} connections")

            # Expand mask for broadcasting to attention scores
            # scores shape: [batch_size, num_heads, n_columns, seq_len, seq_len]
            expanded_mask = expanded_mask.expand(-1, -1, n_columns, -1, -1)

            # Apply mask based on type
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(expanded_mask, float("-inf"))
            else:
                scores = scores + expanded_mask

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Reshape and concat heads
        context = context.permute(
            0, 2, 3, 1, 4
        )  # (batch_size, n_columns, seq_len, num_heads, head_dim)
        context = context.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        # Flatten for output projection
        context_flat = context.reshape(-1, self.embed_dim)

        # Apply output projection
        output = self.out_proj(context_flat)
        output = output.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        return output, attn_weights
