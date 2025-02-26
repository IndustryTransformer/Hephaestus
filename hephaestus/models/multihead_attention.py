import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention4D(nn.Module):
    """
    Multi-head attention implementation for 4D tensors.
    This implementation preserves the 4D structure throughout the attention computation.

    Args:
        embed_dim (int): The embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability (0.0 means no dropout)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None, need_weights=False):
        """
        Compute multi-head attention for 4D tensors

        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, n_columns, seq_len, embed_dim]
            key (torch.Tensor): Key tensor of shape [batch_size, n_columns, seq_len, embed_dim]
            value (torch.Tensor): Value tensor of shape [batch_size, n_columns, seq_len, embed_dim]
            mask (torch.Tensor, optional): Attention mask
            need_weights (bool): Whether to return attention weights

        Returns:
            tuple: (output, attention_weights if need_weights else None)
        """
        batch_size, n_columns, seq_len, _ = query.shape

        # Linear projections
        q = self.q_proj(query)  # [batch, n_columns, seq_len, embed_dim]
        k = self.k_proj(key)  # [batch, n_columns, seq_len, embed_dim]
        v = self.v_proj(value)  # [batch, n_columns, seq_len, embed_dim]

        # Reshape for multi-head attention
        # [batch, n_columns, seq_len, num_heads, head_dim]
        q = q.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, n_columns, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        # [batch, n_columns, num_heads, seq_len, head_dim]
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        # Compute attention scores with improved numerical stability
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Check for NaNs in attention scores
        if torch.isnan(attn_scores).any():
            print("Warning: NaNs found in attention scores")
            attn_scores = torch.nan_to_num(attn_scores, nan=0.0)

        # Apply mask if provided (handle 4D mask)
        if mask is not None:
            # Expand mask to match attention dimensions
            if mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len, seq_len]
            elif mask.dim() == 4:  # [batch, n_columns, seq_len, seq_len]
                mask = mask.unsqueeze(2)  # [batch, n_columns, 1, seq_len, seq_len]

            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Apply softmax with improved numerical stability
        attn_scores_max, _ = torch.max(attn_scores, dim=-1, keepdim=True)
        attn_scores = attn_scores - attn_scores_max  # for numerical stability
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Check for NaNs after softmax
        if torch.isnan(attn_weights).any():
            print("Warning: NaNs found in attention weights")
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(
            attn_weights, v
        )  # [batch, n_columns, num_heads, seq_len, head_dim]

        # Reshape back to original dimensions
        # [batch, n_columns, seq_len, num_heads, head_dim]
        context = context.transpose(2, 3)

        # Combine heads
        # [batch, n_columns, seq_len, embed_dim]
        context = context.reshape(batch_size, n_columns, seq_len, self.embed_dim)

        # Final projection
        output = self.out_proj(context)

        if need_weights:
            return output, attn_weights
        else:
            return output, None
