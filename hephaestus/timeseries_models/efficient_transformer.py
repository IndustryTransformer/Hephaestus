"""
Efficient transformer implementation that supports various attention mechanisms.

This module provides a drop-in replacement for the standard TransformerBlock
that can use different efficient attention implementations.
"""

from typing import Literal, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn

from hephaestus.timeseries_models.efficient_attention import create_efficient_attention
from hephaestus.timeseries_models.model_data_classes import TimeSeriesConfig
from hephaestus.timeseries_models.models import FeedForwardNetwork


class EfficientTransformerBlock(nn.Module):
    """
    Transformer block with configurable efficient attention mechanism.

    Args:
        num_heads: Number of attention heads
        d_model: Dimensionality of the model
        d_ff: Dimensionality of the feed-forward layer
        dropout_rate: Dropout rate
        attention_type: Type of attention mechanism to use
        attention_kwargs: Additional arguments for the attention mechanism
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout_rate: float,
        attention_type: Literal[
            "standard", "local", "sparse", "featurewise", "chunked", "flash"
        ] = "flash",
        **attention_kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention_type = attention_type

        # Create efficient attention mechanism
        self.multi_head_attention = create_efficient_attention(
            attention_type=attention_type,
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            **attention_kwargs,
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward_network = FeedForwardNetwork(
            d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with consistent data type"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight.data, 1.0)
                nn.init.constant_(module.bias.data, 0.0)
                # Ensure LayerNorm parameters are float32
                module.weight.data = module.weight.data.to(torch.float32)
                module.bias.data = module.bias.data.to(torch.float32)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        deterministic: bool = False,
        mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass of the efficient transformer block."""
        # Ensure consistent data types
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        if mask is not None:
            mask = mask.to(q.device)

        # Apply attention
        attention_output, _ = self.multi_head_attention(
            query=q, key=k, value=v, mask=mask
        )

        # Residual connection and layer norm
        out = q + attention_output

        # Apply layer norm
        batch_size, n_columns, seq_len, _ = out.shape
        out = out.view(-1, self.d_model)
        out = out.to(torch.float32)
        out = self.layer_norm1(out)
        out = out.view(batch_size, n_columns, seq_len, self.d_model)

        # Feed forward network
        ffn = self.feed_forward_network(out, deterministic=deterministic)
        out = out + ffn

        # Final layer norm
        out = out.view(-1, self.d_model)
        out = out.to(torch.float32)
        out = self.layer_norm2(out)
        out = out.view(batch_size, n_columns, seq_len, self.d_model)

        return out


class EfficientTimeSeriesTransformer(nn.Module):
    """
    Time series transformer with configurable efficient attention.

    This is a modified version of TimeSeriesTransformer that supports
    various efficient attention mechanisms for longer sequences.
    """

    def __init__(
        self,
        config,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        attention_type: Literal[
            "standard", "local", "sparse", "featurewise", "chunked", "flash"
        ] = "standard",
        attention_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_type = attention_type
        self.attention_kwargs = attention_kwargs or {}

        # Import required components
        from hephaestus.timeseries_models.models import (
            PositionalEncoding,
            ReservoirEmbedding,
        )

        # Embedding layer
        self.embedding = ReservoirEmbedding(config=self.config, features=self.d_model)

        # Create transformer blocks with efficient attention
        self.transformer_blocks = nn.ModuleList(
            [
                EfficientTransformerBlock(
                    num_heads=self.n_heads,
                    d_model=self.d_model,
                    d_ff=64,
                    dropout_rate=0.1,
                    attention_type=self.attention_type,
                    **self.attention_kwargs,
                )
                for _ in range(self.n_layers)
            ]
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(max_len=8192, d_pos_encoding=d_model)

        # Import processing methods from original class
        from hephaestus.timeseries_models.models import TimeSeriesTransformer

        self.process_numeric = TimeSeriesTransformer.process_numeric.__get__(
            self, type(self)
        )
        self.process_categorical = TimeSeriesTransformer.process_categorical.__get__(
            self, type(self)
        )
        self.combine_inputs = TimeSeriesTransformer.combine_inputs.__get__(
            self, type(self)
        )
        self.causal_mask = TimeSeriesTransformer.causal_mask.__get__(self, type(self))

    def forward(
        self,
        numeric_inputs: Optional[torch.Tensor] = None,
        categorical_inputs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        causal_mask: bool = True,
    ):
        """Forward pass of the efficient transformer."""
        # Get device from parameters
        device = next(self.parameters()).device

        # Ensure all inputs are on the same device
        if numeric_inputs is not None:
            numeric_inputs = numeric_inputs.to(device, dtype=torch.float32)

        if categorical_inputs is not None:
            categorical_inputs = categorical_inputs.to(device, dtype=torch.float32)

        # Convert inputs to PyTorch tensors if needed
        if numeric_inputs is not None and not isinstance(numeric_inputs, torch.Tensor):
            numeric_inputs = torch.tensor(
                numeric_inputs,
                dtype=torch.float32,
                device=device,
            )

        if categorical_inputs is not None and not isinstance(
            categorical_inputs, torch.Tensor
        ):
            categorical_inputs = torch.tensor(
                categorical_inputs,
                dtype=torch.float32,
                device=device,
            )

        # Process inputs
        processed_numeric = (
            self.process_numeric(numeric_inputs) if numeric_inputs is not None else None
        )
        processed_categorical = (
            self.process_categorical(categorical_inputs)
            if categorical_inputs is not None
            else None
        )

        combined_inputs = self.combine_inputs(processed_numeric, processed_categorical)

        mask = (
            self.causal_mask(
                numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
            )
            if causal_mask
            else None
        )

        # Pass through transformer blocks
        out = combined_inputs.value_embeddings
        for transformer_block in self.transformer_blocks:
            out = transformer_block(
                q=out,
                k=combined_inputs.column_embeddings,
                v=out,
                deterministic=deterministic,
                mask=mask,
            )

        return out
