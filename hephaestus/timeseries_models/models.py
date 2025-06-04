# %%
import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

try:
    import pytorch_lightning as L
except ImportError:
    # Create a dummy base class if PyTorch Lightning is not available
    class L:
        class LightningModule(nn.Module):
            def __init__(self):
                super().__init__()

            def save_hyperparameters(self):
                pass

            def log(self, *args, **kwargs):
                pass


from hephaestus.timeseries_models.model_data_classes import (
    NumericCategoricalData,
    ProcessedEmbeddings,
    TimeSeriesConfig,
)
from hephaestus.timeseries_models.multihead_attention import MultiHeadAttention4D


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward neural network module.

    Args:
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feed-forward layer.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.dense1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(d_model, d_model)

    def forward(self, x, deterministic: bool = False):
        """Forward pass of the feed-forward network."""
        # Ensure x is float32 to prevent dtype issues
        x = x.to(torch.float32)

        # Save original shape
        orig_shape = x.shape

        # Reshape for linear layer
        x = x.view(-1, self.d_model)

        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x) if not deterministic else x
        x = self.dense2(x)
        x = self.dropout(x) if not deterministic else x

        # Reshape back to original shape
        x = x.view(*orig_shape)

        return x


class TransformerBlock(nn.Module):
    """
    Transformer block module with configurable efficient attention.

    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feed-forward layer.
        dropout_rate (float): Dropout rate.
        attention_type (str): Type of attention mechanism to use.
        attention_kwargs (dict): Additional arguments for the attention mechanism.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout_rate: float,
        attention_type: Literal[
            "standard", "local", "sparse", "featurewise", "chunked", "flash"
        ] = "standard",
        **attention_kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention_type = attention_type

        # Choose attention mechanism based on type
        if attention_type != "standard":
            try:
                from hephaestus.timeseries_models.efficient_attention import (
                    create_efficient_attention,
                )

                self.multi_head_attention = create_efficient_attention(
                    attention_type=attention_type,
                    embed_dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    **attention_kwargs,
                )
            except ImportError:
                # Fallback to standard attention if efficient attention is not available
                self.multi_head_attention = MultiHeadAttention4D(
                    embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate
                )
        else:
            # Use standard 4D MultiHeadAttention
            self.multi_head_attention = MultiHeadAttention4D(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate
            )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward_network = FeedForwardNetwork(
            d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Initialize weights with consistent dtype
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
        mask: torch.Tensor = None,
    ):
        """Forward pass of the transformer block."""
        mask_shape = mask.shape if mask is not None else None
        ic("Transformer Block", q.shape, k.shape, v.shape, mask_shape)

        # Ensure consistent data types - convert all to float32
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        if mask is not None:
            mask = mask.to(q.device)

        # Use our custom 4D MultiHeadAttention
        attention_output, _ = self.multi_head_attention(
            query=q, key=k, value=v, mask=mask
        )

        out = q + attention_output

        # Apply layer norm to last dimension - handles 4D tensor
        batch_size, n_columns, seq_len, _ = out.shape
        out = out.view(-1, self.d_model)

        # Ensure LayerNorm operates on float32
        out = out.to(torch.float32)
        out = self.layer_norm1(out)
        out = out.view(batch_size, n_columns, seq_len, self.d_model)

        # Feed Forward Network
        ffn = self.feed_forward_network(out, deterministic=deterministic)
        out = out + ffn

        # Apply layer norm to last dimension
        out = out.view(-1, self.d_model)

        # Ensure LayerNorm operates on float32
        out = out.to(torch.float32)
        out = self.layer_norm2(out)
        out = out.view(batch_size, n_columns, seq_len, self.d_model)

        return out


class ReservoirEmbedding(nn.Module):
    """
    A module for performing reservoir embedding on a given input.

    Args:
        config (TimeSeriesConfig): Configuration for the time series.
        features (int): The number of features in the embedding.
        frozen_index (int, optional): The index of the embedding to freeze. Defaults
        to 0.
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
        features: int,
        frozen_index: int = 0,
    ):
        super().__init__()
        self.config = config
        self.features = features
        self.frozen_index = frozen_index

        # Register token_reservoir_lookup as a buffer so it moves with the model
        self.register_buffer(
            "token_reservoir_lookup", self.config.reservoir_encoded, persistent=True
        )

        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size, embedding_dim=self.features
        )

    def to(self, device):
        """Ensures all tensors move to the specified device"""
        super().to(device)
        # Move buffers to device
        for _name, buffer in self.named_buffers():
            if buffer is not None:
                buffer.to(device)
        return self

    def forward(self, base_indices: torch.Tensor):
        """
        Perform reservoir embedding on the given input.
        """
        device = base_indices.device

        # Ensure indices are on the correct device
        if not isinstance(base_indices, torch.Tensor):
            base_indices = torch.tensor(base_indices, device=device)
        else:
            base_indices = base_indices.to(device)

        # Ensure token_reservoir_lookup is on the correct device
        if self.token_reservoir_lookup.device != device:
            self.token_reservoir_lookup = self.token_reservoir_lookup.to(device)

        # Get reservoir indices for the provided base_indices
        reservoir_indices = self.token_reservoir_lookup[base_indices.long()]

        # Embed the reservoir indices
        return_embed = self.embedding(reservoir_indices)

        # Sum across the embedding dimension
        return_embed = torch.sum(return_embed, dim=-2, dtype=torch.float32)
        return return_embed


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for time series data with configurable efficient attention.
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
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

        self.embedding = ReservoirEmbedding(config=self.config, features=self.d_model)

        # Create transformer blocks with efficient attention
        self.transformer_block_0 = TransformerBlock(
            num_heads=self.n_heads,
            d_model=self.d_model,
            d_ff=64,
            dropout_rate=0.1,
            attention_type=self.attention_type,
            **self.attention_kwargs,
        )

        self.transformer_block_chain = nn.ModuleList(
            [
                TransformerBlock(
                    num_heads=self.n_heads,
                    d_model=self.d_model,
                    d_ff=64,
                    dropout_rate=0.1,
                    attention_type=self.attention_type,
                    **self.attention_kwargs,
                )
                for i in range(1, self.n_layers)
            ]
        )

        self.pos_encoder = PositionalEncoding(max_len=8192, d_pos_encoding=d_model)

    def process_numeric(
        self, numeric_inputs: Optional[torch.Tensor]
    ) -> ProcessedEmbeddings:
        """Processes the numeric inputs for the transformer model."""
        if numeric_inputs is None:
            return ProcessedEmbeddings(value_embeddings=None, column_embeddings=None)

        device = numeric_inputs.device
        batch_size, num_cols, seq_len = numeric_inputs.shape
        base_indices = torch.arange(num_cols, device=device)
        repeated_numeric_indices = base_indices.repeat(batch_size, seq_len, 1).permute(
            0, 2, 1
        )

        # Ensure consistent data type
        numeric_inputs = numeric_inputs.to(torch.float32)

        # More robust NaN handling
        nan_mask = torch.isnan(numeric_inputs).detach()

        # Replace NaN values with zeros
        numeric_inputs = torch.nan_to_num(numeric_inputs, nan=0.0)

        # Process numeric indices
        if not isinstance(self.config.numeric_indices, torch.Tensor):
            numeric_indices = torch.tensor(
                self.config.numeric_indices, device=next(self.parameters()).device
            )
        else:
            numeric_indices = self.config.numeric_indices.to(
                next(self.parameters()).device
            )

        # Convert to long for embedding lookup
        numeric_indices = numeric_indices.long()

        repeated_numeric_indices = numeric_indices.repeat(
            numeric_inputs.shape[2], 1
        ).t()
        numeric_col_embeddings = self.embedding(repeated_numeric_indices)

        # Nan Masking
        numeric_col_embeddings = numeric_col_embeddings.unsqueeze(0).expand(
            numeric_inputs.shape[0], -1, -1, -1
        )

        ic("col_token type", numeric_col_embeddings.dtype)
        numeric_embedding = self.embedding(
            torch.tensor(
                self.config.token_dict[self.config.numeric_token],
                device=next(self.parameters()).device,
                dtype=torch.long,
            )
        )
        ic(numeric_embedding.shape)

        # Multiply numeric values with embedding
        numeric_embedding = numeric_inputs.unsqueeze(-1) * numeric_embedding
        ic(numeric_embedding.shape)

        # Replace NaN values with mask token embedding
        mask_token = torch.tensor(
            self.config.numeric_mask_token,
            device=next(self.parameters()).device,
            dtype=torch.long,
        )
        mask_embedding = self.embedding(mask_token)
        numeric_embedding = torch.where(
            nan_mask.unsqueeze(-1),
            mask_embedding.expand_as(numeric_embedding),
            numeric_embedding,
        )

        # Ensure output is float32
        numeric_embedding = numeric_embedding.to(torch.float32)
        numeric_col_embeddings = numeric_col_embeddings.to(torch.float32)

        ic(numeric_embedding.shape)
        return ProcessedEmbeddings(
            column_embeddings=numeric_col_embeddings,
            value_embeddings=numeric_embedding,
        )

    def process_categorical(
        self, categorical_inputs: Optional[torch.Tensor]
    ) -> ProcessedEmbeddings:
        """Processes the categorical inputs for the transformer model."""
        if categorical_inputs is None:
            return ProcessedEmbeddings(None, None)

        # Ensure consistent data type
        categorical_inputs = categorical_inputs.to(torch.float32)

        # Make sure nans are set to <NAN> token
        nan_mask = torch.isnan(categorical_inputs)
        mask_token = torch.tensor(
            self.config.token_dict["[NUMERIC_MASK]"],
            device=next(self.parameters()).device,
            dtype=torch.long,
        )
        categorical_inputs = torch.where(
            nan_mask, mask_token.expand_as(categorical_inputs), categorical_inputs
        )

        # Convert to long for embedding
        categorical_inputs = categorical_inputs.long()

        # Get categorical embeddings
        categorical_embeddings = self.embedding(categorical_inputs)

        # Process categorical indices
        if not isinstance(self.config.categorical_indices, torch.Tensor):
            categorical_indices = torch.tensor(
                self.config.categorical_indices, device=next(self.parameters()).device
            )
        else:
            categorical_indices = self.config.categorical_indices.to(
                next(self.parameters()).device
            )

        # Convert to long for embedding lookup
        categorical_indices = categorical_indices.long()

        ic(
            "Issue here",
            categorical_inputs.shape,
            "args",
            categorical_inputs.shape[2],
            categorical_indices.shape,
        )

        repeated_categorical_indices = categorical_indices.repeat(
            categorical_inputs.shape[2], 1
        ).t()
        ic(repeated_categorical_indices.shape)

        categorical_col_embeddings = self.embedding(repeated_categorical_indices)
        ic("Extra dim here?", categorical_col_embeddings.shape)

        categorical_col_embeddings = categorical_col_embeddings.unsqueeze(0).expand(
            categorical_inputs.shape[0], -1, -1, -1
        )
        ic(categorical_col_embeddings.shape)

        # Ensure output is float32
        categorical_embeddings = categorical_embeddings.to(torch.float32)
        categorical_col_embeddings = categorical_col_embeddings.to(torch.float32)

        return ProcessedEmbeddings(
            column_embeddings=categorical_col_embeddings,
            value_embeddings=categorical_embeddings,
        )

    def combine_inputs(
        self, numeric: ProcessedEmbeddings, categorical: ProcessedEmbeddings
    ) -> ProcessedEmbeddings:
        """Combines numeric and categorical embeddings."""
        # Handle None inputs
        if numeric is None:
            numeric = ProcessedEmbeddings(None, None)
        if categorical is None:
            categorical = ProcessedEmbeddings(None, None)

        if (
            numeric.value_embeddings is not None
            and categorical.value_embeddings is not None
        ):
            ic(numeric.value_embeddings.shape, categorical.value_embeddings.shape)
            ic(numeric.column_embeddings.shape, categorical.column_embeddings.shape)
            value_embeddings = torch.cat(
                [numeric.value_embeddings, categorical.value_embeddings],
                dim=1,
            )
            column_embeddings = torch.cat(
                [
                    numeric.column_embeddings,
                    categorical.column_embeddings,
                ],
                dim=1,
            )

        elif numeric.value_embeddings is not None:
            value_embeddings = numeric.value_embeddings
            column_embeddings = numeric.column_embeddings
        elif categorical.value_embeddings is not None:
            value_embeddings = categorical.value_embeddings
            column_embeddings = categorical.column_embeddings
        else:
            raise ValueError("No numeric or categorical inputs provided.")
        value_embeddings = self.pos_encoder(value_embeddings)
        column_embeddings = self.pos_encoder(column_embeddings)
        return ProcessedEmbeddings(
            value_embeddings=value_embeddings, column_embeddings=column_embeddings
        )

    def causal_mask(
        self,
        numeric_inputs: Optional[torch.Tensor],
        categorical_inputs: Optional[torch.Tensor],
    ):
        """Generates a causal mask for the inputs."""
        if numeric_inputs is not None and categorical_inputs is not None:
            mask_input = torch.cat([numeric_inputs, categorical_inputs], dim=1)
        elif numeric_inputs is not None:
            mask_input = numeric_inputs
        elif categorical_inputs is not None:
            mask_input = categorical_inputs
        else:
            # If both inputs are None, return None (no masking needed)
            return None

        # Create causal mask (lower triangular)
        seq_len = mask_input.size(2)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(next(self.parameters()).device)

        # Create padding mask (for nan values)
        if torch.is_floating_point(mask_input):
            pad_mask = torch.isnan(mask_input).any(dim=1)
        else:
            # For categorical data
            pad_mask = (mask_input == self.config.token_dict["[PAD]"]).any(dim=1)

        # Combine masks
        # In PyTorch attention, True values are masked out
        combined_mask = mask
        if pad_mask is not None:
            # Expand pad_mask to match dimensions
            expanded_pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
            combined_mask = combined_mask | expanded_pad_mask

        return combined_mask

    def forward(
        self,
        numeric_inputs: Optional[torch.Tensor] = None,
        categorical_inputs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        causal_mask: bool = True,
    ):
        """Forward pass of the transformer model."""
        # Get device from parameters
        device = next(self.parameters()).device

        # Ensure all inputs are on the same device
        if numeric_inputs is not None:
            numeric_inputs = numeric_inputs.to(device, dtype=torch.float32)

        if categorical_inputs is not None:
            categorical_inputs = categorical_inputs.to(device, dtype=torch.float32)

        # Convert inputs to PyTorch tensors if they aren't already
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

        ic(
            combined_inputs.value_embeddings.shape,
            combined_inputs.column_embeddings.shape,
        )

        out = self.transformer_block_0(
            q=combined_inputs.value_embeddings,
            k=combined_inputs.column_embeddings,
            v=combined_inputs.value_embeddings,
            deterministic=deterministic,
            mask=mask,
        )

        for transformer_block_iter in self.transformer_block_chain:
            out = transformer_block_iter(
                q=out,
                k=combined_inputs.column_embeddings,
                v=out,
                deterministic=deterministic,
                mask=mask,
            )

        return out


class TimeSeriesDecoder(nn.Module):
    """
    Decoder module for time series data with configurable efficient attention.
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
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

        self.time_series_transformer = TimeSeriesTransformer(
            config=self.config,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            attention_type=self.attention_type,
            attention_kwargs=self.attention_kwargs,
        )

        self.numeric_linear1 = nn.Linear(
            d_model * self.config.n_columns, self.d_model * 2
        )

        self.numeric_linear2 = nn.Linear(
            d_model * 2, len(self.config.numeric_col_tokens)
        )

        # Enhanced categorical processing network
        # Increased dimensions and added more layers
        categorical_hidden_dim1 = self.d_model * 4  # Increased from d_model * 2
        categorical_hidden_dim2 = self.d_model * 6  # New middle layer

        self.categorical_dense1 = nn.Linear(self.d_model, categorical_hidden_dim1)
        self.categorical_bn1 = nn.BatchNorm1d(categorical_hidden_dim1)
        self.categorical_dropout1 = nn.Dropout(0.2)

        self.categorical_dense2 = nn.Linear(
            categorical_hidden_dim1, categorical_hidden_dim2
        )
        self.categorical_bn2 = nn.BatchNorm1d(categorical_hidden_dim2)
        self.categorical_dropout2 = nn.Dropout(0.2)

        self.categorical_dense3 = nn.Linear(
            categorical_hidden_dim2, categorical_hidden_dim1
        )
        self.categorical_bn3 = nn.BatchNorm1d(categorical_hidden_dim1)
        self.categorical_dropout3 = nn.Dropout(0.1)

        # Final output layer for token classification
        self.categorical_output = nn.Linear(
            categorical_hidden_dim1, len(self.config.token_decoder_dict.items())
        )

        # Initialize weights to prevent NaN issues
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights to prevent NaN issues"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                # Ensure consistent data type
                module.weight.data = module.weight.data.to(torch.float32)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.float32)

    def forward(
        self,
        numeric_inputs: torch.Tensor,
        categorical_inputs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        causal_mask: bool = True,
    ) -> NumericCategoricalData:
        """Forward pass of the decoder."""
        # Convert inputs to PyTorch tensors if they aren't already
        if not isinstance(numeric_inputs, torch.Tensor):
            numeric_inputs = torch.tensor(
                numeric_inputs,
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )
        else:
            # Ensure numeric_inputs is float32
            numeric_inputs = numeric_inputs.to(torch.float32)

        if categorical_inputs is not None and not isinstance(
            categorical_inputs, torch.Tensor
        ):
            categorical_inputs = torch.tensor(
                categorical_inputs,
                dtype=torch.float32,
                # Changed from torch.long to ensure consistent dtypes
                device=next(self.parameters()).device,
            )
        elif categorical_inputs is not None:
            # Ensure categorical_inputs is float32
            categorical_inputs = categorical_inputs.to(torch.float32)

        # Check for NaNs in input data
        if torch.isnan(numeric_inputs).any():
            # Handle NaNs in input more explicitly
            numeric_inputs = torch.nan_to_num(numeric_inputs, nan=0.0)

        out = self.time_series_transformer(
            numeric_inputs=numeric_inputs,
            categorical_inputs=categorical_inputs,
            deterministic=deterministic,
            causal_mask=causal_mask,
        )

        # Check for NaNs after transformer
        if torch.isnan(out).any():
            print("Warning: NaNs detected in transformer output")
            out = torch.nan_to_num(out, nan=0.0)

        # Process numeric output
        numeric_out = out.transpose(1, 2)
        numeric_out = numeric_out.reshape(
            numeric_out.shape[0], numeric_out.shape[1], -1
        )
        numeric_out = self.numeric_linear1(numeric_out)
        numeric_out = F.relu(numeric_out)  # Add activation function
        numeric_out = self.numeric_linear2(numeric_out)
        numeric_out = numeric_out.transpose(1, 2)  # Adjust dimensions

        # Check for NaNs after numeric processing
        if torch.isnan(numeric_out).any():
            print("Warning: NaNs detected in numeric output")
            numeric_out = torch.nan_to_num(numeric_out, nan=0.0)

        # Enhanced categorical processing with deeper network
        if categorical_inputs is not None:
            batch_size, _n_columns, seq_len, d_model = out.shape
            n_cat_columns = categorical_inputs.shape[1]

            # Extract only the categorical columns from the output
            # This assumes categorical columns are at the end of the column dimension
            categorical_out = out[:, -n_cat_columns:, :, :]
            ic("Categorical columns extracted", categorical_out.shape)

            # Reshape to [batch_size * n_cat_columns * seq_len, d_model]
            categorical_flat = categorical_out.reshape(-1, d_model)

            # First dense layer with batch normalization
            x = self.categorical_dense1(categorical_flat)
            x = self.categorical_bn1(x)
            x = F.relu(x)
            x = self.categorical_dropout1(x) if not deterministic else x

            # Second dense layer with batch normalization
            x = self.categorical_dense2(x)
            x = self.categorical_bn2(x)
            x = F.relu(x)
            x = self.categorical_dropout2(x) if not deterministic else x

            # Third dense layer with batch normalization
            x = self.categorical_dense3(x)
            x = self.categorical_bn3(x)
            x = F.relu(x)
            x = self.categorical_dropout3(x) if not deterministic else x

            # Output layer
            categorical_logits = self.categorical_output(x)

            # Reshape back to [batch_size, n_cat_columns, seq_len, n_tokens]
            n_tokens = len(self.config.token_decoder_dict.items())
            categorical_out = categorical_logits.reshape(
                batch_size, n_cat_columns, seq_len, n_tokens
            )

            ic("Final categorical shape", categorical_out.shape)
        else:
            categorical_out = None

        # Check for NaNs after categorical processing
        if categorical_out is not None and torch.isnan(categorical_out).any():
            print("Warning: NaNs detected in categorical output")
            categorical_out = torch.nan_to_num(categorical_out, nan=0.0)

        return NumericCategoricalData(
            numeric=numeric_out,
            categorical=categorical_out,
        )


class MaskedTabularPretrainer(L.LightningModule):
    """Pre-training module with masked modeling for tabular data using
    configurable efficient attention.

    This module implements masked language modeling (MLM) for categorical features
    and masked numeric modeling (MNM) for numeric features, with support for
    various efficient attention mechanisms for longer sequences.

    Args:
        config (TimeSeriesConfig): Configuration for the time series model
        d_model (int): Dimension of the model embeddings
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer layers
        learning_rate (float): Learning rate for optimization
        mask_probability (float): Probability of masking each element
        attention_type (str): Type of efficient attention to use
        attention_kwargs (dict): Additional arguments for the attention mechanism
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
        d_model: int = 512,
        n_heads: int = 32,
        n_layers: int = 4,
        learning_rate: float = 1e-4,
        mask_probability: float = 0.2,
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
        self.learning_rate = learning_rate
        self.mask_probability = mask_probability
        self.attention_type = attention_type
        self.attention_kwargs = attention_kwargs or {}

        # Save hyperparameters
        self.save_hyperparameters()

        # Create the transformer with efficient attention
        self.transformer = TimeSeriesTransformer(
            config=config,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            attention_type=attention_type,
            attention_kwargs=self.attention_kwargs,
        )

        # Prediction heads for masked modeling
        self.numeric_head = nn.Linear(d_model, 1)  # Predict numeric values
        self.categorical_head = nn.Linear(
            d_model, len(config.token_decoder_dict)
        )  # Predict categorical tokens

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def mask_inputs(self, numeric, categorical):
        """Apply random masking to inputs for pre-training."""
        if numeric is not None:
            batch_size, n_cols, seq_len = numeric.shape
        else:
            batch_size, n_cols, seq_len = categorical.shape

        # Create mask
        mask = (
            torch.rand(batch_size, n_cols, seq_len, device=self.device)
            < self.mask_probability
        )

        masked_numeric = None
        masked_categorical = None
        numeric_targets = None
        categorical_targets = None

        if numeric is not None:
            masked_numeric = numeric.clone()
            numeric_targets = numeric.clone()
            # Set masked positions to 0
            masked_numeric[mask] = 0.0
            # Set non-masked positions to -100 (ignore in loss)
            numeric_targets[~mask] = -100.0

        if categorical is not None:
            masked_categorical = categorical.clone()
            categorical_targets = categorical.clone()
            # Set masked positions to mask token (assuming 0 is mask token)
            masked_categorical[mask] = 0
            # Set non-masked positions to -100 (ignore in loss)
            categorical_targets[~mask] = -100

        return masked_numeric, masked_categorical, numeric_targets, categorical_targets

    def forward(
        self,
        input_numeric: torch.Tensor | None = None,
        input_categorical: torch.Tensor | None = None,
        targets_numeric: torch.Tensor | None = None,
        targets_categorical: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        """Forward pass for masked pre-training."""
        # Get transformer outputs
        outputs = self.transformer(
            numeric_inputs=input_numeric,
            categorical_inputs=input_categorical,
            deterministic=deterministic,
            causal_mask=False,  # No causal masking for pre-training
        )

        losses = {}

        # Numeric prediction loss
        if input_numeric is not None and targets_numeric is not None:
            numeric_predictions = self.numeric_head(outputs.value_embeddings)
            # Only compute loss on masked positions
            valid_mask = targets_numeric != -100.0
            if valid_mask.any():
                numeric_loss = self.mse_loss(
                    numeric_predictions[valid_mask],
                    targets_numeric[valid_mask].unsqueeze(-1),
                )
                losses["numeric_loss"] = numeric_loss

        # Categorical prediction loss
        if input_categorical is not None and targets_categorical is not None:
            categorical_predictions = self.categorical_head(outputs.value_embeddings)
            # Reshape for cross entropy loss
            _, _, _, vocab_size = categorical_predictions.shape
            categorical_predictions = categorical_predictions.view(-1, vocab_size)
            targets_flat = targets_categorical.view(-1)

            categorical_loss = self.ce_loss(categorical_predictions, targets_flat)
            losses["categorical_loss"] = categorical_loss

        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses

    def training_step(self, batch, batch_idx):
        """Training step for masked pre-training."""
        numeric, categorical = batch.numeric, batch.categorical

        # Apply masking
        masked_numeric, masked_categorical, numeric_targets, categorical_targets = (
            self.mask_inputs(numeric, categorical)
        )

        # Forward pass
        losses = self(
            input_numeric=masked_numeric,
            input_categorical=masked_categorical,
            targets_numeric=numeric_targets,
            targets_categorical=categorical_targets,
            deterministic=False,
        )

        # Log losses
        for key, value in losses.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, prog_bar=True)

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step for masked pre-training."""
        numeric, categorical = batch.numeric, batch.categorical

        # Apply masking
        masked_numeric, masked_categorical, numeric_targets, categorical_targets = (
            self.mask_inputs(numeric, categorical)
        )

        # Forward pass
        losses = self(
            input_numeric=masked_numeric,
            input_categorical=masked_categorical,
            targets_numeric=numeric_targets,
            targets_categorical=categorical_targets,
            deterministic=True,
        )

        # Log losses
        for key, value in losses.items():
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, prog_bar=True)

        return losses["total_loss"]

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class PositionalEncoding(nn.Module):  # TODO WHY IS THIS NOT USED?
    """
    Positional encoding module.

    Args:
        max_len (int): Maximum length of the input sequences.
        d_pos_encoding (int): Dimensionality of the positional encoding.

    Attributes:
        max_len (int): Maximum length of the input sequences.
        d_pos_encoding (int): Dimensionality of the positional encoding.
    """

    def __init__(self, max_len: int, d_pos_encoding: int):
        super().__init__()
        self.max_len = max_len  # Maximum length of the input sequences
        self.d_pos_encoding = d_pos_encoding  # Dimensionality of the embeddings/inputs

    def forward(self, x):
        """
        Forward pass of the positional encoding. Adds positional encoding to
        the input.

        Args:
            x: Input data. Shape: (batch_size, n_columns, seq_len, d_model)

        Returns:
            Output with positional encoding added. Shape:
                (batch_size, n_columns, seq_len, d_model)
        """
        n_epochs, n_columns, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} is larger than the "
                f"maximum length {self.max_len}"
            )

        # Calculate positional encoding
        position = torch.arange(
            self.max_len, dtype=torch.float, device=x.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_pos_encoding, 2, dtype=torch.float, device=x.device)
            * (-math.log(10000.0) / self.d_pos_encoding)
        )

        pe = torch.zeros((self.max_len, self.d_pos_encoding), device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Slice to required sequence length
        pe = pe[:seq_len, :]

        # Reshape for broadcasting
        pe = pe.unsqueeze(0).unsqueeze(3)  # Shape: (1, seq_len, d_pos_encoding, 1)

        # Repeat to match batch and columns dimensions
        pe = pe.repeat(
            n_epochs, 1, 1, n_columns
        )  # Shape: (n_epochs, seq_len, d_pos_encoding, n_columns)

        # Permute to match input dimensions
        pe = pe.permute(
            0, 3, 1, 2
        )  # Shape: (n_epochs, n_columns, seq_len, d_pos_encoding)

        # Add to input
        result = x + pe

        return result
