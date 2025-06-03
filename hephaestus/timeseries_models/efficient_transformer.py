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
        ] = "standard",
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


def benchmark_attention_mechanisms(
    seq_lengths: list[int] = [256, 512, 1024, 2048],
    n_columns: int = 10,
    d_model: int = 64,
    n_heads: int = 4,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Benchmark different attention mechanisms for memory usage and speed.

    Args:
        seq_lengths: List of sequence lengths to test
        n_columns: Number of columns (features)
        d_model: Model dimension
        n_heads: Number of attention heads
        batch_size: Batch size
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    import gc
    import time

    results = {}

    attention_configs = {
        "standard": {},
        "local": {"window_size": 256},
        "sparse": {"stride": 4, "local_window": 32},
        "featurewise": {"share_weights": True},
        "chunked": {"chunk_size": 256, "overlap": 32},
    }

    # Add flash attention if available
    if torch.cuda.is_available():
        attention_configs["flash"] = {}

    for seq_len in seq_lengths:
        results[seq_len] = {}

        for attn_type, kwargs in attention_configs.items():
            try:
                # Create attention module
                attention = create_efficient_attention(
                    attn_type, d_model, n_heads, 0.1, **kwargs
                ).to(device)

                # Create dummy input
                x = torch.randn(batch_size, n_columns, seq_len, d_model, device=device)

                # Warmup
                for _ in range(3):
                    _ = attention(x, x, x)

                # Time forward pass
                torch.cuda.synchronize() if device == "cuda" else None
                start_time = time.time()

                for _ in range(10):
                    output, _ = attention(x, x, x)

                torch.cuda.synchronize() if device == "cuda" else None
                end_time = time.time()

                avg_time = (end_time - start_time) / 10

                # Memory usage (approximate)
                if device == "cuda":
                    torch.cuda.synchronize()
                    memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                    torch.cuda.reset_peak_memory_stats(device)
                else:
                    memory_mb = 0  # CPU memory tracking is more complex

                results[seq_len][attn_type] = {
                    "time_ms": avg_time * 1000,
                    "memory_mb": memory_mb,
                    "successful": True,
                }

            except Exception as e:
                results[seq_len][attn_type] = {
                    "time_ms": None,
                    "memory_mb": None,
                    "successful": False,
                    "error": str(e),
                }

            # Clean up
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    return results


class EfficientMaskedTabularPretrainer(L.LightningModule):
    """Pre-training module with masked modeling for tabular data using
    efficient attention.

    This module implements masked language modeling (MLM) for categorical features
    and masked numeric modeling (MNM) for numeric features, using efficient attention
    mechanisms for longer sequences.

    Args:
        config (TimeSeriesConfig): Configuration for the time series model
        d_model (int): Dimension of the model embeddings
        n_heads (int): Number of attention heads
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
        self.learning_rate = learning_rate
        self.mask_probability = mask_probability
        self.attention_type = attention_type
        self.attention_kwargs = attention_kwargs or {}

        # Encoder with efficient attention
        self.encoder = EfficientTimeSeriesTransformer(
            config=self.config,
            d_model=self.d_model,
            n_heads=self.n_heads,
            attention_type=self.attention_type,
            attention_kwargs=self.attention_kwargs,
        )

        # Reconstruction heads with proper initialization
        self.numeric_reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 1),  # Predict single numeric value
        )

        # Initialize weights for numeric reconstruction head
        for module in self.numeric_reconstruction_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.categorical_reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, config.n_tokens),  # Predict token probabilities
        )

        # Initialize weights for categorical reconstruction head
        for module in self.categorical_reconstruction_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Loss functions
        self.numeric_loss_fn = nn.MSELoss()
        self.categorical_loss_fn = nn.CrossEntropyLoss()

        # Save hyperparameters
        self.save_hyperparameters(ignore=["config"])

    def mask_inputs(self, numeric, categorical):
        """Apply masking to inputs."""
        device = numeric.device if numeric is not None else categorical.device

        # Create masks
        if numeric is not None:
            numeric_mask = (
                torch.rand(numeric.shape, device=device) < self.mask_probability
            )
            masked_numeric = numeric.clone()
            # Use a special mask value that's clearly out of normal data range
            # Since data is clipped to [-10, 10], use -15 as mask value
            masked_numeric[numeric_mask] = -15.0
        else:
            masked_numeric = None
            numeric_mask = None

        if categorical is not None:
            categorical_mask = (
                torch.rand(categorical.shape, device=device) < self.mask_probability
            )
            masked_categorical = categorical.clone()
            # Get [MASK] token index from token_dict
            mask_token_idx = self.config.token_dict.get(
                "[MASK]", 2
            )  # Default to 2 if not found
            masked_categorical[categorical_mask] = mask_token_idx
        else:
            masked_categorical = None
            categorical_mask = None

        return masked_numeric, masked_categorical, numeric_mask, categorical_mask

    def forward(
        self,
        input_numeric: torch.Tensor | None = None,
        input_categorical: torch.Tensor | None = None,
        targets_numeric: torch.Tensor | None = None,
        targets_categorical: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        """Forward pass for reconstruction."""
        # Get encoder outputs
        encoder_output = self.encoder(
            numeric_inputs=input_numeric,
            categorical_inputs=input_categorical,
            deterministic=deterministic,
            causal_mask=False,
        )  # [batch, num_features, seq_len, d_model]

        batch_size, _, seq_len, d_model = encoder_output.shape

        # Separate numeric and categorical features for reconstruction
        if input_numeric is not None:
            n_numeric = input_numeric.shape[1]
            numeric_features = encoder_output[
                :, :n_numeric, :, :
            ]  # [batch, n_numeric, seq_len, d_model]
            # Reshape for reconstruction head
            numeric_features = numeric_features.permute(0, 2, 1, 3).reshape(-1, d_model)

            numeric_predictions = self.numeric_reconstruction_head(numeric_features)

            numeric_predictions = numeric_predictions.view(
                batch_size, seq_len, n_numeric
            )
        else:
            numeric_predictions = None
            n_numeric = 0

        # Handle categorical features for reconstruction
        if input_categorical is not None:
            n_categorical = input_categorical.shape[1]
            categorical_features = encoder_output[
                :, n_numeric : n_numeric + n_categorical, :, :
            ]  # [batch, n_categorical, seq_len, d_model]
            # Reshape for reconstruction head
            categorical_features = categorical_features.permute(0, 2, 1, 3).reshape(
                -1, d_model
            )

            categorical_predictions = self.categorical_reconstruction_head(
                categorical_features
            )

            categorical_predictions = categorical_predictions.view(
                batch_size, seq_len, n_categorical, self.config.n_tokens
            )
        else:
            categorical_predictions = None

        return numeric_predictions, categorical_predictions

    def training_step(self, batch, batch_idx):
        """Training step with masked modeling."""
        inputs, _ = batch  # Ignore targets for pre-training
        batch_size = (
            inputs.numeric.shape[0]
            if inputs.numeric is not None
            else inputs.categorical.shape[0]
        )

        # Apply masking
        masked_numeric, masked_categorical, numeric_mask, categorical_mask = (
            self.mask_inputs(inputs.numeric, inputs.categorical)
        )

        # Forward pass
        numeric_predictions, categorical_predictions = self(
            input_numeric=masked_numeric,
            input_categorical=masked_categorical,
            targets_numeric=None,
            targets_categorical=None,
            deterministic=False,
        )

        total_loss = 0.0
        numeric_loss_val = torch.tensor(0.0, device=self.device)
        categorical_loss_val = torch.tensor(0.0, device=self.device)
        categorical_accuracy_val = torch.tensor(0.0, device=self.device)

        # Calculate numeric loss on masked positions
        if numeric_predictions is not None and numeric_mask is not None:
            # Select only masked positions
            # Transpose mask to match prediction shape [batch, seq_len, n_numeric]
            numeric_mask_transposed = numeric_mask.permute(0, 2, 1)
            numeric_transposed = inputs.numeric.permute(0, 2, 1)
            masked_numeric_true = numeric_transposed[numeric_mask_transposed]
            masked_numeric_pred = numeric_predictions[numeric_mask_transposed]

            if masked_numeric_true.numel() > 0:
                # Clip predictions to prevent extreme values
                masked_numeric_pred = torch.clamp(masked_numeric_pred, -10.0, 10.0)
                masked_numeric_true = torch.clamp(masked_numeric_true, -10.0, 10.0)

                numeric_loss = self.numeric_loss_fn(
                    masked_numeric_pred, masked_numeric_true
                )

                # Safeguard: Cap the loss to prevent gradient explosion
                if numeric_loss > 100.0:
                    numeric_loss = torch.clamp(numeric_loss, max=100.0)

                numeric_loss_val = numeric_loss
                total_loss += numeric_loss_val

        # Calculate categorical loss on masked positions
        if categorical_predictions is not None and categorical_mask is not None:
            # Reshape for loss calculation
            cat_pred_flat = categorical_predictions.permute(0, 1, 3, 2).reshape(
                -1, self.config.n_tokens
            )
            cat_true_flat = inputs.categorical.reshape(-1)
            cat_mask_flat = categorical_mask.reshape(-1)

            # Select only masked positions
            if cat_mask_flat.sum() > 0:
                masked_cat_pred = cat_pred_flat[cat_mask_flat]
                masked_cat_true = cat_true_flat[cat_mask_flat].long()

                categorical_loss = self.categorical_loss_fn(
                    masked_cat_pred, masked_cat_true
                )
                categorical_loss_val = categorical_loss
                total_loss += categorical_loss_val

                # Calculate categorical accuracy
                categorical_accuracy = (
                    (masked_cat_pred.argmax(dim=-1) == masked_cat_true).float().mean()
                )
                categorical_accuracy_val = categorical_accuracy

        # Log training metrics at both step and epoch level
        self.log(
            "train/numeric_loss",
            numeric_loss_val,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "train/categorical_loss",
            categorical_loss_val,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "train/categorical_accuracy",
            categorical_accuracy_val,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step with masked modeling."""
        inputs, _ = batch
        batch_size = (
            inputs.numeric.shape[0]
            if inputs.numeric is not None
            else inputs.categorical.shape[0]
        )

        # Apply masking
        masked_numeric, masked_categorical, numeric_mask, categorical_mask = (
            self.mask_inputs(inputs.numeric, inputs.categorical)
        )

        # Forward pass
        numeric_predictions, categorical_predictions = self(
            input_numeric=masked_numeric,
            input_categorical=masked_categorical,
            targets_numeric=None,
            targets_categorical=None,
            deterministic=True,
        )

        total_loss = 0.0
        numeric_loss_val = torch.tensor(0.0, device=self.device)
        categorical_loss_val = torch.tensor(0.0, device=self.device)
        categorical_accuracy_val = torch.tensor(0.0, device=self.device)

        # Calculate numeric loss
        if numeric_predictions is not None and numeric_mask is not None:
            # Transpose mask to match prediction shape [batch, seq_len, n_numeric]
            numeric_mask_transposed = numeric_mask.permute(0, 2, 1)
            numeric_transposed = inputs.numeric.permute(0, 2, 1)
            masked_numeric_true = numeric_transposed[numeric_mask_transposed]
            masked_numeric_pred = numeric_predictions[numeric_mask_transposed]

            if masked_numeric_true.numel() > 0:
                numeric_loss = self.numeric_loss_fn(
                    masked_numeric_pred, masked_numeric_true
                )
                numeric_loss_val = numeric_loss
                total_loss += numeric_loss_val

        # Calculate categorical loss and accuracy
        if categorical_predictions is not None and categorical_mask is not None:
            # Reshape for loss calculation
            cat_pred_flat = categorical_predictions.permute(0, 1, 3, 2).reshape(
                -1, self.config.n_tokens
            )
            cat_true_flat = inputs.categorical.reshape(-1)
            cat_mask_flat = categorical_mask.reshape(-1)

            # Select only masked positions
            if cat_mask_flat.sum() > 0:
                masked_cat_pred = cat_pred_flat[cat_mask_flat]
                masked_cat_true = cat_true_flat[cat_mask_flat].long()

                categorical_loss = self.categorical_loss_fn(
                    masked_cat_pred, masked_cat_true
                )
                categorical_loss_val = categorical_loss
                total_loss += categorical_loss_val

                # Calculate accuracy for categorical predictions
                categorical_accuracy = (
                    (masked_cat_pred.argmax(dim=-1) == masked_cat_true).float().mean()
                )
                categorical_accuracy_val = categorical_accuracy

        # Epoch-level logging (only for validation)
        self.log(
            "val/numeric_loss",
            numeric_loss_val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "val/categorical_loss",
            categorical_loss_val,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "val/categorical_accuracy",
            categorical_accuracy_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,  # Show val accuracy in progress bar
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )

        return total_loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedules."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/categorical_loss",  # Monitor simplified validation loss
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_validation_epoch_end(self):
        """Create combined charts with train/val metrics for simplified logging"""
        # Ensure logger is available
        if not self.logger or not hasattr(self.logger, "experiment"):
            return

        # Get current epoch metrics
        current_epoch_metrics = self.trainer.callback_metrics

        # Numeric Loss Chart (train vs validation)
        train_numeric_loss = current_epoch_metrics.get(
            "train/numeric_loss", torch.tensor(0.0, device=self.device)
        )
        val_numeric_loss = current_epoch_metrics.get(
            "val/numeric_loss", torch.tensor(0.0, device=self.device)
        )
        self.logger.experiment.add_scalars(
            "Numeric Loss",
            {"train": train_numeric_loss, "validation": val_numeric_loss},
            self.current_epoch,
        )

        # Categorical Loss Chart (train vs validation)
        train_categorical_loss = current_epoch_metrics.get(
            "train/categorical_loss", torch.tensor(0.0, device=self.device)
        )
        val_categorical_loss = current_epoch_metrics.get(
            "val/categorical_loss", torch.tensor(0.0, device=self.device)
        )
        self.logger.experiment.add_scalars(
            "Categorical Loss",
            {"train": train_categorical_loss, "validation": val_categorical_loss},
            self.current_epoch,
        )

        # Categorical Accuracy Chart (train vs validation)
        train_categorical_accuracy = current_epoch_metrics.get(
            "train/categorical_accuracy", torch.tensor(0.0, device=self.device)
        )
        val_categorical_accuracy = current_epoch_metrics.get(
            "val/categorical_accuracy", torch.tensor(0.0, device=self.device)
        )
        self.logger.experiment.add_scalars(
            "Categorical Accuracy",
            {
                "train": train_categorical_accuracy,
                "validation": val_categorical_accuracy,
            },
            self.current_epoch,
        )


if __name__ == "__main__":
    # Run benchmarks if executed directly
    print("Running attention mechanism benchmarks...")
    results = benchmark_attention_mechanisms()

    print("\nBenchmark Results:")
    print("=" * 80)

    for seq_len, attn_results in results.items():
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        for attn_type, metrics in attn_results.items():
            if metrics["successful"]:
                # Print successful benchmark results, wrapped for readability
                print(
                    f"{attn_type:15} | Time: {metrics['time_ms']:8.2f}ms | "
                    f"Memory: {metrics['memory_mb']:8.2f}MB"
                )
            else:
                print(
                    f"{attn_type:15} | Failed: {metrics.get('error', 'Unknown error')}"
                )
