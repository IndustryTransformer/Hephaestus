from typing import Literal, Optional

import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn.functional as F

from hephaestus.analysis import DFComparison, Results, process_results
from hephaestus.timeseries_models import TimeSeriesDecoder


class TabularDecoder(L.LightningModule):
    def __init__(
        self,
        time_series_config,
        d_model,
        n_heads,
        n_layers: int = 4,
        attention_type: Literal[
            "standard", "local", "sparse", "featurewise", "chunked", "flash"
        ] = "standard",
        attention_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_type = attention_type
        self.attention_kwargs = attention_kwargs or {}

        self.model = TimeSeriesDecoder(
            time_series_config,
            self.d_model,
            self.n_heads,
            n_layers=self.n_layers,
            attention_type=self.attention_type,
            attention_kwargs=self.attention_kwargs,
        )

    def forward(self, x):
        return self.model(x.numeric, x.categorical)

    def add_input_offsets(self, inputs, outputs, inputs_offset=1):
        """Add offsets to inputs and apply mask to outputs (PyTorch version).

        Args:
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.
            inputs_offset (int, optional): Offset for inputs. Defaults to 1.

        Returns:
            tuple: Tuple containing modified inputs, outputs, and nan mask.
        """
        # Process inputs with offset
        batch_size, seq_len, _feat_dim = inputs.shape

        # Process inputs with offset
        inputs_shifted = inputs[:, :, inputs_offset:]
        tmp_null = torch.full(
            (batch_size, seq_len, inputs_offset), float("nan"), device=inputs.device
        )
        inputs_padded = torch.cat([inputs_shifted, tmp_null], dim=2)

        # Create mask and handle NaNs consistently
        nan_mask = torch.isnan(inputs_padded)
        inputs_clean = torch.where(
            nan_mask, torch.zeros_like(inputs_padded), inputs_padded
        )

        # Mask outputs consistently with the JAX version
        if outputs.dim() > inputs_clean.dim():
            # Handle categorical outputs
            nan_mask_expanded = nan_mask.unsqueeze(-1).expand(
                -1, -1, -1, outputs.shape[-1]
            )
            outputs_masked = torch.where(
                nan_mask_expanded, torch.zeros_like(outputs), outputs
            )
        else:
            # Handle numeric outputs
            outputs_masked = torch.where(nan_mask, torch.zeros_like(outputs), outputs)

        return inputs_clean, outputs_masked, nan_mask

    def categorical_loss(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Calculate categorical loss between true and predicted values."""
        # Keep predictions as float32 for softmax operation
        y_pred = y_pred.to(torch.float32)

        # Convert target to long type but handle it carefully
        y_true = y_true.to(torch.long)  # Convert to long (int64) for class indices

        # Apply offset adjustment to inputs and outputs
        y_true, y_pred, _ = self.add_input_offsets(y_true, y_pred, inputs_offset=1)

        mask = ~torch.isnan(y_true)
        if mask.any():
            # Apply log_softmax to predictions
            log_probs = F.log_softmax(y_pred[mask].reshape(-1, y_pred.size(-1)), dim=-1)

            # Get targets and make sure they're valid indices
            targets = y_true[mask].reshape(-1).long()

            # Important! Verify targets are in valid range for nll_loss
            num_classes = y_pred.size(-1)
            valid_targets_mask = (targets >= 0) & (targets < num_classes)

            # Only use valid targets
            if valid_targets_mask.any():
                valid_log_probs = log_probs[valid_targets_mask]
                valid_targets = targets[valid_targets_mask]
                loss = F.nll_loss(valid_log_probs, valid_targets, reduction="mean")
                return loss
            else:
                # Return zero loss if no valid targets found
                return torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)

        return torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)

    def numeric_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate numeric loss between true and predicted values."""
        y_true = y_true.to(torch.float32)  # TODO Remove this to
        y_pred = y_pred.to(torch.float32)
        # Apply offset adjustment to inputs and outputs
        y_true, y_pred, _ = self.add_input_offsets(y_true, y_pred, inputs_offset=1)

        # Remove NaN values from the loss calculation
        mask = ~torch.isnan(y_true)
        if mask.any():
            loss = F.mse_loss(y_pred[mask], y_true[mask], reduction="mean")
            return loss
        return torch.tensor(0.0, device=y_true.device, dtype=torch.float32)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        numeric_loss = self.numeric_loss(batch.numeric, outputs.numeric)
        if batch.categorical is not None:
            categorical_loss = self.categorical_loss(
                batch.categorical, outputs.categorical
            )
            loss = numeric_loss + categorical_loss
        else:
            categorical_loss = torch.tensor(0.0, device=self.device)
            loss = numeric_loss
        self.log("train_loss", loss)
        self.log("train_numeric_loss", numeric_loss)
        self.log("train_categorical_loss", categorical_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        numeric_loss = self.numeric_loss(batch.numeric, outputs.numeric)
        if batch.categorical is not None:
            categorical_loss = self.categorical_loss(
                batch.categorical, outputs.categorical
            )
            loss = numeric_loss + categorical_loss
        else:
            categorical_loss = torch.tensor(0.0, device=self.device)
            loss = numeric_loss
        self.log("val_loss", loss)
        self.log("val_numeric_loss", numeric_loss)
        self.log("val_categorical_loss", categorical_loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return optimizer

    def predict_step(self, batch):
        with torch.no_grad():
            return self.forward(batch)

    def return_results(self, dataset, idx, mask_start):
        inputs = dataset[idx]
        inputs.numeric = inputs.numeric[:, :mask_start].unsqueeze(0)
        inputs.categorical = inputs.categorical[:, :mask_start].unsqueeze(
            0
        )  # adding batch dimension
        outputs = self.predict_step(inputs)

        return Results(
            numeric_out=outputs.numeric,
            categorical_out=outputs.categorical,
            numeric_inputs=inputs.numeric,
            categorical_inputs=inputs.categorical,
        )

    def return_results_df(self, inputs):
        with torch.no_grad():
            results = self.return_results(inputs)

        input_categorical = process_results(
            results.categorical_inputs,
            self.categorical_col_tokens,
            self.time_series_config,
        )

        input_numeric = process_results(
            results.numeric_inputs,
            self.time_series_config.numeric_col_tokens,
            self.time_series_config,
        )
        output_categorical = process_results(
            results.categorical_out,
            self.time_series_config.categorical_col_tokens,
            self.time_series_config,
        )
        output_numeric = process_results(
            results.numeric_out,
            self.time_series_config.numeric_col_tokens,
            self.time_series_config,
        )
        input_df = pd.concat([input_categorical, input_numeric], axis=1)
        output_df = pd.concat([output_categorical, output_numeric], axis=1)

        return DFComparison(input_df, output_df)


class TabularTransformer(L.LightningModule):
    def __init__(self, time_series_config, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.model = TimeSeriesDecoder(time_series_config, self.d_model, self.n_heads)

    def forward(self, x):
        out = self.model(x.numeric, x.categorical)
        return out
