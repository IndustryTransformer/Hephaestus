import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn.functional as F

from hephaestus.analysis import DFComparison, Results, process_results
from hephaestus.timeseries_models import TimeSeriesDecoder


class TabularDecoder(L.LightningModule):
    def __init__(self, time_series_config, d_model, n_heads):
        super().__init__()
        # self.save_hyperparameters()
        self.d_model = d_model
        self.n_heads = n_heads
        self.time_series_config = time_series_config  # Store the config as an attribute
        self.model = TimeSeriesDecoder(time_series_config, self.d_model, self.n_heads)

    def forward(self, x):
        """
        Forward pass for the model.
        Handles both batched inputs from DataLoader and single inputs.
        """
        # Check if x is already in the expected format (NumericCategoricalData)
        if hasattr(x, "numeric") and hasattr(x, "categorical"):
            return self.model(x.numeric, x.categorical)

        # Handle case where x is a raw tensor or dictionary
        if isinstance(x, torch.Tensor):
            # Assume it's just numeric data
            return self.model(x, None)
        elif isinstance(x, dict):
            # Convert dict to expected format
            numeric = x.get("numeric", None)
            categorical = x.get("categorical", None)
            return self.model(numeric, categorical)
        else:
            # For dataset item, they might need reshaping
            try:
                numeric = x[0].unsqueeze(0) if isinstance(x, tuple) else x.numeric
                categorical = (
                    x[1].unsqueeze(0) if isinstance(x, tuple) else x.categorical
                )
                return self.model(numeric, categorical)
            except (IndexError, AttributeError) as e:
                raise ValueError(f"Unsupported input format: {type(x)}. Error: {e}")

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
        batch_size, seq_len, feat_dim = inputs.shape

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
        y_true = y_true.to(torch.long)

        # For sequence prediction, we need to set the offset correctly
        # We should use offset=1 to predict the NEXT token, not the current one
        # This means the model input at position i predicts the target at position i+1
        y_true, y_pred, mask = self.add_input_offsets(y_true, y_pred, inputs_offset=1)

        # Log shapes for debugging
        self.log("cat_true_shape", float(y_true.numel()))
        self.log("cat_pred_shape", float(y_pred.numel()))

        # Create a mask for valid values (not NaN)
        mask = ~torch.isnan(y_true)
        if mask.any():
            # Apply log_softmax to predictions
            log_probs = F.log_softmax(y_pred[mask].reshape(-1, y_pred.size(-1)), dim=-1)

            # Get targets and make sure they're valid indices
            targets = y_true[mask].reshape(-1).long()

            # Important! Verify targets are in valid range for nll_loss
            num_classes = y_pred.size(-1)
            valid_targets_mask = (targets >= 0) & (targets < num_classes)

            # Log number of valid targets for debugging
            self.log("valid_targets", float(valid_targets_mask.sum()))

            # Only use valid targets
            if valid_targets_mask.any():
                valid_log_probs = log_probs[valid_targets_mask]
                valid_targets = targets[valid_targets_mask]
                loss = F.nll_loss(valid_log_probs, valid_targets, reduction="mean")

                # Add a small regularization term to encourage diversity in predictions
                entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=1).mean()
                loss = loss - 0.01 * entropy

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

            # Find the is_odd column index
            is_odd_idx = None
            for i, col in enumerate(self.time_series_config.categorical_col_tokens):
                if "is_odd" in col:
                    is_odd_idx = i
                    break

            # Add enhanced monitoring for the is_odd column specifically
            with torch.no_grad():
                if (
                    is_odd_idx is not None and batch_idx % 5 == 0
                ):  # Check more frequently
                    # Get predictions for is_odd column
                    pred_cats = outputs.categorical[:, is_odd_idx].argmax(dim=-1)
                    # true_cats = batch.categorical[:, is_odd_idx].long()

                    # Check alternating pattern in predictions
                    seq_len = pred_cats.shape[1]
                    alternating_count = 0
                    total_checks = 0

                    for i in range(1, seq_len):
                        mask = ~torch.isnan(
                            batch.categorical[:, is_odd_idx, i]
                        ) & ~torch.isnan(batch.categorical[:, is_odd_idx, i - 1])
                        if mask.any():
                            # Count how often predictions alternate (odd->even, even->odd)
                            alternating = (pred_cats[:, i] != pred_cats[:, i - 1])[mask]
                            alternating_count += alternating.sum().item()
                            total_checks += mask.sum().item()

                    if total_checks > 0:
                        alternating_rate = alternating_count / total_checks
                        self.log("alternating_rate", alternating_rate)

                        # Log specific examples to understand the pattern
                        # if batch_idx % 20 == 0:
                        #     sample_idx = 0
                        #     seq_preds = pred_cats[sample_idx].cpu().numpy()
                        #     seq_true = true_cats[sample_idx].cpu().numpy()
                        # print(f"Sample predictions: {seq_preds[:10]}")
                        # print(f"Sample true values: {seq_true[:10]}")

            # Give more weight to categorical loss to emphasize learning patterns
            loss = (
                numeric_loss + 3.0 * categorical_loss
            )  # Increased weight from 2.0 to 3.0
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
            self.time_series_config.categorical_col_tokens,  # Using the stored config
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
