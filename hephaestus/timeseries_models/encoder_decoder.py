from __future__ import annotations

import pytorch_lightning as L
import torch
from torch import nn

from hephaestus.timeseries_models.model_data_classes import (
    NumericCategoricalData,
    TimeSeriesConfig,
)
from hephaestus.timeseries_models.models import (
    TimeSeriesTransformer,
)


class TabularEncoderDecoder(L.LightningModule):
    """Encoder-Decoder architecture for anomaly detection.

    The encoder processes input features (numeric and categorical),
    while a simple feed-forward network predicts the anomaly class.

    Args:
        config (TimeSeriesConfig): Configuration for the time series model
        d_model (int): Dimension of the model embeddings
        n_heads (int): Number of attention heads
        learning_rate (float): Learning rate for optimization
        classification_values (list, optional): List of classification values

    """

    def __init__(
        self,
        config: TimeSeriesConfig,
        d_model: int = 512,
        n_heads: int = 32,
        learning_rate: float = 1e-4,
        classification_values=None,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads
        self.learning_rate = learning_rate
        self.classification_values = classification_values

        # Class index for the only categorical target (class)
        # self.class_token_index = 0

        # Encoder - processes all input features
        self.encoder = TimeSeriesTransformer(
            config=self.config,
            d_model=self.d_model,
            n_heads=self.n_heads,
        )

        # Output projection for classification
        if classification_values is not None:
            n_classes = len(classification_values)
        else:
            # Fallback to number of unique values in the class token
            # Try to infer from token dictionary
            class_tokens = [
                k
                for k in config.token_dict
                if isinstance(k, str) and (k.startswith("Normal") or k in ["class"])
            ]
            n_classes = len(class_tokens) if class_tokens else 9  # Default to 9 classes

        self.class_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, n_classes),
        )

        # Cross-entropy loss for classification
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Save hyperparameters for loading from checkpoints
        self.save_hyperparameters(ignore=["config"])

    def forward(
        self,
        input_numeric: torch.Tensor | None = None,
        input_categorical: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_numeric: Numeric input features (optional)
            input_categorical: Categorical input features (optional)
            target_numeric: Numeric target values (not used in simplified model)
            target_categorical: Categorical target values (not used in simplified model)
            deterministic: Whether to use deterministic forward pass

        Returns:
            Dictionary containing predicted class logits and encoder outputs
        """
        # Verify that at least one of the inputs is not None
        if input_numeric is None and input_categorical is None:
            raise ValueError(
                "At least one of input_numeric or input_categorical must be provided"
            )

        # Encoder pass - no causal masking
        encoder_output = self.encoder(
            numeric_inputs=input_numeric,
            categorical_inputs=input_categorical,
            deterministic=deterministic,
            causal_mask=False,
        )  # [batch, num_features, seq_len, d_model]

        batch_size, num_features, seq_len, d_model = encoder_output.shape

        # Pool across features (mean or max)
        pooled = encoder_output.mean(dim=1)  # [batch, seq_len, d_model]
        # pooled = encoder_output.max(dim=1).values  # Alternative: max pooling

        # Flatten for FC: [batch * seq_len, d_model]
        x = pooled.contiguous().view(-1, d_model)  # [batch * seq_len, d_model]

        # Pass through FC
        class_logits = self.class_predictor(x)  # [batch * seq_len, n_classes]

        # Reshape back: [batch, 1, seq_len, n_classes]
        class_logits = class_logits.view(batch_size, 1, seq_len, -1)
        class_logits = class_logits.permute(0, 3, 1, 2)
        return class_logits  # [batch, 1, seq_len, n_classes]

    def training_step(self, batch, batch_idx):
        """Training step for Lightning module."""
        inputs, targets = batch

        # Forward pass
        class_logits = self(
            input_numeric=inputs.numeric,
            input_categorical=inputs.categorical,
            deterministic=False,
        )

        target_classes = targets.categorical  # [batch, seq_len]
        class_logits = class_logits.squeeze(2)  # [batch, n_classes, seq_len]

        # Flatten for loss
        class_logits = class_logits.permute(0, 2, 1).reshape(
            -1, class_logits.size(1)
        )  # [batch*seq_len, n_classes]
        target_classes = target_classes.reshape(-1)  # [batch*seq_len]
        valid_mask = ~torch.isnan(target_classes)  # Mask for valid targets
        target_classes = target_classes[valid_mask]  # Remove NaNs
        class_logits = class_logits[valid_mask]
        # Calculate loss

        # if torch.isnan(class_logits).any():
        #     print(
        #         "NaNs found in class_logits at indices:",
        #         torch.nonzero(torch.isnan(class_logits)),
        #     )
        # if torch.isnan(target_classes).any():
        #     print(
        #         "NaNs found in target_classes at indices:",
        #         torch.nonzero(torch.isnan(target_classes)),
        #     )

        loss = self.loss_fn(class_logits, target_classes.long())

        # Calculate accuracy
        predictions = torch.argmax(class_logits, dim=-1)
        accuracy = (predictions == target_classes).float().mean()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning module."""
        inputs, targets = batch

        # Forward pass
        class_logits = self(
            input_numeric=inputs.numeric,
            input_categorical=inputs.categorical,
            deterministic=True,
        )

        target_classes = targets.categorical  # [batch, seq_len]
        class_logits = class_logits.squeeze(2)  # [batch, n_classes, seq_len]

        # Flatten for loss
        class_logits = class_logits.permute(0, 2, 1).reshape(
            -1, class_logits.size(1)
        )  # [batch*seq_len, n_classes]
        target_classes = target_classes.reshape(-1)  # [batch*seq_len]

        # Calculate loss
        loss = self.loss_fn(class_logits, target_classes.long())

        # Calculate accuracy
        predictions = torch.argmax(class_logits, dim=-1)
        accuracy = (predictions == target_classes).float().mean()

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for Lightning module."""
        # For single item prediction

        inputs = batch[0]

        outputs = self(
            input_numeric=inputs.numeric,
            input_categorical=inputs.categorical,
            deterministic=True,
        )

        class_logits = outputs
        predictions = torch.argmax(class_logits, dim=-1)

        return {
            "class_predictions": predictions,
            "class_logits": class_logits,
        }
