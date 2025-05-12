import torch
import torch.nn as nn
import pytorch_lightning as L
from typing import Optional, Dict

from hephaestus.timeseries_models.model_data_classes import (
    TimeSeriesConfig,
    NumericCategoricalData,
)
from hephaestus.timeseries_models.models import (
    TimeSeriesTransformer,
)


class TabularEncoderDecoder(L.LightningModule):
    """
    Encoder-Decoder architecture for anomaly detection.

    The encoder processes input features (numeric and categorical),
    while the decoder predicts the anomaly class using causal masking.

    Args:
        config (TimeSeriesConfig): Configuration for the time series model
        d_model (int): Dimension of the model embeddings
        n_heads (int): Number of attention heads
        learning_rate (float): Learning rate for optimization
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
        d_model: int = 512,
        n_heads: int = 32,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads
        self.learning_rate = learning_rate

        # Class index in token dictionary for target prediction
        self.class_token_index = config.token_dict["class"]

        # Encoder - processes all input features
        self.encoder = TimeSeriesTransformer(
            config=self.config,
            d_model=self.d_model,
            n_heads=self.n_heads,
        )

        # Decoder - processes encoded representation and predicts the class
        self.decoder = TimeSeriesTransformer(
            config=self.config,
            d_model=self.d_model,
            n_heads=self.n_heads,
        )

        # Output projection for classification
        n_classes = len(config.object_tokens)
        self.class_predictor = nn.Linear(self.d_model, n_classes)

        # Cross-entropy loss for classification
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Save hyperparameters for loading from checkpoints
        self.save_hyperparameters(ignore=["config"])

    def forward(
        self,
        input_numeric: torch.Tensor,
        input_categorical: Optional[torch.Tensor] = None,
        target_numeric: Optional[torch.Tensor] = None,
        target_categorical: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the encoder-decoder model.

        Args:
            input_numeric: Numeric input features
            input_categorical: Categorical input features (optional)
            target_numeric: Numeric target values (optional)
            target_categorical: Categorical target values (optional)
            deterministic: Whether to use deterministic forward pass

        Returns:
            Dictionary containing predicted class logits and encoder outputs
        """
        # Encoder pass - no causal masking in encoder
        encoder_output = self.encoder(
            numeric_inputs=input_numeric,
            categorical_inputs=input_categorical,
            deterministic=deterministic,
            causal_mask=False,  # No causal masking in encoder
        )

        # For predictions (not training), we only need encoder outputs
        if target_numeric is None and target_categorical is None:
            # Extract only the class column from encoder outputs
            # Assume class column is at a specific position in the model
            batch_size, _, seq_len, _ = encoder_output.shape

            # Use last token from sequence dimension for classification
            class_features = encoder_output[:, self.class_token_index, -1, :]
            class_logits = self.class_predictor(class_features)

            return {
                "class_logits": class_logits,
                "encoder_output": encoder_output,
            }

        # Decoder pass with causal masking
        decoder_output = self.decoder(
            numeric_inputs=target_numeric,
            categorical_inputs=target_categorical,
            deterministic=deterministic,
            causal_mask=True,  # Use causal masking in decoder
        )

        # Extract features for class prediction
        # Use the class column from categorical inputs
        batch_size, n_cols, seq_len, _ = decoder_output.shape

        # Find column index for class prediction (using class token from config)
        class_column_idx = None
        for i, col_name in enumerate(self.config.categorical_col_tokens):
            if col_name == "class":
                class_column_idx = i
                break

        if class_column_idx is None:
            raise ValueError("'class' column not found in categorical columns")

        # Extract features from the class column
        class_features = decoder_output[:, class_column_idx, :, :]

        # Apply classification head to predict class logits
        class_logits = self.class_predictor(class_features)

        return {
            "class_logits": class_logits,
            "encoder_output": encoder_output,
            "decoder_output": decoder_output,
        }

    def training_step(self, batch, batch_idx):
        """Training step for Lightning module."""
        inputs, targets = batch

        # Forward pass
        outputs = self(
            input_numeric=inputs.numeric,
            input_categorical=inputs.categorical,
            target_numeric=targets.numeric,
            target_categorical=targets.categorical,
            deterministic=False,
        )

        # Get class logits and target classes
        class_logits = outputs["class_logits"]

        # Find target classes from categorical targets
        class_col_idx = None
        for i, col_name in enumerate(self.config.categorical_col_tokens):
            if col_name == "class":
                class_col_idx = i
                break

        if class_col_idx is None:
            raise ValueError("'class' column not found in categorical columns")

        target_classes = targets.categorical[:, class_col_idx, :]

        # Reshape for cross-entropy loss
        batch_size, seq_len, n_classes = class_logits.shape
        class_logits = class_logits.view(-1, n_classes)
        target_classes = target_classes.view(-1)

        # Calculate loss
        loss = self.loss_fn(class_logits, target_classes)

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
        outputs = self(
            input_numeric=inputs.numeric,
            input_categorical=inputs.categorical,
            target_numeric=targets.numeric,
            target_categorical=targets.categorical,
            deterministic=True,
        )

        # Get class logits and target classes
        class_logits = outputs["class_logits"]

        # Find target classes from categorical targets
        class_col_idx = None
        for i, col_name in enumerate(self.config.categorical_col_tokens):
            if col_name == "class":
                class_col_idx = i
                break

        if class_col_idx is None:
            raise ValueError("'class' column not found in categorical columns")

        target_classes = targets.categorical[:, class_col_idx, :]

        # Reshape for cross-entropy loss
        batch_size, seq_len, n_classes = class_logits.shape
        class_logits = class_logits.view(-1, n_classes)
        target_classes = target_classes.view(-1)

        # Calculate loss
        loss = self.loss_fn(class_logits, target_classes)

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
        if isinstance(batch, NumericCategoricalData):
            inputs = batch

            outputs = self(
                input_numeric=inputs.numeric.unsqueeze(0)
                if inputs.numeric.dim() == 2
                else inputs.numeric,
                input_categorical=inputs.categorical.unsqueeze(0)
                if inputs.categorical is not None and inputs.categorical.dim() == 2
                else inputs.categorical,
                deterministic=True,
            )

            class_logits = outputs["class_logits"]
            predictions = torch.argmax(class_logits, dim=-1)

            return {
                "class_predictions": predictions,
                "class_logits": class_logits,
            }

        # For batch prediction
        if isinstance(batch, tuple) and len(batch) == 2:
            inputs, _ = batch
        else:
            inputs = batch

        outputs = self(
            input_numeric=inputs.numeric,
            input_categorical=inputs.categorical,
            deterministic=True,
        )

        class_logits = outputs["class_logits"]
        predictions = torch.argmax(class_logits, dim=-1)

        return {
            "class_predictions": predictions,
            "class_logits": class_logits,
        }
