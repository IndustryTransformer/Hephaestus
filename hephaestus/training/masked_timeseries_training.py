from typing import Literal, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn

from hephaestus.timeseries_models import TimeSeriesTransformer
from hephaestus.timeseries_models.model_data_classes import (
    TimeSeriesConfig,
)


class MaskedTabularPretrainer(L.LightningModule):
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
        ] = "flash",
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
        self.encoder = TimeSeriesTransformer(
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

        self.categorical_reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, config.n_tokens),  # Predict token probabilities
        )

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
            masked_numeric[numeric_mask] = float(
                "-inf"
            )  # Use -inf to indicate masked values
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

        batch_size, _, seq_len, d_model = encoder_output.value_embeddings.shape

        # Separate numeric and categorical features for reconstruction
        if input_numeric is not None:
            n_numeric = input_numeric.shape[1]
            numeric_features = encoder_output.value_embeddings[
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
            categorical_features = encoder_output.value_embeddings[
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
        masked_numeric, masked_categorical, _numeric_mask, _categorical_mask = (
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

        # Initialize loss components (don't use requires_grad on initial tensor)
        losses = []
        numeric_loss_val = torch.tensor(0.0, device=self.device)
        categorical_loss_val = torch.tensor(0.0, device=self.device)
        categorical_accuracy_val = torch.tensor(0.0, device=self.device)

        # Calculate numeric loss on ALL positions (reconstruction task)
        if numeric_predictions is not None:
            # Reconstruct ALL numeric values, not just masked ones
            # Transpose to match prediction shape [batch, seq_len, n_numeric]
            numeric_transposed = inputs.numeric.permute(0, 2, 1)

            # Clip predictions to prevent extreme values
            # numeric_pred_clipped = torch.clamp(numeric_predictions, -10.0, 10.0)
            # numeric_true_clipped = torch.clamp(numeric_transposed, -10.0, 10.0)

            numeric_loss = self.numeric_loss_fn(numeric_transposed, inputs.numeric)

            numeric_loss_val = numeric_loss
            losses.append(numeric_loss)

        # Calculate categorical loss on ALL positions (reconstruction task)
        if categorical_predictions is not None:
            # Reconstruct ALL categorical values, not just masked ones
            # categorical_predictions shape: [batch, seq_len, n_categorical, n_tokens]
            # inputs.categorical shape: [batch, n_categorical, seq_len]

            # Reshape predictions: [batch, seq_len, n_categorical, n_tokens] ->
            #       [batch, n_categorical, seq_len, n_tokens]
            cat_pred_reshaped = categorical_predictions.permute(0, 2, 1, 3)
            # Flatten: [batch * n_categorical * seq_len, n_tokens]
            cat_pred_flat = cat_pred_reshaped.reshape(-1, self.config.n_tokens)

            # Reshape targets to match: [batch, n_categorical, seq_len] ->
            #           [batch * n_categorical * seq_len]
            cat_true_flat = inputs.categorical.reshape(-1).long()

            categorical_loss = self.categorical_loss_fn(cat_pred_flat, cat_true_flat)
            categorical_loss_val = categorical_loss
            losses.append(categorical_loss)

            # Calculate categorical accuracy on ALL positions
            predicted_tokens = cat_pred_flat.argmax(dim=-1)
            categorical_accuracy_val = (
                (predicted_tokens == cat_true_flat).float().mean()
            )

        # Combine losses properly to maintain gradient flow
        if losses:
            total_loss = sum(losses)
        else:
            # If no losses, create a zero loss that still allows gradients
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Log only the primary training metric (total loss) for simplified logging
        self.log(
            "train_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        # Log individual losses and accuracy per epoch
        self.log(
            "train_numeric_loss",
            numeric_loss_val,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "train_categorical_loss",
            categorical_loss_val,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "train_categorical_accuracy",
            categorical_accuracy_val,
            on_step=False,
            on_epoch=True,
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
        masked_numeric, masked_categorical, _numeric_mask, _categorical_mask = (
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

        # Initialize loss components
        losses = []
        numeric_loss_val = torch.tensor(0.0, device=self.device)
        categorical_loss_val = torch.tensor(0.0, device=self.device)
        categorical_accuracy_val = torch.tensor(0.0, device=self.device)

        # Calculate numeric loss on ALL positions (reconstruction task)
        if numeric_predictions is not None:
            # Reconstruct ALL numeric values, not just masked ones
            # Transpose to match prediction shape [batch, seq_len, n_numeric]
            numeric_transposed = inputs.numeric.permute(0, 2, 1)

            numeric_loss = self.numeric_loss_fn(numeric_predictions, numeric_transposed)
            numeric_loss_val = numeric_loss
            losses.append(numeric_loss)

        # Calculate categorical loss and accuracy on ALL positions
        if categorical_predictions is not None:
            # Reconstruct ALL categorical values, not just masked ones
            # categorical_predictions shape: [batch, seq_len, n_categorical, n_tokens]
            # inputs.categorical shape: [batch, n_categorical, seq_len]

            # Reshape predictions: [batch, seq_len, n_categorical, n_tokens] ->
            #        [batch, n_categorical, seq_len, n_tokens]
            cat_pred_reshaped = categorical_predictions.permute(0, 2, 1, 3)
            # Flatten: [batch * n_categorical * seq_len, n_tokens]
            cat_pred_flat = cat_pred_reshaped.reshape(-1, self.config.n_tokens)

            # Reshape targets to match: [batch, n_categorical, seq_len] ->
            #       [batch * n_categorical * seq_len]
            cat_true_flat = inputs.categorical.reshape(-1).long()

            categorical_loss = self.categorical_loss_fn(cat_pred_flat, cat_true_flat)
            categorical_loss_val = categorical_loss
            losses.append(categorical_loss)

            # Calculate accuracy on ALL positions
            categorical_accuracy_val = (
                (cat_pred_flat.argmax(dim=-1) == cat_true_flat).float().mean()
            )

        # Combine losses properly to maintain gradient flow
        total_loss = sum(losses)

        # Log only the primary validation metric (total loss) for simplified monitoring
        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        # Log individual losses and accuracy per epoch
        self.log(
            "val_numeric_loss",
            numeric_loss_val,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "val_categorical_loss",
            categorical_loss_val,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "val_categorical_accuracy",
            categorical_accuracy_val,
            on_step=False,
            on_epoch=True,
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
                "monitor": "val_loss",  # Monitor simplified validation loss
                "interval": "epoch",
                "frequency": 1,
            },
        }


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
        pretrained_encoder: TimeSeriesTransformer | None = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads
        self.learning_rate = learning_rate
        self.classification_values = classification_values

        # Class index for the only categorical target (class)
        # self.class_token_index = 0

        # Encoder - use pretrained if provided, otherwise create new
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
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

        batch_size, _num_features, seq_len, d_model = (
            encoder_output.value_embeddings.shape
        )

        # Pool across features (mean or max)
        pooled = encoder_output.value_embeddings.mean(
            dim=1
        )  # [batch, seq_len, d_model]
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

        # For classification, we only have categorical loss (same as total loss)
        # Set numeric loss to zero for consistency with pretraining logging
        numeric_loss_val = torch.tensor(0.0, device=self.device)
        categorical_loss_val = loss

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        self.log(
            "train_numeric_loss",
            numeric_loss_val,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "train_categorical_loss",
            categorical_loss_val,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "train_categorical_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

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
        valid_mask = ~torch.isnan(target_classes)
        target_classes = target_classes[valid_mask]
        class_logits = class_logits[valid_mask]
        # Calculate loss
        loss = self.loss_fn(class_logits, target_classes.long())

        # Calculate accuracy
        predictions = torch.argmax(class_logits, dim=-1)
        accuracy = (predictions == target_classes).float().mean()

        # For classification, we only have categorical loss (same as total loss)
        # Set numeric loss to zero for consistency with pretraining logging
        numeric_loss_val = torch.tensor(0.0, device=self.device)
        categorical_loss_val = loss

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log(
            "val_numeric_loss",
            numeric_loss_val,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_categorical_loss",
            categorical_loss_val,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_categorical_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        """Configure optimizers for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        """Prediction step for Lightning module.

        Returns predictions and targets for evaluation.
        """
        # Extract inputs and targets from batch
        inputs, targets = batch

        # Forward pass
        outputs = self(
            input_numeric=inputs.numeric,
            input_categorical=inputs.categorical,
            deterministic=True,
        )

        # Get predictions
        class_logits = outputs
        predictions = torch.argmax(class_logits, dim=1)  # [batch, seq_len, 1]

        # Extract targets for comparison
        target_classes = targets.categorical  # [batch, seq_len]

        return {"predictions": predictions, "targets": target_classes}

    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
