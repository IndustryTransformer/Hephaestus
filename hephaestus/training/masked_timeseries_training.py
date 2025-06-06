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

    def create_masks(self, inputs, mask_probability=None):
        """Create proper BERT-style masks for reconstruction"""
        if inputs is None:
            return None, None

        if mask_probability is None:
            mask_probability = self.mask_probability

        batch_size, n_features, seq_len = inputs.shape
        device = inputs.device

        # Create random mask (True = masked)
        mask = (
            torch.rand(batch_size, n_features, seq_len, device=device)
            < mask_probability
        )

        # Don't mask padding positions (assume 0 or NaN are padding)
        if torch.is_floating_point(inputs):
            padding_mask = torch.isnan(inputs) | (inputs == 0)
        else:
            pad_token_id = self.config.token_dict.get("[PAD]", 0)
            padding_mask = inputs == pad_token_id

        mask = mask & ~padding_mask

        return mask

    def bert_style_masking(self, inputs, mask):
        """BERT-style masking: 80% [MASK], 10% random, 10% unchanged"""
        if inputs is None or mask is None:
            return inputs, mask

        masked_inputs = inputs.clone()

        if torch.is_floating_point(inputs):
            # For numeric data: use NaN as mask token
            masked_inputs[mask] = torch.nan
        else:
            # For categorical data: BERT-style masking
            device = inputs.device

            # 80% of masked tokens become [MASK]
            mask_token_positions = mask & (torch.rand_like(mask.float()) < 0.8)

            # 10% become random tokens
            random_positions = (
                mask & ~mask_token_positions & (torch.rand_like(mask.float()) < 0.5)
            )

            # 10% stay unchanged (but still predicted)

            mask_token_idx = self.config.token_dict.get("[MASK]", 2)
            masked_inputs[mask_token_positions] = mask_token_idx

            # Random tokens (excluding special tokens)
            vocab_size = self.config.n_tokens
            special_tokens = {0, 1, 2}  # [PAD], [UNK], [MASK]
            valid_tokens = [i for i in range(vocab_size) if i not in special_tokens]

            if valid_tokens:
                random_tokens = torch.randint(
                    0,
                    len(valid_tokens),
                    random_positions.sum().unsqueeze(0).shape,
                    device=device,
                )
                random_token_ids = torch.tensor(valid_tokens, device=device)[
                    random_tokens
                ]
                masked_inputs[random_positions] = random_token_ids

        return masked_inputs, mask

    def create_attention_mask(self, input_mask, causal=False):
        """Create attention mask that respects both causal and padding constraints"""
        if input_mask is None:
            return None

        batch_size, seq_len = input_mask.shape
        device = input_mask.device

        # Create causal mask (lower triangular) if needed
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            causal_mask = torch.zeros(
                batch_size, seq_len, seq_len, dtype=torch.bool, device=device
            )

        # Combine with padding mask
        # input_mask: True for valid positions, False for padding
        # attention expects: False for attend, True for mask
        padding_mask = ~input_mask.unsqueeze(1) | ~input_mask.unsqueeze(2)

        # Combine masks
        combined_mask = causal_mask | padding_mask

        return combined_mask

    def calculate_reconstruction_loss(self, predictions, targets, mask):
        """Calculate loss ONLY on masked positions"""
        if predictions is None or mask is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Only compute loss where mask is True
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]

        if masked_predictions.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if torch.is_floating_point(targets):
            # Numeric loss
            return self.numeric_loss_fn(masked_predictions, masked_targets)
        else:
            # Categorical loss
            return self.categorical_loss_fn(masked_predictions, masked_targets.long())

    def get_mask_probability(self, epoch, total_epochs):
        """Progressive masking: start low, increase over training"""
        min_prob = 0.05
        max_prob = 0.25

        # Linear schedule
        progress = epoch / max(total_epochs, 1)
        return min_prob + (max_prob - min_prob) * progress

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
        """Training step with proper masked modeling."""
        inputs, _ = batch  # Ignore targets for pre-training
        batch_size = (
            inputs.numeric.shape[0]
            if inputs.numeric is not None
            else inputs.categorical.shape[0]
        )

        # Use progressive masking if enabled
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        mask_prob = self.get_mask_probability(current_epoch, 50)  # 50 max epochs

        # Create masks for reconstruction task
        numeric_mask = (
            self.create_masks(inputs.numeric, mask_prob)
            if inputs.numeric is not None
            else None
        )
        categorical_mask = (
            self.create_masks(inputs.categorical, mask_prob)
            if inputs.categorical is not None
            else None
        )

        # Apply BERT-style masking
        masked_numeric, _ = self.bert_style_masking(inputs.numeric, numeric_mask)
        masked_categorical, _ = self.bert_style_masking(
            inputs.categorical, categorical_mask
        )

        # Note: attention masking could be added here if needed in future
        # Currently TimeSeriesTransformer doesn't use explicit attention masks

        # Forward pass with masked inputs
        numeric_predictions, categorical_predictions = self(
            input_numeric=masked_numeric,
            input_categorical=masked_categorical,
            deterministic=False,
        )

        # Initialize loss components
        losses = []
        numeric_loss_val = torch.tensor(0.0, device=self.device)
        categorical_loss_val = torch.tensor(0.0, device=self.device)
        categorical_accuracy_val = torch.tensor(0.0, device=self.device)

        # Calculate numeric loss ONLY on masked positions
        if numeric_predictions is not None and numeric_mask is not None:
            # Transpose to match prediction shape [batch, seq_len, n_numeric]
            numeric_transposed = inputs.numeric.permute(0, 2, 1)

            # Transpose mask to match
            numeric_mask_transposed = numeric_mask.permute(0, 2, 1)

            # Calculate loss only on masked positions
            numeric_loss = self.calculate_reconstruction_loss(
                numeric_predictions, numeric_transposed, numeric_mask_transposed
            )

            numeric_loss_val = numeric_loss
            losses.append(numeric_loss)

            # Debug info
            masked_count = numeric_mask_transposed.sum().item()
            print(
                f"Numeric: {masked_count} masked tokens, loss: {numeric_loss_val.item():.4f}"
            )

        # Calculate categorical loss ONLY on masked positions
        if categorical_predictions is not None and categorical_mask is not None:
            # categorical_predictions shape: [batch, seq_len, n_categorical, n_tokens]
            # inputs.categorical shape: [batch, n_categorical, seq_len]
            # categorical_mask shape: [batch, n_categorical, seq_len]

            # Reshape predictions: [batch, seq_len, n_categorical, n_tokens] ->
            #       [batch, n_categorical, seq_len, n_tokens]
            cat_pred_reshaped = categorical_predictions.permute(0, 2, 1, 3)

            # Only get predictions for masked positions
            # Get flattened versions for easier indexing
            cat_pred_flat = cat_pred_reshaped.reshape(-1, self.config.n_tokens)
            mask_flat = categorical_mask.reshape(-1)

            # Select only masked predictions
            masked_cat_pred = cat_pred_flat[mask_flat]

            # Get corresponding target tokens
            masked_cat_targets = inputs.categorical[categorical_mask]

            if masked_cat_pred.numel() > 0:
                # Reshape to [num_masked_tokens, n_tokens] and [num_masked_tokens]
                masked_cat_pred = masked_cat_pred.view(-1, self.config.n_tokens)
                masked_cat_targets = masked_cat_targets.view(-1).long()

                categorical_loss = self.categorical_loss_fn(
                    masked_cat_pred, masked_cat_targets
                )
                categorical_loss_val = categorical_loss
                losses.append(categorical_loss)

                # Calculate accuracy on masked positions only
                predicted_tokens = masked_cat_pred.argmax(dim=-1)
                categorical_accuracy_val = (
                    (predicted_tokens == masked_cat_targets).float().mean()
                )

                # Debug info
                masked_count = categorical_mask.sum().item()
                print(
                    f"Categorical: {masked_count} masked tokens, loss: {categorical_loss_val.item():.4f}, acc: {categorical_accuracy_val.item():.4f}"
                )

        # Combine losses properly to maintain gradient flow
        if losses:
            total_loss = sum(losses)
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Log only the primary training metric (total loss) for simplified logging
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
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
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "train_categorical_loss",
            categorical_loss_val,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        self.log(
            "train_categorical_accuracy",
            categorical_accuracy_val,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
            reduce_fx="mean",
        )
        if torch.isnan(total_loss):
            raise ValueError(
                "Total loss is NaN. Check if inputs or predictions contain NaNs."
            )
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step with proper masked modeling."""
        inputs, _ = batch
        batch_size = (
            inputs.numeric.shape[0]
            if inputs.numeric is not None
            else inputs.categorical.shape[0]
        )

        # Use same masking strategy as training
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        mask_prob = self.get_mask_probability(current_epoch, 50)  # 50 max epochs

        # Create masks for reconstruction task
        numeric_mask = (
            self.create_masks(inputs.numeric, mask_prob)
            if inputs.numeric is not None
            else None
        )
        categorical_mask = (
            self.create_masks(inputs.categorical, mask_prob)
            if inputs.categorical is not None
            else None
        )

        # Apply BERT-style masking
        masked_numeric, _ = self.bert_style_masking(inputs.numeric, numeric_mask)
        masked_categorical, _ = self.bert_style_masking(
            inputs.categorical, categorical_mask
        )

        # Forward pass
        numeric_predictions, categorical_predictions = self(
            input_numeric=masked_numeric,
            input_categorical=masked_categorical,
            deterministic=True,
        )

        # Initialize loss components
        losses = []
        numeric_loss_val = torch.tensor(0.0, device=self.device)
        categorical_loss_val = torch.tensor(0.0, device=self.device)
        categorical_accuracy_val = torch.tensor(0.0, device=self.device)

        # Calculate numeric loss ONLY on masked positions
        if numeric_predictions is not None and numeric_mask is not None:
            # Transpose to match prediction shape [batch, seq_len, n_numeric]
            numeric_transposed = inputs.numeric.permute(0, 2, 1)
            numeric_mask_transposed = numeric_mask.permute(0, 2, 1)

            # Calculate loss only on masked positions
            numeric_loss = self.calculate_reconstruction_loss(
                numeric_predictions, numeric_transposed, numeric_mask_transposed
            )

            numeric_loss_val = numeric_loss
            losses.append(numeric_loss)

        # Calculate categorical loss ONLY on masked positions
        if categorical_predictions is not None and categorical_mask is not None:
            # Reshape predictions: [batch, seq_len, n_categorical, n_tokens] ->
            #       [batch, n_categorical, seq_len, n_tokens]
            cat_pred_reshaped = categorical_predictions.permute(0, 2, 1, 3)

            # Only get predictions for masked positions
            # Get flattened versions for easier indexing
            cat_pred_flat = cat_pred_reshaped.reshape(-1, self.config.n_tokens)
            mask_flat = categorical_mask.reshape(-1)

            # Select only masked predictions
            masked_cat_pred = cat_pred_flat[mask_flat]

            # Get corresponding target tokens
            masked_cat_targets = inputs.categorical[categorical_mask]

            if masked_cat_pred.numel() > 0:
                # Reshape to [num_masked_tokens, n_tokens] and [num_masked_tokens]
                masked_cat_pred = masked_cat_pred.view(-1, self.config.n_tokens)
                masked_cat_targets = masked_cat_targets.view(-1).long()

                categorical_loss = self.categorical_loss_fn(
                    masked_cat_pred, masked_cat_targets
                )
                categorical_loss_val = categorical_loss
                losses.append(categorical_loss)

                # Calculate accuracy on masked positions only
                predicted_tokens = masked_cat_pred.argmax(dim=-1)
                categorical_accuracy_val = (
                    (predicted_tokens == masked_cat_targets).float().mean()
                )

        # Combine losses properly to maintain gradient flow
        if losses:
            total_loss = sum(losses)
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

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
