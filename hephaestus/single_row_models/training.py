from typing import Optional

import pytorch_lightning as L
import torch
from torch import nn

from hephaestus.single_row_models.model_data_classes import InputsTarget
from hephaestus.single_row_models.single_row_models import (
    TabularEncoderRegressor,
    MaskedTabularEncoder,
)
from hephaestus.utils import NumericCategoricalData


class TabularRegressor(L.LightningModule):
    def __init__(self, model_config, d_model, n_heads, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.n_heads = n_heads
        self.lr = lr

        self.model = TabularEncoderRegressor(
            model_config=model_config,
            d_model=d_model,
            n_heads=n_heads,
        )
        self.loss_fn = nn.MSELoss()

        self.example_input_array = self._create_example_input(3)

    def forward(self, x: Optional[InputsTarget] = None, *args, **kwargs):
        if x is None:
            return self.model(
                kwargs["inputs"]["numeric"], kwargs["inputs"]["categorical"]
            )
        return self.model(x.inputs.numeric, x.inputs.categorical)

    def training_step(self, batch: InputsTarget, batch_idx):
        X = batch.inputs
        y = batch.target
        y_hat = self.model(X.numeric, X.categorical)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: InputsTarget, batch_idx):
        X = batch.inputs
        y = batch.target
        y_hat = self.model(X.numeric, X.categorical)
        # print(f"{y_hat.shape=}, {y.shape=}")
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=3,  # Reduced from 10 to 3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def predict_step(self, batch: InputsTarget):
        with torch.no_grad():
            return self.forward(batch)

    def _create_example_input(self, batch_size: int):
        numeric = torch.rand(batch_size, self.model.model_config.n_numeric_cols)
        categorical = torch.randint(
            0,
            self.model.model_config.n_tokens,
            (batch_size, self.model.model_config.n_cat_cols),  # Please dont cat call
        ).float()

        target = torch.rand(batch_size, 1)
        return {
            "inputs": {"numeric": numeric, "categorical": categorical},
            "target": target,
        }


def tabular_collate_fn(batch):
    """Custom collate function for NumericCategoricalData objects."""
    numeric_tensors = torch.stack([item.inputs.numeric for item in batch])

    if batch[0].inputs.categorical is not None:
        categorical_tensors = torch.stack([item.inputs.categorical for item in batch])
    else:
        categorical_tensors = None
    if batch[0].target is not None:
        target_tensors = torch.stack([item.target for item in batch])
        if target_tensors.dim() == 1:
            target_tensors = target_tensors.unsqueeze(
                -1
            )  # Ensure target tensors have shape (batch_size, 1)
    else:
        target_tensors = None

    return InputsTarget(
        inputs=NumericCategoricalData(
            numeric=numeric_tensors, categorical=categorical_tensors
        ),
        target=target_tensors,
    )


def masked_tabular_collate_fn(batch):
    """Custom collate function for NumericCategoricalData objects."""
    numeric_tensors = torch.stack([item.inputs.numeric for item in batch])

    if batch[0].inputs.categorical is not None:
        categorical_tensors = torch.stack([item.inputs.categorical for item in batch])
    else:
        categorical_tensors = None
    if batch[0].target is not None:
        target_tensors = torch.stack([item.target for item in batch])
        if target_tensors.dim() == 1:
            target_tensors = target_tensors.unsqueeze(
                -1
            )  # Ensure target tensors have shape (batch_size, 1)
    else:
        target_tensors = None

    return NumericCategoricalData(
        numeric=numeric_tensors, categorical=categorical_tensors
    )


class MaskedTabularModeling(L.LightningModule):
    def __init__(self, model_config, d_model, n_heads, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.n_heads = n_heads
        self.lr = lr

        self.numeric_loss_fn = nn.MSELoss()
        self.categorical_loss_fn = nn.CrossEntropyLoss()
        self.model_config = model_config

        self.model = MaskedTabularEncoder(model_config, d_model, n_heads)

    def forward(self, x: NumericCategoricalData) -> NumericCategoricalData:
        return self.model(x.numeric, x.categorical)

    def aggregate_loss(
        self, actual: NumericCategoricalData, predicted: NumericCategoricalData
    ):
        numeric_loss = self.numeric_loss_fn(actual.numeric, predicted.numeric)

        # Ensure actual.categorical is 1D and matches the sequence length
        actual_categorical = actual.categorical.view(
            -1
        )  # Flatten to [batch_size * seq_len]

        # Ensure predicted.categorical is logits and flattened correctly
        batch_size, seq_len, num_classes = (
            predicted.categorical.size()
        )  # Extract dimensions
        predicted_categorical = predicted.categorical.view(
            batch_size * seq_len, num_classes
        )

        categorical_loss = self.categorical_loss_fn(
            predicted_categorical, actual_categorical
        )
        return numeric_loss + categorical_loss

    def training_step(self, x: NumericCategoricalData, probability: float = 0.8):
        numeric = x.numeric
        categorical = x.categorical

        numeric_masked = mask_tensor(numeric, self.model, probability)
        categorical_masked = mask_tensor(categorical, self.model, probability)

        predicted = self.model(numeric_masked, categorical_masked)

        loss = self.aggregate_loss(x, predicted)
        self.log("train_loss", loss)
        self.log("lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def predict_step(self, batch: InputsTarget):
        with torch.no_grad():
            return self.forward(batch)

    def validation_step(self, x: NumericCategoricalData, probability: float = 0.8):
        numeric = x.numeric
        categorical = x.categorical
        numeric_masked = mask_tensor(numeric, self.model, probability)
        categorical_masked = mask_tensor(categorical, self.model, probability)
        predicted = self.model(numeric_masked, categorical_masked)
        loss = self.aggregate_loss(x, predicted)
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=3,  # Reduced from 10 to 3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


def mask_tensor(tensor, model, probability=0.8):
    if tensor.dtype == torch.float32:
        is_numeric = True
    elif tensor.dtype == torch.int32 or tensor.dtype == torch.int64:
        is_numeric = False
    else:
        raise ValueError(f"Task {tensor.dtype} not supported.")

    tensor = tensor.clone()
    bit_mask = torch.rand(tensor.shape, device=tensor.device) > probability
    if is_numeric:
        tensor[bit_mask] = torch.tensor(float("-Inf"))
    else:
        # Get the cat_mask_token and move it to the same device as the tensor
        # For MaskedTabularEncoder, access through tabular_encoder
        if hasattr(model, 'tabular_encoder'):
            mask_token = model.tabular_encoder.cat_mask_token.to(tensor.device)
        else:
            mask_token = model.cat_mask_token.to(tensor.device)
        tensor[bit_mask] = mask_token
    # Use the tensor's own device instead of model.device
    return tensor
