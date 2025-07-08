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
    def __init__(self, model_config, d_model, n_heads, lr=1e-3, use_linear_numeric_embedding=True, numeric_embedding_type="standard"):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.n_heads = n_heads
        self.lr = lr

        self.model = TabularEncoderRegressor(
            model_config=model_config,
            d_model=d_model,
            n_heads=n_heads,
            use_linear_numeric_embedding=use_linear_numeric_embedding,
            numeric_embedding_type=numeric_embedding_type,
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
        
        # Main task loss
        main_loss = self.loss_fn(y_hat, y)
        
        # MoE auxiliary loss
        auxiliary_loss = self.model.get_moe_auxiliary_loss()
        
        # Total loss combines main task and auxiliary losses
        total_loss = main_loss + auxiliary_loss
        
        # Log losses
        self.log("train_loss", main_loss)
        self.log("train_auxiliary_loss", auxiliary_loss)
        self.log("train_total_loss", total_loss)
        
        # Log MoE metrics if available
        moe_metrics = self.model.get_moe_metrics()
        for layer_name, metrics in moe_metrics.items():
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, torch.Tensor) and metric_value.numel() == 1:
                    self.log(f"{layer_name}_{metric_name}", metric_value.item())
        
        # Log Neural Feature Engineering metrics if available
        nfe_metrics = self.model.get_neural_feature_engineering_metrics()
        if nfe_metrics:
            self._log_nfe_metrics("train", nfe_metrics)
        
        return total_loss

    def validation_step(self, batch: InputsTarget, batch_idx):
        X = batch.inputs
        y = batch.target
        y_hat = self.model(X.numeric, X.categorical)
        
        # Main task loss
        main_loss = self.loss_fn(y_hat, y)
        
        # MoE auxiliary loss
        auxiliary_loss = self.model.get_moe_auxiliary_loss()
        
        # Total loss combines main task and auxiliary losses
        total_loss = main_loss + auxiliary_loss
        
        # Log losses
        self.log("val_loss", main_loss)
        self.log("val_auxiliary_loss", auxiliary_loss)
        self.log("val_total_loss", total_loss)
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        
        # Log MoE metrics if available
        moe_metrics = self.model.get_moe_metrics()
        for layer_name, metrics in moe_metrics.items():
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, torch.Tensor) and metric_value.numel() == 1:
                    self.log(f"val_{layer_name}_{metric_name}", metric_value.item())
        
        return total_loss

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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Add gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step(closure=optimizer_closure)

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
    
    def _log_nfe_metrics(self, stage: str, nfe_metrics: dict):
        """Log Neural Feature Engineering metrics."""
        try:
            # Log scalar metrics
            if "feature_magnitude" in nfe_metrics:
                self.log(f"{stage}_nfe_feature_magnitude", nfe_metrics["feature_magnitude"])
            
            if "total_engineered_features" in nfe_metrics:
                self.log(f"{stage}_nfe_total_features", nfe_metrics["total_engineered_features"])
            
            # Log component metrics
            if "component_metrics" in nfe_metrics:
                comp_metrics = nfe_metrics["component_metrics"]
                
                # Log feature type weights if available
                if "feature_type_weights" in nfe_metrics and isinstance(nfe_metrics["feature_type_weights"], torch.Tensor):
                    weights = nfe_metrics["feature_type_weights"]
                    for i, weight in enumerate(weights):
                        self.log(f"{stage}_nfe_type_weight_{i}", weight.item())
                
                # Log component-specific metrics
                for comp_name, comp_data in comp_metrics.items():
                    if isinstance(comp_data, dict):
                        for metric_name, metric_value in comp_data.items():
                            if isinstance(metric_value, torch.Tensor) and metric_value.numel() == 1:
                                self.log(f"{stage}_nfe_{comp_name}_{metric_name}", metric_value.item())
                            elif isinstance(metric_value, (int, float)):
                                self.log(f"{stage}_nfe_{comp_name}_{metric_name}", metric_value)
        except Exception as e:
            # Gracefully handle any logging errors
            pass


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
    def __init__(self, model_config, d_model, n_heads, lr=1e-3, use_linear_numeric_embedding=True, numeric_embedding_type="standard"):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.n_heads = n_heads
        self.lr = lr

        self.numeric_loss_fn = nn.MSELoss()
        self.categorical_loss_fn = nn.CrossEntropyLoss()
        self.model_config = model_config

        self.model = MaskedTabularEncoder(model_config, d_model, n_heads, use_linear_numeric_embedding, numeric_embedding_type)

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

        # Main task loss
        main_loss = self.aggregate_loss(x, predicted)
        
        # MoE auxiliary loss
        auxiliary_loss = self.model.get_moe_auxiliary_loss()
        
        # Total loss combines main task and auxiliary losses
        total_loss = main_loss + auxiliary_loss
        
        # Log losses
        self.log("train_loss", main_loss)
        self.log("train_auxiliary_loss", auxiliary_loss)
        self.log("train_total_loss", total_loss)
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        
        # Log MoE metrics if available
        moe_metrics = self.model.get_moe_metrics()
        for layer_name, metrics in moe_metrics.items():
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, torch.Tensor) and metric_value.numel() == 1:
                    self.log(f"{layer_name}_{metric_name}", metric_value.item())
        
        # Log Neural Feature Engineering metrics if available
        nfe_metrics = self.model.get_neural_feature_engineering_metrics()
        if nfe_metrics:
            self._log_nfe_metrics("train", nfe_metrics)
        
        return total_loss

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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Add gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step(closure=optimizer_closure)

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
