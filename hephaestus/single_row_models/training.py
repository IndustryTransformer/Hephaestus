import pytorch_lightning as L
import torch
from torch import nn

from hephaestus.single_row_models.model_data_classes import InputsTarget
from hephaestus.single_row_models.single_row_models import TabularEncoderRegressor
from hephaestus.utils import NumericCategoricalData


class TabularRegressor(L.LightningModule):
    def __init__(self, model_config, d_model, n_heads):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.n_heads = n_heads

        self.model = TabularEncoderRegressor(
            model_config=model_config,
            d_model=d_model,
            n_heads=n_heads,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: InputsTarget):
        return self.model(x.inputs.numeric, x.inputs.categorical)

    def training_step(self, batch: InputsTarget, batch_idx):
        X = batch.inputs
        y = batch.target
        y_hat = self.model(X.numeric, X.categorical)
        # y_hat = y_hat.squeeze()
        # change y from size 20 to size (20, 1)
        # y = y.unsqueeze(-1)
        print(f"{y_hat.shape=}, {y.shape=}, {y_hat=}, {y=}")
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: InputsTarget, batch_idx):
        X = batch.inputs
        y = batch.target
        y_hat = self.model(X.numeric, X.categorical)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch: InputsTarget):
        with self.no_grad():
            return self.forward(batch)


# def tabular_collate_fn(batch):
#     """Custom collate function for NumericCategoricalData objects."""
#     # return batch
#     numeric_tensors = torch.stack([item.inputs.numeric for item in batch])

#     if batch[0].inputs.categorical is not None:
#         categorical_tensors = torch.stack([item.inputs.categorical for item in batch])

#     else:
#         categorical_tensors = None


#     target_tensors = torch.stack([item.target for item in batch])
#     target_tensors = target_tensors.squeeze(-1).unsqueeze(-1)
#     return InputsTarget(
#         inputs=NumericCategoricalData(
#             numeric=numeric_tensors, categorical=categorical_tensors
#         ),
#         target=target_tensors,
#     )
def tabular_collate_fn(batch):
    """Custom collate function for NumericCategoricalData objects."""
    numeric_tensors = torch.stack([item.inputs.numeric for item in batch])

    if batch[0].inputs.categorical is not None:
        categorical_tensors = torch.stack([item.inputs.categorical for item in batch])
    else:
        categorical_tensors = None

    target_tensors = torch.stack([item.target for item in batch])
    if target_tensors.dim() == 1:
        target_tensors = target_tensors.unsqueeze(
            -1
        )  # Ensure target tensors have shape (batch_size, 1)

    return InputsTarget(
        inputs=NumericCategoricalData(
            numeric=numeric_tensors, categorical=categorical_tensors
        ),
        target=target_tensors,
    )
