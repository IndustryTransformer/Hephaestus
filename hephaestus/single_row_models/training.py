import pytorch_lightning as L
from torch import nn

from hephaestus.single_row_models.single_row_models import TabularEncoderRegressor

from .model_data_classes import InputsTarget


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
        optimizer = L.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch: InputsTarget):
        with self.no_grad():
            return self.forward(batch)
