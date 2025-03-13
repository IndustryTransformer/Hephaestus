import pytorch_lightning as L

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

    def forward(self, x: InputsTarget):
        return self.model(x.inputs.numeric, x.inputs.categorical)
