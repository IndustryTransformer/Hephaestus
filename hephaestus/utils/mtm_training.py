from datetime import datetime as dt
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jaxlib.xla_extension import ArrayImpl as ArrayImpl
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange

from ..models import models
from .data_utils import MTMModelInputs, TabularDS, create_mtm_model_inputs

# The rest of your script


def calculate_mlm_loss(params: dict, mtm: models.MTM, mi: MTMModelInputs):
    numeric_loss_scaler = 1
    logits, regression = mtm.apply(
        {"params": params},
        mi.categorical_mask,
        mi.numeric_mask,
    )
    categorical_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, mi.categorical_targets
    ).mean()
    numeric_loss = optax.squared_error(regression, mi.numeric_targets).mean()
    total_loss = categorical_loss + numeric_loss * numeric_loss_scaler

    return {
        "total_loss": total_loss,
        "categorical_loss": categorical_loss,
        "numeric_loss": numeric_loss,
    }


def create_mtm_train_state(
    params_key: ArrayImpl,
    mi: MTMModelInputs,
    dataset: TabularDS,
    lr=0.01,
    device=None,
):
    if device is None:
        device = jax.devices()[0]
    model = models.MTM(dataset, d_model=64, n_heads=4)
    params = jax.device_put(
        model.init(
            params_key,
            mi.categorical_mask,
            mi.numeric_mask,
        )["params"],
        device,
    )
    tx = optax.adam(learning_rate=lr)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# def calculate_mlm_loss(params: dict, mtm: nn.Module, mi: MTMModelInputs):


# @jax.jit
def mtm_train_step_no_jit(model: models.MTM, state: train_state, mi: MTMModelInputs):
    """Train step for MLM. Makes use of jit to speed up training."""

    def loss_fn(params):
        return calculate_mlm_loss(params, model, mi)["total_loss"]

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss


# @jax.jit
def mtm_eval_step_no_jit(model: models.MTM, params: dict, mi: MTMModelInputs):
    return calculate_mlm_loss(params, model, mi)


def train_mtm(
    model_state: train_state,
    model: nn.Module,
    dataset: TabularDS,
    model_name: str = "MTM",
    epochs: int = 100,
    batch_size: int = 10_000,
) -> dict:
    total_loss = []
    categorical_loss = []
    numeric_loss = []
    summary_writer = SummaryWriter(
        "runs/" + dt.now().strftime("%Y-%m:%dT%H:%M:%S") + "_" + model_name
    )
    test_mi = create_mtm_model_inputs(dataset, set="test")
    data = [
        create_mtm_model_inputs(dataset, i, batch_size)
        for i in trange(0, len(dataset.X_train_numeric), batch_size)
    ]
    pbar = trange(epochs)
    batch_counter = 0

    # Jit the functions
    mtm_train_step_partial = partial(mtm_train_step_no_jit, model)
    mtm_eval_step_partial = partial(mtm_eval_step_no_jit, model)
    mtm_train_step = jax.jit(mtm_train_step_partial)
    mtm_eval_step = jax.jit(mtm_eval_step_partial)
    for epoch in pbar:
        for mi in data:
            # mi = models.create_mi(dataset, i, batch_size)

            mtm_state, loss = mtm_train_step(model_state, mi)
            train_loss_dict = mtm_eval_step(model_state.params, mi)

            total_loss.append(train_loss_dict["total_loss"].item())
            categorical_loss.append(train_loss_dict["categorical_loss"].item())
            numeric_loss.append(train_loss_dict["numeric_loss"].item())
            # Train Loss
            summary_writer.add_scalar(
                "TrainLoss/total",
                np.array(train_loss_dict["total_loss"].item()),
                batch_counter,
            )
            summary_writer.add_scalar(
                "TrainLoss/categorical",
                np.array(train_loss_dict["categorical_loss"].item()),
                batch_counter,
            )
            summary_writer.add_scalar(
                "TrainLoss/numeric",
                np.array(train_loss_dict["numeric_loss"].item()),
                batch_counter,
            )
            batch_counter += 1
            # Test Loss
            if epoch % 10 == 0:
                test_loss_dict = mtm_eval_step(mtm_state.params, test_mi)
                summary_writer.add_scalar(
                    "TestLoss/total",
                    np.array(test_loss_dict["total_loss"].item()),
                    batch_counter,
                )
                summary_writer.add_scalar(
                    "TestLoss/categorical",
                    np.array(test_loss_dict["categorical_loss"].item()),
                    batch_counter,
                )
                summary_writer.add_scalar(
                    "TestLoss/numeric",
                    np.array(test_loss_dict["numeric_loss"].item()),
                    batch_counter,
                )
            pbar.set_description(
                f"Train Loss: {train_loss_dict['total_loss'].item():.4f}"
            )
            # all_losses.append(loss.item())
            # logger.add_scalar("Loss/train", loss.item(), i)

    total_loss = jnp.array(total_loss)
    categorical_loss = jnp.array(categorical_loss)
    numeric_loss = jnp.array(
        numeric_loss
    )  # A100: 14.00it/s V100: 8.71it/s T4: 2.70it/s
    return {"mtm_state": mtm_state, "total_loss": total_loss}
