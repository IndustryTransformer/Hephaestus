from datetime import datetime as dt
from functools import partial

import flax.linen as nn
import jax

# import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jaxlib.xla_extension import ArrayImpl as ArrayImpl
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange

from ..models import time_series
from .data_utils import (
    TabularDS,
    TimeSeriesModelInputs,
    create_masked_time_series_model_inputs,
)

# The rest of your script


def calculate_masked_time_series_loss(
    params: dict,
    masked_time_series: time_series.MaskedTimeSeries,
    mi: TimeSeriesModelInputs,
):
    numeric_loss_scaler = 1
    logits, regression = masked_time_series.apply(
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


def create_masked_time_series_train_state(
    params_key: ArrayImpl,
    mi: TimeSeriesModelInputs,
    dataset: TabularDS,
    lr=0.01,
    device=None,
    n_heads=4,
):
    if device is None:
        device = jax.devices()[0]
    model = time_series.MaskedTimeSeries(dataset, d_model=64, n_heads=n_heads)
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


# def calculate_mts_loss(params: dict, mts: nn.Module, mi: mtsModelInputs):


# @jax.jit
def masked_time_series_train_step_no_jit(
    model: time_series.MaskedTimeSeries,
    state: train_state,
    mi: TimeSeriesModelInputs,
):
    """Train step for MLM. Makes use of jit to speed up training."""

    def loss_fn(params):
        return calculate_masked_time_series_loss(params, model, mi)["total_loss"]

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss


# @jax.jit
def masked_time_series_eval_step_no_jit(
    model: time_series.MaskedTimeSeries, params: dict, mi: TimeSeriesModelInputs
):
    return calculate_masked_time_series_loss(params, model, mi)


def train_masked_time_series(
    model_state: train_state,
    model: nn.Module,
    dataset: TabularDS,
    model_name: str = "masked_time_series",
    epochs: int = 100,
    batch_size: int = 10_000,
    n_rows: int = None,
    early_stopping=None,
) -> dict:
    # best_params = None
    if n_rows is None:
        n_rows = len(dataset.X_train_numeric)
    if batch_size > n_rows:
        batch_size = n_rows
    summary_writer = SummaryWriter(
        "runs/" + dt.now().strftime("%Y-%m:%dT%H:%M:%S") + "_" + model_name
    )
    test_mi = create_masked_time_series_model_inputs(dataset, set="test")
    data = [
        create_masked_time_series_model_inputs(dataset, i, batch_size)
        for i in trange(0, n_rows, batch_size)
    ]
    pbar = trange(epochs)
    batch_counter = 0

    # Jit the functions
    masked_time_series_train_step_partial = partial(
        masked_time_series_train_step_no_jit, model
    )
    masked_time_series_eval_step_partial = partial(
        masked_time_series_eval_step_no_jit, model
    )
    masked_time_series_train_step = jax.jit(masked_time_series_train_step_partial)
    masked_time_series_eval_step = jax.jit(masked_time_series_eval_step_partial)
    best_test_loss = float("inf")
    for epoch in pbar:
        for mi in data:
            # mi = models.create_mi(dataset, i, batch_size)

            model_state, loss = masked_time_series_train_step(model_state, mi)
            train_loss_dict = masked_time_series_eval_step(model_state.params, mi)

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
        if epoch % 1 == 0:  # all logged to tensorboard
            test_loss_dict = masked_time_series_eval_step(model_state.params, test_mi)
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
                f"Train Loss: {train_loss_dict['total_loss'].item():.4f}, "
                + f"Test Loss: {test_loss_dict['total_loss'].item():.4f}"
            )
            if test_loss_dict["total_loss"].item() < best_test_loss:
                best_test_loss = test_loss_dict["total_loss"].item()
            if early_stopping is not None:
                improved, early_stopping = early_stopping.update(
                    test_loss_dict["total_loss"]
                )
                if improved:
                    best_params = model_state.params
                if early_stopping.should_stop:
                    print(
                        f"Early stopping triggered. Best loss: {best_test_loss:.4f},",
                        f"Test loss: {test_loss_dict['total_loss'].item():.4f}",
                    )
                    model_state = model_state.replace(params=best_params)
                    break
                    # best_model_state = model_state
                # all_losses.append(loss.item())
                # logger.add_scalar("Loss/train", loss.item(), i)

    # A100: 14.00it/s V100: 8.71it/s T4: 2.70it/s
    if "best_params" in locals():
        model_state = model_state.replace(params=best_params)
        return {
            "model_state": model_state,
            "losses": {
                "train_loss": train_loss_dict["total_loss"].item(),
                "test_loss": test_loss_dict["total_loss"].item(),
                "best_test_loss": best_test_loss,
            },
        }
    else:
        return {
            "model_state": model_state,
            "losses": {
                "train_loss": train_loss_dict["total_loss"].item(),
                "test_loss": test_loss_dict["total_loss"].item(),
                "best_test_loss": best_test_loss,
            },
        }
