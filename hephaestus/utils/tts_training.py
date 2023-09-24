from datetime import datetime as dt
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jaxlib.xla_extension import ArrayImpl as ArrayImpl
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange

from ..models import models
from .data_utils import TabularDS, TRMModelInputs, create_trm_model_inputs


def create_tts_train_state(
    params_key: ArrayImpl,
    mi: TRMModelInputs,
    dataset: TabularDS,
    mtm_params=None,
    lr=0.01,
    device=None,
    n_heads=4,
):
    if device is None:
        device = jax.devices()[0]

    model = models.TRM(dataset, d_model=64, n_heads=n_heads)
    params = jax.device_put(
        model.init(params_key, mi.categorical_inputs, mi.numeric_inputs)["params"],
        device,
    )
    if mtm_params is not None:
        params["TabTransformer_0"] = jax.device_put(
            mtm_params["TabTransformer_0"], device
        )
    tx = optax.adam(learning_rate=lr)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def calculate_reg_loss(
    params,
    tts,
    mi: TRMModelInputs,
):
    tts_out = tts.apply({"params": params}, mi.categorical_inputs, mi.numeric_inputs)

    loss = optax.squared_error(tts_out, mi.y).mean()

    return loss


# @jax.jit
def tts_train_step_no_jit(model: models.TRM, state: train_state, mi: TRMModelInputs):
    """Remember to jit this function before using it in training"""

    def loss_fn(params):
        return calculate_reg_loss(params, model, mi)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss


# @jax.jit
def tts_eval_step_no_jit(model: models.TRM, params: dict, mi: TRMModelInputs):
    return calculate_reg_loss(params, model, mi)


def train_tts(
    model_state: train_state,
    model: models.TRM,
    dataset: TabularDS,
    epochs: int = 100,
    model_name: str = "TRM",
    batch_size=10_000,
    n_rows=None,
    early_stopping=None,
):
    """Train a Tabular Regression Model (TRM)"""
    if n_rows is None:
        n_rows = len(dataset.X_train_numeric)
    if batch_size > n_rows:
        batch_size = n_rows

    total_loss = []
    summary_writer = SummaryWriter(
        "runs/" + dt.now().strftime("%Y-%m:%dT%H:%M:%S") + "_" + model_name
    )

    reg_test_mi = create_trm_model_inputs(dataset, set="test")
    data = [
        create_trm_model_inputs(dataset, i, batch_size)
        for i in trange(0, n_rows, batch_size)
    ]
    # mi = models.create_mi(dataset)
    pbar = trange(epochs)
    batch_counter = 0

    # Jit the functions
    tts_train_step_partial = partial(tts_train_step_no_jit, model)
    tts_eval_step_partial = partial(tts_eval_step_no_jit, model)
    tts_train_step = jax.jit(tts_train_step_partial)
    tts_eval_step = jax.jit(tts_eval_step_partial)
    test_loss = tts_eval_step(model_state.params, reg_test_mi)

    best_test_loss = float("inf")
    for epoch in pbar:
        for mi in data:
            # mi = models.create_mi(dataset, i, batch_size)

            model_state, loss = tts_train_step(model_state, mi)
            # train_loss_dict = tts_eval_step(tts_state.params, mi)

            total_loss.append(loss.item())
            # Train Loss
            summary_writer.add_scalar(
                "TrainLoss/tts_total",
                np.array(loss),
                batch_counter,
            )
            batch_counter += 1
            # Test Loss
        if epoch % 1 == 0:  # evaluate every epoch
            test_loss = tts_eval_step(model_state.params, reg_test_mi)
            summary_writer.add_scalar(
                "TestLoss/tts_total",
                np.array(test_loss),
                batch_counter,
            )
        pbar.set_description(
            f"Train Loss: {loss.item():,.0f}, Test Loss: {test_loss.item():,.0f}"
        )
        # print(f"Test Loss: {test_loss.item():,.0f}, Best Test Loss: {best_test_loss}")
        if early_stopping is not None:
            improved, early_stopping = early_stopping.update(test_loss)
            if improved:
                best_params = model_state.params
            if early_stopping.should_stop:
                print("Early stopping triggered")
                break

        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            # best_model_state = model_state

    total_loss = jnp.array(total_loss)
    if "best_params" in locals():
        model_state = model_state.replace(params=best_params)
        return {
            "tts_state": model_state,
            "losses": {
                "train_loss": loss.item(),
                "test_loss": test_loss.item(),
                "best_test_loss": best_test_loss,
            },
        }
    else:
        # categorical_loss = jnp.array(categorical_loss)
        #  # A100: 14.00it/s V100: 8.71it/s T4: 2.70it/s
        return {
            "tts_state": model_state,
            "losses": {
                "train_loss": loss.item(),
                "test_loss": test_loss.item(),
                "best_test_loss": best_test_loss,
            },
        }
