import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib.xla_extension.ArrayImpl as ArrayImpl
import optax
from data_utils import MLMModelInputs, MTMModelInputs, TabularDS
from flax.training import train_state
from jax import jit

import model.hephaestus as hp


def create_regression_state(
    params_key, mi: hp.TRMModelInputs, mtm_params=None, lr=0.01, device=None
):
    if device is None:
        device = jax.devices()[0]

    model = hp.TRM(dataset, d_model=64, n_heads=4)
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
    trm,
    mi: hp.TRMModelInputs,
):
    regression = trm.apply({"params": params}, mi.categorical_inputs, mi.numeric_inputs)

    loss = optax.squared_error(regression, mi.y).mean()

    return loss


@jit
def reg_train_step(state, mi):
    def loss_fn(params):
        return calculate_reg_loss(params, trm, mi)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss


@jit
def reg_eval_step(params, mi):
    return calculate_reg_loss(params, trm, mi)
