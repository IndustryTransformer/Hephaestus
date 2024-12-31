# import flax.linen as nn
import jax

# import jax.numpy as jnp
#  from jaxlib.xla_extension import ArrayImpl as ArrayImpl
# import model.hephaestus as hp
import optax
from flax.training import train_state

from .data_utils import MLMModelInputs
from .hephaestus import hp


def create_trm_state(
    params_key, mi: MLMModelInputs, mtm_params=None, lr=0.01, device=None
):
    if device is None:
        device = jax.devices()[0]

    model = hp.TRM(mi, d_model=64, n_heads=4)
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


def calculate_trm_loss(
    params,
    trm,
    mi: hp.TRMModelInputs,
):
    trmression = trm.apply({"params": params}, mi.categorical_inputs, mi.numeric_inputs)

    loss = optax.squared_error(trmression, mi.y).mean()

    return loss


@jax.jit
def trm_train_step(state, mi, trm):
    def loss_fn(params):
        return calculate_trm_loss(params, trm, mi)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss


@jax.jit
def trm_eval_step(params, mi, trm):
    return calculate_trm_loss(params, trm, mi)
