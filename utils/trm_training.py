from datetime import datetime as dt

import jax
import jax.numpy as jnp
import jaxlib.xla_extension.ArrayImpl as ArrayImpl
import numpy as np
import optax
from data_utils import TabularDS, TRMModelInputs
from flax.training import train_state
from jax import jit
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import trange

import model.hephaestus as hp


def create_regression_state(
    params_key,
    mi: hp.TRMModelInputs,
    dataset: TabularDS,
    mtm_params=None,
    lr=0.01,
    device=None,
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
def reg_train_step(state: train_state, model: hp.TRM, mi: TRMModelInputs):
    def loss_fn(params):
        return calculate_reg_loss(params, model, mi)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss


@jit
def reg_eval_step(params: dict, mi: TRMModelInputs, model: hp.TRM):
    return calculate_reg_loss(params, model, mi)


# regression_root_key = random.PRNGKey(0)
# reg_main_key, reg_params_key, reg_dropout_key = random.split(root_key, 3)

# reg_mi = hp.create_trm_model_inputs(dataset, idx=0, batch_size=3, set="train")
# trm_state = create_regression_state(params_key, reg_mi, mtm_state.params)


def train_trm(
    trm_state: train_state,
    model: hp.TRM,
    dataset: TabularDS,
    epochs: int = 100,
    model_name: str = "TRM",
):
    total_loss = []
    summary_writer = SummaryWriter(
        "runs/" + dt.now().strftime("%Y-%m:%dT%H:%M:%S") + "_" + model_name
    )

    reg_test_mi = hp.create_trm_model_inputs(dataset, set="test")
    batch_size = 10_000
    data = [
        hp.create_trm_model_inputs(dataset, i, batch_size)
        for i in trange(0, len(dataset.X_train_numeric), batch_size)
    ]
    test_loss = reg_eval_step(trm_state.params, reg_test_mi)
    # mi = hp.create_mi(dataset)
    pbar = trange(epochs)
    batch_counter = 0
    for epoch in pbar:
        for mi in data:
            # mi = hp.create_mi(dataset, i, batch_size)

            trm_state, loss = reg_train_step(trm_state, model, mi)
            # train_loss_dict = trm_eval_step(trm_state.params, mi)

            total_loss.append(loss.item())
            # Train Loss
            summary_writer.add_scalar(
                "TrainLoss/regression_total",
                np.array(loss),
                batch_counter,
            )
            batch_counter += 1
            # Test Loss
            if epoch % 10 == 0:
                test_loss = reg_eval_step(trm_state.params, reg_test_mi)
                summary_writer.add_scalar(
                    "TestLoss/regression_total",
                    np.array(test_loss),
                    batch_counter,
                )
            pbar.set_description(
                f"Train Loss: {loss.item():,.0f}, Test Loss: {test_loss.item():,.0f}"
            )

    total_loss = jnp.array(total_loss)
    # categorical_loss = jnp.array(categorical_loss)
    #  # A100: 14.00it/s V100: 8.71it/s T4: 2.70it/s
    return {"trm_state": trm_state, "total_loss": total_loss}
