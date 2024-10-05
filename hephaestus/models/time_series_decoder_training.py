import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import random


def clip_gradients(gradients, max_norm):
    total_norm = jnp.sqrt(sum(jnp.sum(jnp.square(grad)) for grad in gradients.values()))
    scale = max_norm / (total_norm + 1e-6)
    clipped_gradients = jax.tree_map(
        lambda grad: jnp.where(total_norm > max_norm, grad * scale, grad), gradients
    )
    return clipped_gradients


def add_input_offsets(
    inputs: jnp.array, outputs: jnp.array, inputs_offset: int = 1
) -> jnp.array:
    print("InputOffset:", inputs_offset, type(inputs_offset))
    inputs = inputs[:, :, inputs_offset:]
    tmp_null = jnp.full((inputs.shape[0], inputs.shape[1], inputs_offset), jnp.nan)
    inputs = jnp.concatenate([inputs, tmp_null], axis=2)
    nan_mask = jnp.isnan(inputs)
    inputs = jnp.where(nan_mask, jnp.zeros_like(inputs), inputs)
    # Add ext
    if outputs.ndim == inputs.ndim + 1:
        nan_mask_expanded = jnp.expand_dims(nan_mask, axis=-1)
        nan_mask_expanded = jnp.broadcast_to(nan_mask_expanded, outputs.shape)
    else:
        nan_mask_expanded = nan_mask

    # Apply mask to outputs
    outputs = jnp.where(nan_mask_expanded, jnp.zeros_like(outputs), outputs)

    return inputs, outputs, nan_mask
    outputs = jnp.where(nan_mask, jnp.zeros_like(outputs), outputs)

    return inputs, outputs, nan_mask


def numeric_loss(inputs, outputs, input_offset: int = 1):
    inputs, outputs, nan_mask = add_input_offsets(inputs, outputs, input_offset)
    # TODO make loss SSL for values greater than 0.5 and MSE for values less than 0.5
    raw_loss = jnp.abs(outputs - inputs)
    masked_loss = jnp.where(nan_mask, 0.0, raw_loss)
    loss = masked_loss.sum() / (~nan_mask).sum()
    return loss


def categorical_loss(inputs, outputs, input_offset: int = 1):
    inputs, outputs, nan_mask = add_input_offsets(inputs, outputs, input_offset)
    inputs = inputs.astype(jnp.int32)
    raw_loss = optax.softmax_cross_entropy_with_integer_labels(outputs, inputs)
    masked_loss = jnp.where(nan_mask, 0.0, raw_loss).mean()
    return masked_loss


def base_loss(
    numeric_inputs,
    categorical_inputs,
    outputs,
    input_offset: int = 1,
):
    numeric_out = outputs["numeric_out"]
    categorical_out = outputs["categorical_out"]
    numeric = numeric_loss(numeric_inputs, numeric_out, input_offset)
    categorical = categorical_loss(categorical_inputs, categorical_out, input_offset)
    return numeric + categorical


def calculate_loss_inner(
    params,
    state,
    numeric_inputs,
    categorical_inputs,
    dropout_key,
    causal_mask: bool = True,
    input_offset: int = 1,
):
    outputs = state.apply_fn(
        {"params": params},
        # hp.mask_tensor(inputs, dataset, prng_key=mask_key),
        numeric_inputs=numeric_inputs,
        categorical_inputs=categorical_inputs.astype(jnp.int32),
        rngs={"dropout": dropout_key},
        deterministic=False,
        causal_mask=causal_mask,
    )
    loss = base_loss(
        numeric_inputs=numeric_inputs,
        categorical_inputs=categorical_inputs,
        outputs=outputs,
        input_offset=input_offset,
    )
    # Create mask for nan inputs

    return loss


# @jax.jit
def train_step(
    state: train_state.TrainState,
    numeric_inputs,
    categorical_inputs,
    base_key,
    input_offset: int = 1,
):
    dropout_key, mask_key, new_key = jax.random.split(base_key, 3)
    print("InputOffset:", input_offset, type(input_offset))

    def calculate_loss(params):
        return calculate_loss_inner(
            params,
            state,
            numeric_inputs=numeric_inputs,
            categorical_inputs=categorical_inputs,
            dropout_key=dropout_key,
            causal_mask=True,
            input_offset=input_offset,
        )

    def loss_fn(params):
        return calculate_loss(params)

    grad_fn = jax.value_and_grad(loss_fn)

    # (loss, individual_losses), grad = grad_fn(state.params)
    loss, grad = grad_fn(state.params)
    # grad = replace_nans(grad)
    # grad = clip_gradients(grad, 1.0)
    state = state.apply_gradients(grads=grad)

    return state, loss, new_key


def evaluate(params, state, inputs, causal_mask: bool = True):
    outputs = state.apply_fn(
        {"params": params},
        # hp.mask_tensor(inputs, dataset, prng_key=mask_key),
        inputs,
        deterministic=True,
        causal_mask=causal_mask,
    )
    loss = base_loss(inputs, outputs)
    return loss


@jax.jit
def eval_step(
    state: train_state.TrainState, numeric_inputs, categorical_inputs, base_key
):
    # causal_mask=True
    mask_key, dropout_key, new_key = jax.random.split(base_key, 3)

    def calculate_loss(params):
        return calculate_loss_inner(
            params,
            state,
            numeric_inputs=numeric_inputs,
            categorical_inputs=categorical_inputs,
            dropout_key=dropout_key,
            causal_mask=True,
        )

    def loss_fn(params):
        return calculate_loss(params)

    loss = loss_fn(state.params)
    return loss, new_key


def create_train_state(model, prng, batch, lr: float):
    init_key, dropout_key = random.split(prng)
    params = model.init(
        {"params": init_key, "dropout": dropout_key},
        batch["numeric"],
        batch["categorical"],
        deterministic=False,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(0.4), optax.adam(lr))
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optimizer,
    )
