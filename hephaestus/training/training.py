import jax.numpy as jnp
import optax
from flax import nnx


def add_input_offsets(
    inputs: jnp.array, outputs: jnp.array, inputs_offset: int = 1
) -> jnp.array:
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
    # outputs = jnp.where(nan_mask, jnp.zeros_like(outputs), outputs)

    # return inputs, outputs, nan_mask


def numeric_loss(inputs, outputs, input_offset: int = 1):
    # print("Doing Numeric Loss")
    inputs, outputs, nan_mask = add_input_offsets(
        inputs=inputs, outputs=outputs, inputs_offset=input_offset
    )
    # print(f"Numeric: {inputs.shape=}, {outputs.shape=}, {nan_mask.shape=}")
    # TODO make loss SSL for values greater than 0.5 and MSE for values less than 0.5
    raw_loss = jnp.abs(outputs - inputs)
    masked_loss = jnp.where(nan_mask, 0.0, raw_loss)
    # print(f"{masked_loss.shape=}, {nan_mask.shape=}")
    loss = masked_loss.sum() / (~nan_mask).sum()
    return loss


def categorical_loss(inputs, outputs, input_offset: int = 1):
    # print("Doing Categorical Loss")
    inputs, outputs, nan_mask = add_input_offsets(
        inputs=inputs, outputs=outputs, inputs_offset=input_offset
    )
    # print(f"Categorical: {inputs.shape=}, {outputs.shape=}, {nan_mask.shape=}")
    inputs = inputs.astype(jnp.int32)
    # print(f"Categorical Loss Shapes: {inputs.shape=}, {outputs.shape=}")
    raw_loss = optax.softmax_cross_entropy_with_integer_labels(outputs, inputs)
    masked_loss = jnp.where(nan_mask, 0.0, raw_loss).mean()
    return masked_loss


# @partial(jax.jit, static_argnums=(0, 2, 3))
# def train_step(
#     model: hp.TimeSeriesDecoder,
#     inputs: dict,
#     optimizer: nnx.Optimizer,
#     metrics: nnx.MultiMetric,
# ):
#     def loss_fn(model):
#         res = model(
#             numeric_inputs=inputs["numeric"],
#             categorical_inputs=inputs["categorical"],
#             deterministic=False,
#         )

#         numeric_loss_value = numeric_loss(inputs["numeric"], res["numeric_out"])
#         categorical_loss_value = categorical_loss(
#             inputs["categorical"], res["categorical_out"]
#         )
#         loss = numeric_loss_value + categorical_loss_value
#         return loss, (numeric_loss_value, categorical_loss_value)

#     grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
#     # grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
#     # loss, grads = grad_fn(model)
#     (loss, (numeric_loss_value, categorical_loss_value)), grads = grad_fn(model)
#     metrics.update(
#         loss=loss,
#         numeric_loss=numeric_loss_value,
#         categorical_loss=categorical_loss_value,
#     )

#     optimizer.update(grads)


def create_metric_history():
    return {
        "loss": [],
        "numeric_loss": [],
        "categorical_loss": [],
    }


# Then create the optimizer with the schedule
def create_optimizer(
    model, learning_rate, momentum: float = 0.4, warmup_steps: int = 500
):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=10000,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.adamw(learning_rate=learning_rate, b1=momentum),
            optax.scale_by_schedule(schedule),
        ),
    )
    return optimizer


def create_metrics():
    return nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        categorical_loss=nnx.metrics.Average("categorical_loss"),
        numeric_loss=nnx.metrics.Average("numeric_loss"),
    )
