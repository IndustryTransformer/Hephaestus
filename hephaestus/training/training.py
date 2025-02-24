import jax.numpy as jnp
import optax
from flax import nnx

from hephaestus.models.models import TimeSeriesDecoder


def add_input_offsets(
    inputs: jnp.array, outputs: jnp.array, inputs_offset: int = 1
) -> jnp.array:
    """Add offsets to inputs and apply mask to outputs.

    Args:
        inputs (jnp.array): Input array.
        outputs (jnp.array): Output array.
        inputs_offset (int, optional): Offset for inputs. Defaults to 1.

    Returns:
        tuple: Tuple containing modified inputs, outputs, and nan mask.
    """
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


def numeric_loss(inputs, outputs, input_offset: int = 1):
    """Calculate numeric loss.

    Args:
        inputs (jnp.array): Input array.
        outputs (jnp.array): Output array.
        input_offset (int, optional): Offset for inputs. Defaults to 1.

    Returns:
        jnp.array: Calculated loss.
    """
    inputs, outputs, nan_mask = add_input_offsets(
        inputs=inputs, outputs=outputs, inputs_offset=input_offset
    )
    raw_loss = jnp.abs(outputs - inputs)
    masked_loss = jnp.where(nan_mask, 0.0, raw_loss)
    loss = masked_loss.sum() / (~nan_mask).sum()
    return loss


def categorical_loss(inputs, outputs, input_offset: int = 1):
    """Calculate categorical loss.

    Args:
        inputs (jnp.array): Input array.
        outputs (jnp.array): Output array.
        input_offset (int, optional): Offset for inputs. Defaults to 1.

    Returns:
        jnp.array: Calculated loss.
    """
    inputs, outputs, nan_mask = add_input_offsets(
        inputs=inputs, outputs=outputs, inputs_offset=input_offset
    )
    inputs = inputs.astype(jnp.int32)
    raw_loss = optax.softmax_cross_entropy_with_integer_labels(outputs, inputs)
    masked_loss = jnp.where(nan_mask, 0.0, raw_loss).mean()
    return masked_loss


def create_train_step(
    model: TimeSeriesDecoder, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric
):
    """Create a training step function.

    Args:
        model (TimeSeriesDecoder): The model to train.
        optimizer (nnx.Optimizer): The optimizer to use.
        metrics (nnx.MultiMetric): The metrics to track.

    Returns:
        function: The training step function.
    """

    @nnx.jit
    def train_step(
        model: TimeSeriesDecoder,
        inputs: dict,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
    ):
        """Perform a single training step.

        Args:
            model (TimeSeriesDecoder): The model to train.
            inputs (dict): Dictionary of inputs.
            optimizer (nnx.Optimizer): The optimizer to use.
            metrics (nnx.MultiMetric): The metrics to track.
        """

        def loss_fn(model):
            res = model(
                numeric_inputs=inputs["numeric"],
                categorical_inputs=inputs["categorical"],
                deterministic=False,
            )

            numeric_loss_value = numeric_loss(inputs["numeric"], res["numeric_out"])
            categorical_loss_value = categorical_loss(
                inputs["categorical"], res["categorical_out"]
            )
            loss = numeric_loss_value + categorical_loss_value
            return loss, (numeric_loss_value, categorical_loss_value)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, (numeric_loss_value, categorical_loss_value)), grads = grad_fn(model)
        metrics.update(
            loss=loss,
            numeric_loss=numeric_loss_value,
            categorical_loss=categorical_loss_value,
        )

        optimizer.update(grads)

    return train_step


def create_metric_history():
    """Create a dictionary to store metric history.

    Returns:
        dict: Dictionary to store metric history.
    """
    return {
        "loss": [],
        "numeric_loss": [],
        "categorical_loss": [],
    }


def create_optimizer(
    model,
    learning_rate,
    momentum: float = 0.4,
    warmup_steps: int = 500,
    clip_norm: float = 1.0,
):
    """Create an optimizer with a learning rate schedule.

    Args:
        model: The model to optimize.
        learning_rate (float): The learning rate.
        momentum (float, optional): Momentum for the optimizer. Defaults to 0.4.
        warmup_steps (int, optional): Number of warmup steps. Defaults to 500.
        clip_norm (float, optional): Gradient clipping norm. Defaults to 1.0.

    Returns:
        nnx.Optimizer: The created optimizer.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=10000,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(
                clip_norm
            ),  # clip gradients to avoid exploding gradients
            optax.adamw(learning_rate=learning_rate, b1=momentum),
            optax.scale_by_schedule(schedule),
        ),
    )
    return optimizer


def create_metrics():
    """Create metrics for tracking training progress.

    Returns:
        nnx.MultiMetric: The created metrics.
    """
    return nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        categorical_loss=nnx.metrics.Average("categorical_loss"),
        numeric_loss=nnx.metrics.Average("numeric_loss"),
    )
