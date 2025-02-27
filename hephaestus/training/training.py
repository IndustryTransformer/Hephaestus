from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def add_input_offsets(inputs, outputs, inputs_offset=1):
    """Add offsets to inputs and apply mask to outputs (PyTorch version).

    Args:
        inputs (torch.Tensor): Input tensor.
        outputs (torch.Tensor): Output tensor.
        inputs_offset (int, optional): Offset for inputs. Defaults to 1.

    Returns:
        tuple: Tuple containing modified inputs, outputs, and nan mask.
    """

    # Process inputs with offset
    batch_size, seq_len, feat_dim = inputs.shape

    # Process inputs with offset
    inputs_shifted = inputs[:, :, inputs_offset:]
    tmp_null = torch.full(
        (batch_size, seq_len, inputs_offset), float("nan"), device=inputs.device
    )
    inputs_padded = torch.cat([inputs_shifted, tmp_null], dim=2)

    # Create mask and handle NaNs consistently
    nan_mask = torch.isnan(inputs_padded)
    inputs_clean = torch.where(nan_mask, torch.zeros_like(inputs_padded), inputs_padded)

    # Mask outputs consistently with the JAX version
    if outputs.dim() > inputs_clean.dim():
        # Handle categorical outputs
        nan_mask_expanded = nan_mask.unsqueeze(-1).expand(-1, -1, -1, outputs.shape[-1])
        outputs_masked = torch.where(
            nan_mask_expanded, torch.zeros_like(outputs), outputs
        )
    else:
        # Handle numeric outputs
        outputs_masked = torch.where(nan_mask, torch.zeros_like(outputs), outputs)

    return inputs_clean, outputs_masked, nan_mask


def numeric_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate numeric loss between true and predicted values."""
    # Ensure float32 dtype
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)

    # Remove NaN values from the loss calculation
    mask = ~torch.isnan(y_true)
    if mask.any():
        return F.mse_loss(y_pred[mask], y_true[mask], reduction="mean")
    return torch.tensor(0.0, device=y_true.device, dtype=torch.float32)


def categorical_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate categorical loss between true and predicted values."""
    # Ensure float32 dtype
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)

    # Remove NaN values from the loss calculation
    mask = ~torch.isnan(y_true)
    if mask.any():
        return F.cross_entropy(
            y_pred[mask].reshape(-1, y_pred.size(-1)),
            y_true[mask].reshape(-1).long(),
            reduction="mean",
        )
    return torch.tensor(0.0, device=y_true.device, dtype=torch.float32)


def create_optimizer(
    model, learning_rate, momentum=0.9, weight_decay=1e-5, warmup_steps=500
):
    """Create an optimizer with a learning rate schedule (PyTorch version).

    Args:
        model: The model to optimize.
        learning_rate (float): The learning rate.
        momentum (float, optional): Beta1 parameter for AdamW. Defaults to 0.9.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        warmup_steps (int, optional): Number of warmup steps. Defaults to 500.

    Returns:
        tuple: The created optimizer and scheduler.
    """
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(momentum, 0.999),
        weight_decay=weight_decay,
    )

    # Create a cosine learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)

    return optimizer, scheduler


def train_step(model, inputs, optimizer):
    """Single training step."""
    optimizer.zero_grad()

    # Ensure all inputs are float32
    # for key in inputs: # Error here?
    #     inputs[key] = inputs[key].to(torch.float32)

    outputs = model(
        numeric_inputs=inputs["numeric"], categorical_inputs=inputs["categorical"]
    )
    offset_numeric, offset_outputs_numeric, _ = add_input_offsets(
        inputs["numeric"], outputs["numeric"], inputs_offset=1
    )
    # Calculate losses
    numeric_loss_val = numeric_loss(inputs["numeric"], outputs["numeric"])
    categorical_loss_val = (
        categorical_loss(inputs["categorical"], outputs["categorical"])
        if outputs["categorical"] is not None
        else 0.0
    )

    # Ensure loss is float32
    if isinstance(categorical_loss_val, (int, float)):
        categorical_loss_val = torch.tensor(
            categorical_loss_val, dtype=torch.float32, device=numeric_loss_val.device
        )

    loss = numeric_loss_val + categorical_loss_val

    # Check for NaN values
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        # Print gradient norms to debug
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        print(f"Gradient norm: {total_norm}")

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    else:
        print(f"Skipping update due to bad loss value: {loss.item()}")

    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "numeric_loss": numeric_loss_val.item(),
        "categorical_loss": categorical_loss_val.item(),
    }


def eval_step(model, inputs):
    """Perform a single evaluation step.

    Args:
        model (TimeSeriesDecoder): The model to evaluate.
        inputs (dict): Dictionary of inputs.

    Returns:
        dict: Dictionary of loss values.
    """
    model.eval()
    with torch.no_grad():
        res = model(
            numeric_inputs=inputs["numeric"], categorical_inputs=inputs["categorical"]
        )

        numeric_loss_value = numeric_loss(inputs["numeric"], res["numeric"])
        categorical_loss_value = categorical_loss(
            inputs["categorical"], res["categorical"]
        )
        loss = numeric_loss_value + categorical_loss_value

    return {
        "loss": loss.item(),
        "numeric_loss": numeric_loss_value.item(),
        "categorical_loss": categorical_loss_value.item(),
    }


def create_metric_history():
    """Create a dictionary to store metric history.

    Returns:
        dict: Dictionary to store metric history.
    """
    return {
        "train_loss": [],
        "train_numeric_loss": [],
        "train_categorical_loss": [],
        "val_loss": [],
        "val_numeric_loss": [],
        "val_categorical_loss": [],
    }


def compute_batch_loss(
    model_output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the combined loss for a batch.

    Args:
        model_output: Dictionary with 'numeric' and 'categorical' tensors
        batch: Dictionary with input 'numeric' and 'categorical' tensors

    Returns:
        total_loss: Combined loss value
        component_losses: Dictionary with individual loss components
    """
    losses = {}

    # Numeric loss
    if "numeric" in model_output and "numeric" in batch:
        num_loss = numeric_loss(batch["numeric"], model_output["numeric"])
        losses["numeric_loss"] = num_loss

    # Categorical loss
    if "categorical" in model_output and "categorical" in batch:
        cat_loss = categorical_loss(batch["categorical"], model_output["categorical"])
        losses["categorical_loss"] = cat_loss

    # Total loss is the sum of component losses
    total_loss = sum(losses.values())

    return total_loss, losses
