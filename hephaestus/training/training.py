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
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    # Apply offset adjustment to inputs and outputs
    y_true, y_pred, _ = add_input_offsets(y_true, y_pred, inputs_offset=1)

    # Remove NaN values from the loss calculation
    mask = ~torch.isnan(y_true)
    if mask.any():
        loss = F.mse_loss(y_pred[mask], y_true[mask], reduction="mean")
        return loss
    return torch.tensor(0.0, device=y_true.device, dtype=torch.float32)


def categorical_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate categorical loss between true and predicted values."""
    # Keep predictions as float32 for softmax operation
    y_pred = y_pred.to(torch.float32)
    
    # Convert target to long type but handle it carefully
    y_true = y_true.to(torch.long)  # Convert to long (int64) for class indices
    
    # Apply offset adjustment to inputs and outputs
    y_true, y_pred, _ = add_input_offsets(y_true, y_pred, inputs_offset=1)

    mask = ~torch.isnan(y_true)
    if mask.any():
        # Apply log_softmax to predictions
        log_probs = F.log_softmax(y_pred[mask].reshape(-1, y_pred.size(-1)), dim=-1)
        
        # Get targets and make sure they're valid indices
        targets = y_true[mask].reshape(-1).long()
        
        # Important! Verify targets are in valid range for nll_loss
        num_classes = y_pred.size(-1)
        valid_targets_mask = (targets >= 0) & (targets < num_classes)
        
        # Only use valid targets
        if valid_targets_mask.any():
            valid_log_probs = log_probs[valid_targets_mask]
            valid_targets = targets[valid_targets_mask]
            loss = F.nll_loss(valid_log_probs, valid_targets, reduction="mean")
            return loss
        else:
            # Return zero loss if no valid targets found
            return torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)
            
    return torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)

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
    scheduler = CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-6)

    return optimizer, scheduler


def train_step(
    model, inputs, optimizer=None, scheduler=None, accumulating_gradients=False
):
    """Single training step with simplified gradient handling."""
    # Only zero gradients if not accumulating and optimizer is provided
    if optimizer is not None and not accumulating_gradients:
        optimizer.zero_grad()

    # Forward pass
    outputs = model(
        numeric_inputs=inputs.numeric, categorical_inputs=inputs.categorical
    )

    # Calculate losses
    numeric_loss_val = numeric_loss(inputs.numeric, outputs.numeric)
    categorical_loss_val = (
        categorical_loss(inputs.categorical, outputs.categorical)
        if outputs.categorical is not None
        else torch.tensor(0.0, device=inputs.numeric.device)
    )

    # Combine losses
    loss = numeric_loss_val + categorical_loss_val

    # Backward pass
    loss.backward()

    # Update parameters if not accumulating gradients
    if optimizer is not None and not accumulating_gradients:
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
            numeric_inputs=inputs.numeric, categorical_inputs=inputs.categorical
        )

        numeric_loss_value = numeric_loss(inputs.numeric, res.numeric)
        categorical_loss_value = categorical_loss(inputs.categorical, res.categorical)
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
    model_output, batch
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the combined loss for a batch.

    Args:
        model_output: TimeSeriesOutput with 'numeric' and 'categorical' tensors
        batch: TimeSeriesInputs with input 'numeric' and 'categorical' tensors

    Returns:
        total_loss: Combined loss value
        component_losses: Dictionary with individual loss components
    """
    losses = {}

    # Numeric loss
    if hasattr(model_output, "numeric") and hasattr(batch, "numeric"):
        num_loss = numeric_loss(batch.numeric, model_output.numeric)
        losses["numeric_loss"] = num_loss

    # Categorical loss
    if (
        hasattr(model_output, "categorical")
        and hasattr(batch, "categorical")
        and model_output.categorical is not None
    ):
        cat_loss = categorical_loss(batch.categorical, model_output.categorical)
        losses["categorical_loss"] = cat_loss

    # Total loss is the sum of component losses
    total_loss = sum(losses.values())

    return total_loss, losses
