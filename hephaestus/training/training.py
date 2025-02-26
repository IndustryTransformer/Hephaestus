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
    # Print debug info before processing
    print(f"Original input shape: {inputs.shape}")
    print(f"Original output shape: {outputs.shape}")

    # Check for sequence length mismatch
    if inputs.shape[1] != outputs.shape[1]:
        print(
            f"Sequence length mismatch: inputs={inputs.shape[1]}, outputs={outputs.shape[1]}"
        )
        # Use the minimum sequence length
        min_seq_len = min(inputs.shape[1], outputs.shape[1])
        inputs = inputs[:, :min_seq_len, :]
        outputs = outputs[:, :min_seq_len, ...]
        print(
            f"After sequence adjustment - inputs: {inputs.shape}, outputs: {outputs.shape}"
        )

    # Process inputs with offset
    inputs = inputs[:, :, inputs_offset:]
    batch_size, seq_len, feat_dim = inputs.shape
    tmp_null = torch.full(
        (batch_size, seq_len, inputs_offset), float("nan"), device=inputs.device
    )
    inputs = torch.cat([inputs, tmp_null], dim=2)
    nan_mask = torch.isnan(inputs)
    inputs = torch.where(nan_mask, torch.zeros_like(inputs), inputs)

    # Debug shapes after processing
    print(f"Processed input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"NaN mask shape: {nan_mask.shape}")

    # Handle feature dimension mismatch
    if outputs.dim() == inputs.dim() and outputs.shape[2] != inputs.shape[2]:
        print(
            f"Feature dimension mismatch: inputs={inputs.shape[2]}, outputs={outputs.shape[2]}"
        )
        min_features = min(outputs.shape[2], inputs.shape[2])
        nan_mask_adjusted = nan_mask[:, :, :min_features]
        masked_outputs = outputs.clone()
        masked_outputs[:, :, :min_features] = torch.where(
            nan_mask_adjusted,
            torch.zeros_like(outputs[:, :, :min_features]),
            outputs[:, :, :min_features],
        )
        return inputs, masked_outputs, nan_mask

    # Handle categorical outputs (different dimensionality)
    elif outputs.dim() == inputs.dim() + 1:
        nan_mask_expanded = nan_mask.unsqueeze(-1).expand_as(outputs)
        outputs = torch.where(nan_mask_expanded, torch.zeros_like(outputs), outputs)
    else:
        # Ensure mask size matches output features
        if nan_mask.shape[2] > outputs.shape[2]:
            nan_mask = nan_mask[:, :, : outputs.shape[2]]
        elif nan_mask.shape[2] < outputs.shape[2]:
            padding = torch.zeros(
                (
                    nan_mask.shape[0],
                    nan_mask.shape[1],
                    outputs.shape[2] - nan_mask.shape[2],
                ),
                dtype=torch.bool,
                device=nan_mask.device,
            )
            nan_mask = torch.cat([nan_mask, padding], dim=2)

        outputs = torch.where(nan_mask, torch.zeros_like(outputs), outputs)

    return inputs, outputs, nan_mask


def numeric_loss(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    Compute the MSE loss for numeric outputs, ignoring NaN values.

    Args:
        inputs: Input numeric values [batch_size, n_cols, seq_len]
        outputs: Predicted numeric values [batch_size, n_cols, seq_len]

    Returns:
        torch.Tensor: Scalar MSE loss
    """
    # Create mask for non-NaN values in inputs
    mask = ~torch.isnan(inputs)

    # Replace NaN with zeros for computation
    inputs_clean = torch.nan_to_num(inputs, nan=0.0)

    # Apply mask to both inputs and outputs
    masked_inputs = inputs_clean * mask.float()
    masked_outputs = outputs * mask.float()

    # Calculate MSE on non-NaN elements only
    mse = F.mse_loss(masked_outputs, masked_inputs, reduction="sum")

    # Normalize by the number of non-NaN values
    n_valid = mask.sum()
    if n_valid > 0:
        mse = mse / n_valid

    return mse


def categorical_loss(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross entropy loss for categorical outputs, ignoring padding.

    Args:
        inputs: Input categorical indices [batch_size, n_cat_cols, seq_len]
        outputs: Model logits [batch_size, n_cat_cols, seq_len, n_tokens]

    Returns:
        torch.Tensor: Scalar cross entropy loss
    """
    # Get dimensions
    batch_size, n_cols, seq_len = inputs.shape
    n_tokens = outputs.shape[-1]

    # Create mask for non-NaN values in inputs
    mask = ~torch.isnan(inputs)

    # Replace NaN with zeros for computation
    inputs_clean = torch.nan_to_num(inputs, nan=0.0).long()

    # Reshape for cross entropy loss
    inputs_flat = inputs_clean.reshape(-1)
    outputs_flat = outputs.reshape(-1, n_tokens)
    mask_flat = mask.reshape(-1)

    # Only compute loss on non-NaN elements
    valid_indices = mask_flat.nonzero(as_tuple=True)[0]

    if len(valid_indices) == 0:
        return torch.tensor(0.0, device=inputs.device)

    valid_inputs = inputs_flat[valid_indices]
    valid_outputs = outputs_flat[valid_indices]

    # Compute cross entropy loss
    loss = F.cross_entropy(valid_outputs, valid_inputs, reduction="sum")

    # Normalize by number of valid entries
    loss = loss / len(valid_indices)

    return loss


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
    """Perform a single training step.

    Args:
        model (TimeSeriesDecoder): The model to train.
        inputs (dict): Dictionary of inputs.
        optimizer: The optimizer to use.

    Returns:
        dict: Dictionary of loss values.
    """
    model.train()
    optimizer.zero_grad()

    res = model(
        numeric_inputs=inputs["numeric"], categorical_inputs=inputs["categorical"]
    )

    # Add debug prints before loss calculation
    print(f"Numeric inputs shape: {inputs['numeric'].shape}")
    print(f"Numeric outputs shape: {res['numeric'].shape}")

    numeric_loss_value = numeric_loss(inputs["numeric"], res["numeric"])
    categorical_loss_value = categorical_loss(inputs["categorical"], res["categorical"])
    loss = numeric_loss_value + categorical_loss_value

    # Check for NaN values before backprop
    if torch.isnan(loss).any():
        print("Warning: NaN detected in loss. Skipping backpropagation.")
        return {
            "loss": float("nan"),
            "numeric_loss": float("nan"),
            "categorical_loss": float("nan"),
        }

    loss.backward()

    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return {
        "loss": loss.item(),
        "numeric_loss": numeric_loss_value.item(),
        "categorical_loss": categorical_loss_value.item(),
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
