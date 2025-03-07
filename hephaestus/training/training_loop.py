import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from hephaestus.models import tabular_collate_fn
from hephaestus.training.training import (
    compute_batch_loss,
    create_metric_history,
    create_optimizer,
)


def train_model(
    model,
    train_dataset,
    val_dataset,
    batch_size=32,
    epochs=100,
    learning_rate=1e-4,
    log_dir="runs/experiment",
    save_dir="models",
    device=None,
    num_workers=0,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    explosion_threshold=10.0,
    max_explosions_per_epoch=5,
    writer: SummaryWriter = None,
):
    """Train a time series model with tensorboard logging.

    Args:
        model: The PyTorch model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        log_dir: Directory for tensorboard logs
        save_dir: Directory to save model checkpoints
        device: Device to use for training (None for auto-detection)
        num_workers: Number of worker processes for data loading
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum norm for gradients
        explosion_threshold: Threshold above which a gradient is considered exploding
        max_explosions_per_epoch: Maximum gradient explosions allowed per epoch
        writer: Optional tensorboard writer for logging

    Returns:
        dict: Training history
    """
    # Set up device
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)
    model.train()
    for param in model.parameters():
        param.data = param.data.to(torch.float32)

    # Create data loaders - ALWAYS use tabular_collate_fn for TimeSeriesDS
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=tabular_collate_fn,  # Always use our custom collation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=tabular_collate_fn,  # Always use our custom collation
    )

    # Get all model parameters
    all_params = list(model.parameters())
    categorical_params = []

    # Separate the categorical parameters
    for name, module in model.named_modules():
        if (
            "categorical" in name.lower()
            and isinstance(module, nn.Module)
            and not isinstance(module, nn.ModuleList)
            and not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleDict)
            and not isinstance(module, nn.Parameter)
        ):
            for param_name, param in module.named_parameters():
                # Only add if directly owned by this module (avoid duplicates)
                if "." not in param_name:
                    categorical_params.append(param)

    # Use parameter id() to filter instead of direct tensor comparison
    categorical_param_ids = {id(p) for p in categorical_params}
    numeric_params = [p for p in all_params if id(p) not in categorical_param_ids]

    # Create optimizers with different learning rates
    numeric_optimizer = torch.optim.AdamW(numeric_params, lr=learning_rate)
    categorical_optimizer = torch.optim.AdamW(
        categorical_params, lr=learning_rate * 1.5
    )

    # Add standard optimizer/scheduler as fallback
    optimizer, scheduler = create_optimizer(model, learning_rate)

    # Add patience for early stopping
    patience = 5
    plateau_counter = 0
    best_val_loss = float("inf")
    explosion_count = 0

    # Create TensorBoard writer
    if writer is None:
        writer = SummaryWriter(log_dir, flush_secs=10)
    print(f"TensorBoard log directory: {log_dir}")

    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize metric history
    history = create_metric_history()

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_numeric_loss = 0.0
        train_categorical_loss = 0.0
        batch_count = 0

        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for i, batch in enumerate(train_iterator):
            # Move batch to device
            batch = batch.to(device)

            # Forward pass
            outputs = model(
                numeric_inputs=batch.numeric, categorical_inputs=batch.categorical
            )

            # Compute losses
            total_loss, component_losses = compute_batch_loss(outputs, batch)

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / gradient_accumulation_steps
            scaled_loss.backward()

            # Update training metrics
            train_loss += total_loss.item()
            train_numeric_loss += component_losses.get("numeric_loss", 0.0)
            train_categorical_loss += component_losses.get("categorical_loss", 0.0)
            batch_count += 1

            # Update progress bar
            train_iterator.set_postfix(
                {
                    "loss": f"{train_loss/batch_count:.4f}",
                    "num_loss": f"{train_numeric_loss/batch_count:.4f}",
                    "cat_loss": f"{train_categorical_loss/batch_count:.4f}",
                }
            )

            # Perform optimizer step after accumulating gradients
            if (i + 1) % gradient_accumulation_steps == 0:
                # Check for exploding gradients
                numeric_grad_norm = torch.nn.utils.clip_grad_norm_(
                    numeric_params, max_grad_norm
                )
                categorical_grad_norm = torch.nn.utils.clip_grad_norm_(
                    categorical_params, max_grad_norm
                )

                # Detect gradient explosion
                if (
                    numeric_grad_norm > explosion_threshold
                    or categorical_grad_norm > explosion_threshold
                ):
                    explosion_count += 1
                    if explosion_count > max_explosions_per_epoch:
                        print(
                            f"Too many gradient explosions ({explosion_count}), reducing learning rate"
                        )
                        for param_group in numeric_optimizer.param_groups:
                            param_group["lr"] *= 0.5
                        for param_group in categorical_optimizer.param_groups:
                            param_group["lr"] *= 0.5
                        explosion_count = 0

                # Optimizer step
                numeric_optimizer.step()
                categorical_optimizer.step()
                numeric_optimizer.zero_grad()
                categorical_optimizer.zero_grad()

                # Log batch-level metrics every 50 batches
                if (i + 1) % 50 == 0:
                    global_step = epoch * len(train_loader) + i
                    writer.add_scalars(
                        "Batch/Loss",
                        {
                            "train": total_loss.item(),
                            "numeric": component_losses.get("numeric_loss", 0.0),
                            "categorical": component_losses.get(
                                "categorical_loss", 0.0
                            ),
                        },
                        global_step,
                    )
                    writer.flush()  # Force writing to disk

        # Calculate average training metrics
        avg_train_loss = train_loss / batch_count
        avg_train_numeric_loss = train_numeric_loss / batch_count
        avg_train_categorical_loss = train_categorical_loss / batch_count

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_numeric_loss = 0.0
        val_categorical_loss = 0.0
        val_batch_count = 0

        val_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
        with torch.no_grad():
            for batch in val_iterator:
                # Move batch to device
                batch = batch.to(device)

                # Forward pass
                outputs = model(
                    numeric_inputs=batch.numeric, categorical_inputs=batch.categorical
                )

                # Compute losses
                total_loss, component_losses = compute_batch_loss(outputs, batch)

                # Update validation metrics
                val_loss += total_loss.item()
                val_numeric_loss += component_losses.get("numeric_loss", 0.0)
                val_categorical_loss += component_losses.get("categorical_loss", 0.0)
                val_batch_count += 1

                # Update progress bar
                val_iterator.set_postfix(
                    {
                        "val_loss": f"{val_loss/val_batch_count:.4f}",
                        "val_num_loss": f"{val_numeric_loss/val_batch_count:.4f}",
                        "val_cat_loss": f"{val_categorical_loss/val_batch_count:.4f}",
                    }
                )

        # Calculate average validation metrics
        avg_val_loss = val_loss / val_batch_count
        avg_val_numeric_loss = val_numeric_loss / val_batch_count
        avg_val_categorical_loss = val_categorical_loss / val_batch_count

        # Calculate epoch time before logging
        epoch_time = time.time() - start_time

        # TensorBoard logging
        writer.add_scalars(
            "Loss", {"train": avg_train_loss, "val": avg_val_loss}, epoch
        )

        writer.add_scalars(
            "Numeric_Loss",
            {"train": avg_train_numeric_loss, "val": avg_val_numeric_loss},
            epoch,
        )

        writer.add_scalars(
            "Categorical_Loss",
            {"train": avg_train_categorical_loss, "val": avg_val_categorical_loss},
            epoch,
        )

        # Log learning rates
        writer.add_scalar(
            "Learning_Rate/numeric", numeric_optimizer.param_groups[0]["lr"], epoch
        )
        writer.add_scalar(
            "Learning_Rate/categorical",
            categorical_optimizer.param_groups[0]["lr"],
            epoch,
        )

        # Log epoch metrics
        writer.add_scalar("Epoch_Time", epoch_time, epoch)

        # Force a write to disk
        writer.flush()

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["train_numeric_loss"].append(avg_train_numeric_loss)
        history["train_categorical_loss"].append(avg_train_categorical_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_numeric_loss"].append(avg_val_numeric_loss)
        history["val_categorical_loss"].append(avg_val_categorical_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            plateau_counter = 0
            # Save best model with both optimizers
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "numeric_optimizer_state_dict": numeric_optimizer.state_dict(),
                    "categorical_optimizer_state_dict": categorical_optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
            print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
        else:
            plateau_counter += 1

        # Reduce learning rate on plateau
        if plateau_counter >= patience:
            plateau_counter = 0
            for param_group in numeric_optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5

            for param_group in categorical_optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5

            new_lr_numeric = numeric_optimizer.param_groups[0]["lr"]
            new_lr_categorical = categorical_optimizer.param_groups[0]["lr"]
            print(
                f"Reducing learning rates to {new_lr_numeric:.2e}/{new_lr_categorical:.2e} due to plateau"
            )

            # Early stop if learning rate gets too small
            if new_lr_numeric < 1e-7:
                print("Learning rate too small, stopping training")
                break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "numeric_optimizer_state_dict": numeric_optimizer.state_dict(),
                    "categorical_optimizer_state_dict": categorical_optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(
            f"Train Numeric Loss: {avg_train_numeric_loss:.4f}, Val Numeric Loss: {avg_val_numeric_loss:.4f}"
        )
        print(
            f"Train Cat Loss: {avg_train_categorical_loss:.4f}, Val Cat Loss: {avg_val_categorical_loss:.4f}"
        )
        print(
            f"Current numeric learning rate: {numeric_optimizer.param_groups[0]['lr']:.2e}"
        )
        print(
            f"Current categorical learning rate: {categorical_optimizer.param_groups[0]['lr']:.2e}"
        )
        print("-" * 50)

    # Ensure final metrics are written
    writer.flush()
    writer.close()

    # Print final message with log directory
    print(f"Training completed! TensorBoard logs saved to {log_dir}")
    print("To view training metrics, run:")
    print(f"tensorboard --logdir={os.path.dirname(log_dir)}")

    return history
