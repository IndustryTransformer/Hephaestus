import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from hephaestus.training.training import create_optimizer, eval_step, train_step


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    log_dir: str = "runs/default",
    save_dir: str = "checkpoints",
    device: torch.device = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 0.1,
    explosion_threshold: float = 10.0,
    max_explosions_per_epoch: int = 5,
    writer: Optional[SummaryWriter] = None,
):
    """Train a model with gradient monitoring and TensorBoard logging.

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Initial learning rate
        log_dir: Directory for TensorBoard logs
        save_dir: Directory to save model checkpoints
        device: Device to use for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        explosion_threshold: Threshold for gradient explosion detection
        max_explosions_per_epoch: Maximum allowed gradient explosions per epoch
        writer: TensorBoard writer (optional, will create if None)

    Returns:
        Dict: Training history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, learning_rate)

    # Setup tensorboard writer if not provided
    if writer is None:
        writer = SummaryWriter(log_dir=log_dir)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Create output directories
    os.makedirs(save_dir, exist_ok=True)

    # Initialize metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_numeric_loss": [],
        "train_categorical_loss": [],
        "val_numeric_loss": [],
        "val_categorical_loss": [],
        "gradient_explosions": [],
        "avg_gradient_norm": [],
    }

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()

        # Initialize metrics for this epoch
        train_loss = 0.0
        train_numeric_loss = 0.0
        train_categorical_loss = 0.0
        gradient_norm_sum = 0.0
        gradient_norm_count = 0
        explosion_count = 0

        # Progress bar for training
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}/{epochs} [Training]")

        # Training loop
        for batch_idx, batch in progress_bar:
            # Move batch to device
            batch.numeric = batch.numeric.to(device)
            batch.categorical = batch.categorical.to(device)

            # Check if we're accumulating gradients
            is_accumulating = (batch_idx + 1) % gradient_accumulation_steps != 0

            # Perform training step
            step_result = train_step(
                model,
                batch,
                optimizer if not is_accumulating else None,
                scheduler,
                accumulating_gradients=is_accumulating,
            )

            # Log step metrics
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar(
                "train/step_loss",
                step_result["loss"],
                global_step=batch_idx + (epoch - 1) * len(train_loader),
            )
            writer.add_scalar(
                "train/step_numeric_loss",
                step_result["numeric_loss"],
                global_step=batch_idx + (epoch - 1) * len(train_loader),
            )
            writer.add_scalar(
                "train/step_categorical_loss",
                step_result["categorical_loss"],
                global_step=batch_idx + (epoch - 1) * len(train_loader),
            )
            writer.add_scalar(
                "train/learning_rate",
                current_lr,
                global_step=batch_idx + (epoch - 1) * len(train_loader),
            )

            # Track gradient metrics
            if "gradient_norm" in step_result:
                gradient_norm = step_result["gradient_norm"]
                writer.add_scalar(
                    "train/gradient_norm",
                    gradient_norm,
                    global_step=batch_idx + (epoch - 1) * len(train_loader),
                )

                # Check for gradient explosion
                is_exploded = step_result["gradient_status"] == "exploded"
                writer.add_scalar(
                    "train/gradient_exploded",
                    int(is_exploded),
                    global_step=batch_idx + (epoch - 1) * len(train_loader),
                )

                if is_exploded:
                    explosion_count += 1

                # Track avg gradient norm
                if gradient_norm > 0:
                    gradient_norm_sum += gradient_norm
                    gradient_norm_count += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{step_result['loss']:.2f}",
                    "num_loss": f"{step_result['numeric_loss']:.2f}",
                    "cat_loss": f"{step_result['categorical_loss']:.2f}",
                    "grad_issues": explosion_count,
                    "lr": f"{current_lr:.6f}",
                }
            )

            # Update epoch metrics
            train_loss += step_result["loss"]
            train_numeric_loss += step_result["numeric_loss"]
            train_categorical_loss += step_result["categorical_loss"]

            # Check if we need to stop due to too many explosions
            if explosion_count > max_explosions_per_epoch:
                writer.add_text(
                    "train/early_stop",
                    f"Stopped epoch {epoch} early due to {explosion_count} gradient explosions",
                    global_step=batch_idx + (epoch - 1) * len(train_loader),
                )
                break

        # Compute average metrics for the epoch
        train_loss /= len(train_loader)
        train_numeric_loss /= len(train_loader)
        train_categorical_loss /= len(train_loader)
        avg_gradient_norm = (
            gradient_norm_sum / gradient_norm_count if gradient_norm_count > 0 else 0
        )

        # Add epoch metrics to history
        history["train_loss"].append(train_loss)
        history["train_numeric_loss"].append(train_numeric_loss)
        history["train_categorical_loss"].append(train_categorical_loss)
        history["gradient_explosions"].append(explosion_count)
        history["avg_gradient_norm"].append(avg_gradient_norm)

        # Log epoch metrics to tensorboard
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/train_numeric_loss", train_numeric_loss, epoch)
        writer.add_scalar("epoch/train_categorical_loss", train_categorical_loss, epoch)
        writer.add_scalar("epoch/gradient_explosions", explosion_count, epoch)
        writer.add_scalar("epoch/avg_gradient_norm", avg_gradient_norm, epoch)

        # Evaluate model
        model.eval()
        val_loss = 0.0
        val_numeric_loss = 0.0
        val_categorical_loss = 0.0

        # Progress bar for validation
        val_progress = tqdm(val_loader)
        val_progress.set_description(f"Epoch {epoch}/{epochs} [Validation]")

        for val_batch in val_progress:
            # Move batch to device
            val_batch.numeric = val_batch.numeric.to(device)
            val_batch.categorical = val_batch.categorical.to(device)

            # Evaluate step
            eval_result = eval_step(model, val_batch)

            # Update validation metrics
            val_loss += eval_result["loss"]
            val_numeric_loss += eval_result["numeric_loss"]
            val_categorical_loss += eval_result["categorical_loss"]

        # Compute average validation metrics
        val_loss /= len(val_loader)
        val_numeric_loss /= len(val_loader)
        val_categorical_loss /= len(val_loader)

        # Add validation metrics to history
        history["val_loss"].append(val_loss)
        history["val_numeric_loss"].append(val_numeric_loss)
        history["val_categorical_loss"].append(val_categorical_loss)

        # Log validation metrics to tensorboard
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_numeric_loss", val_numeric_loss, epoch)
        writer.add_scalar("epoch/val_categorical_loss", val_categorical_loss, epoch)

        # Print epoch results
        print(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.4f} "
            f"(Num: {train_numeric_loss:.4f}, Cat: {train_categorical_loss:.4f}) - "
            f"Val Loss: {val_loss:.4f} "
            f"(Num: {val_numeric_loss:.4f}, Cat: {val_categorical_loss:.4f}) - "
            f"Gradient explosions: {explosion_count}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
            print(f"✓ Saved new best model with validation loss: {val_loss:.4f}")

        # Save latest model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            },
            os.path.join(save_dir, "latest_model.pt"),
        )

        # Update learning rate scheduler
        scheduler.step()

    writer.close()
    return history
