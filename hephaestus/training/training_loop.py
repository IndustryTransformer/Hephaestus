import os
import time

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from hephaestus.models import tabular_collate_fn
from hephaestus.training.training import (
    create_metric_history,
    create_optimizer,
    eval_step,
    train_step,
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
        torch.device(
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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=tabular_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=tabular_collate_fn,
    )

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, learning_rate)

    # Add patience for early stopping
    patience = 5
    plateau_counter = 0
    best_val_loss = float("inf")

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
        train_losses = {"loss": 0.0, "numeric_loss": 0.0, "categorical_loss": 0.0}

        train_iterator = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(train_iterator):
            # Calculate global step for logging
            global_step = epoch * len(train_loader) + batch_idx
            batch.to(device)

            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            # Train step
            batch_losses = train_step(model, batch)

            # Log batch-level metrics
            writer.add_scalars(
                "Batch/Loss",
                {
                    "train": batch_losses["loss"],
                    "numeric": batch_losses["numeric_loss"],
                    "categorical": batch_losses["categorical_loss"],
                },
                global_step,
            )

            # Update running loss
            for key in train_losses:
                train_losses[key] += batch_losses[key]

            # Only update optimizer after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(train_loader):
                # Apply regular gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # Update progress bar
            train_iterator.set_postfix(
                {
                    "loss": batch_losses["loss"],
                    "num_loss": batch_losses["numeric_loss"],
                    "cat_loss": batch_losses["categorical_loss"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Force a write to disk every 100 batches
            if batch_idx % 100 == 0:
                writer.flush()

        # Calculate average training losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)

        # Update scheduler after each epoch
        scheduler.step()

        # Validation phase
        model.eval()
        val_losses = {"loss": 0.0, "numeric_loss": 0.0, "categorical_loss": 0.0}

        val_iterator = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                batch.to(device)
                batch_losses = eval_step(model, batch)

                # Update running loss
                for key in val_losses:
                    val_losses[key] += batch_losses[key]

                # Update progress bar
                val_iterator.set_postfix({"val_loss": batch_losses["loss"]})

        # Calculate average validation losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)

        # Calculate epoch time before logging
        epoch_time = time.time() - start_time

        # TensorBoard logging
        writer.add_scalars(
            "Loss", {"train": train_losses["loss"], "val": val_losses["loss"]}, epoch
        )

        writer.add_scalars(
            "Numeric_Loss",
            {"train": train_losses["numeric_loss"], "val": val_losses["numeric_loss"]},
            epoch,
        )

        writer.add_scalars(
            "Categorical_Loss",
            {
                "train": train_losses["categorical_loss"],
                "val": val_losses["categorical_loss"],
            },
            epoch,
        )

        # Log learning rate
        writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

        # Log epoch metrics
        writer.add_scalar("Epoch_Time", epoch_time, epoch)

        # Force a write to disk
        writer.flush()

        # Update history
        history["train_loss"].append(train_losses["loss"])
        history["train_numeric_loss"].append(train_losses["numeric_loss"])
        history["train_categorical_loss"].append(train_losses["categorical_loss"])
        history["val_loss"].append(val_losses["loss"])
        history["val_numeric_loss"].append(val_losses["numeric_loss"])
        history["val_categorical_loss"].append(val_losses["categorical_loss"])

        # Early stopping logic
        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
            plateau_counter = 0
            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": best_val_loss,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
        else:
            plateau_counter += 1

        # Reduce learning rate on plateau
        if plateau_counter >= patience:
            plateau_counter = 0
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5
                new_lr = param_group["lr"]
                print(f"Reducing learning rate to {new_lr} due to plateau")

            # Early stop if learning rate gets too small
            if new_lr < 1e-7:
                print("Learning rate too small, stopping training")
                break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_losses["loss"],
                    "val_loss": val_losses["loss"],
                },
                os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s")
        print(
            f"Train Loss: {train_losses['loss']:.4f}, Val Loss: {val_losses['loss']:.4f}"
        )
        print(
            f"Train Numeric Loss: {train_losses['numeric_loss']:.4f}, Val Numeric Loss: {val_losses['numeric_loss']:.4f}"
        )
        print(
            f"Train Cat Loss: {train_losses['categorical_loss']:.4f}, Val Cat Loss: {val_losses['categorical_loss']:.4f}"
        )
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 50)

    # Ensure final metrics are written
    writer.flush()
    writer.close()

    # Print final message with log directory
    print(f"Training completed! TensorBoard logs saved to {log_dir}")
    print("To view training metrics, run:")
    print(f"tensorboard --logdir={os.path.dirname(log_dir)}")

    return history
