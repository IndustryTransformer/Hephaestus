import os
import time

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from hephaestus.models.models import TimeSeriesInputs
from hephaestus.training.training import (
    create_metric_history,
    create_optimizer,
    eval_step,
    train_step,
)


def custom_collate_fn(batch):
    """Custom collate function for TimeSeriesInputs objects."""
    numeric_tensors = torch.stack([item.numeric for item in batch])

    if batch[0].categorical is not None:
        categorical_tensors = torch.stack([item.categorical for item in batch])
        return TimeSeriesInputs(
            numeric=numeric_tensors, categorical=categorical_tensors
        )
    else:
        return TimeSeriesInputs(numeric=numeric_tensors, categorical=None)


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
    num_workers=0,  # Changed default to 0 to avoid multiprocessing issues
    gradient_accumulation_steps=1,  # Added to reduce memory pressure
    max_grad_norm=1.0,  # Added for controlling gradient norms
    explosion_threshold=10.0,  # New param: threshold for gradient explosion
    max_explosions_per_epoch=5,  # New param: max explosions before lr reduction
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
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn,
    )

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, learning_rate)

    # Add patience for early stopping and learning rate reduction
    patience = 5
    plateau_counter = 0
    best_val_loss = float("inf")

    # Add tracking for gradient issues
    gradient_issues_counter = 0

    # Create TensorBoard writer with flush_secs=10 to ensure more frequent writes
    writer = SummaryWriter(log_dir, flush_secs=10)
    print(f"TensorBoard log directory: {log_dir}")

    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize metric history
    history = create_metric_history()
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        start_time = time.time()

        # Training phase
        model.train()
        train_losses = {"loss": 0.0, "numeric_loss": 0.0, "categorical_loss": 0.0}
        skipped_batches = 0
        gradient_issues = 0
        explosion_count = 0  # Track gradient explosions per epoch

        train_iterator = tqdm(train_loader, desc="Training")
        for batch_idx, batch_data in enumerate(train_iterator):
            # Calculate global step for logging
            global_step = epoch * len(train_loader) + batch_idx
            batch = batch_data
            batch.to(device)
            # Check batch structure and convert to expected format
            # TimeSeriesDS likely returns a tuple of (numeric, categorical)
            # if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
            #     numeric_data, categorical_data = batch_data
            #     # Move to device
            #     numeric_data = numeric_data.to(device)
            #     categorical_data = categorical_data.to(device)
            #     batch = {"numeric": numeric_data, "categorical": categorical_data}
            # else:
            #     # Handle if it's already a dict
            #     batch = batch_data
            #     for key in batch:
            #         batch[key] = batch[key].to(device)
            #         # Ensure data type consistency
            #         batch[key] = batch[key].to(torch.float32)

            # # Convert batch to float32
            # for key in batch:
            #     batch[key] = batch[key].to(device).to(torch.float32)

            # Implement gradient accumulation steps
            # Only zero gradients at the start of accumulation
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            # Train step with scheduler
            batch_losses = train_step(model, batch, optimizer, scheduler)

            # Track gradient issues
            if "gradient_status" in batch_losses:
                if batch_losses["gradient_status"] == "exploded":
                    gradient_issues += 1
                    writer.add_scalar("Gradient_Issues/Exploded", 1, global_step)

                    # Log the gradient norm
                    if "gradient_norm" in batch_losses:
                        writer.add_scalar(
                            "Gradient_Issues/Norm",
                            batch_losses["gradient_norm"],
                            global_step,
                        )

                # Skip problematic batches in metrics if needed
                if batch_losses["gradient_status"] in ["bad_loss", "error"]:
                    skipped_batches += 1
                    continue

            # Check for NaN in loss
            if torch.isnan(torch.tensor(batch_losses["loss"])).item():
                print(f"NaN loss detected in batch {batch_idx}, skipping")
                skipped_batches += 1
                optimizer.zero_grad()  # Reset gradients
                continue

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
                # Check for gradient explosion before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=False
                )

                if grad_norm > explosion_threshold:
                    explosion_count += 1
                    gradient_issues += 1
                    print("Training Loop Train Model: Gradient explosion detected!")
                    print(f"Exploding gradients detected! Norm: {grad_norm}")

                    # Skip batch if gradient is extremely large
                    if grad_norm > explosion_threshold * 10:  # Extreme case
                        print(
                            f"Extreme gradient explosion: {grad_norm}, skipping update"
                        )
                        optimizer.zero_grad()  # Reset gradients
                        skipped_batches += 1
                        continue

                    # Temporary reduce learning rate for this update
                    orig_lr = optimizer.param_groups[0]["lr"]
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = orig_lr * 0.1
                    print("Reducing learning rate temporarily due to gradient issues")

                # Apply regular gradient clipping with max_grad_norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Attempt the optimizer step within a try-except block
                try:
                    optimizer.step()
                    # If we temporarily reduced the LR, restore it
                    if grad_norm > explosion_threshold:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = orig_lr
                except RuntimeError as e:
                    print(f"Optimizer step failed: {e}")
                    skipped_batches += 1
                    optimizer.zero_grad()  # Reset gradients

                # Record step in tracking
                if batch_idx % 10 == 0:  # Log every 10 batches
                    writer.add_scalar("Grad_Norm", grad_norm, global_step)

            # Update progress bar with more info
            train_iterator.set_postfix(
                {
                    "loss": batch_losses["loss"],
                    "num_loss": batch_losses["numeric_loss"],
                    "cat_loss": batch_losses["categorical_loss"],
                    "grad_issues": gradient_issues,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Force a write to disk every 100 batches
            if batch_idx % 100 == 0:
                writer.flush()

        # Calculate average training losses accounting for skipped batches
        effective_batches = max(1, len(train_loader) - skipped_batches)
        for key in train_losses:
            train_losses[key] /= effective_batches

        # Log gradient issues for the epoch
        writer.add_scalar("Epoch/Gradient_Issues", gradient_issues, epoch)
        gradient_issues_counter += gradient_issues

        # Add learning rate monitoring and reduction
        current_lr = scheduler.get_last_lr()[0]

        # If we have too many gradient issues, reduce learning rate more aggressively
        if (
            gradient_issues > len(train_loader) * 0.1
        ):  # More than 10% of batches had issues
            print(
                f"Too many gradient issues ({gradient_issues}), reducing learning rate"
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr * 0.5

        # Check if we need to permanently reduce LR due to many explosions
        if explosion_count >= max_explosions_per_epoch:
            print(
                f"Too many gradient explosions ({explosion_count}), permanently reducing learning rate"
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5
                print(f"New learning rate: {param_group['lr']}")

            # Early stop if training is too unstable
            if param_group["lr"] < 1e-7 or (
                epoch > 0 and explosion_count > max_explosions_per_epoch * 2
            ):
                print("Training too unstable, stopping early")
                break

        # Update scheduler after each epoch (normal schedule)
        scheduler.step()

        # Validation phase
        model.eval()
        val_losses = {"loss": 0.0, "numeric_loss": 0.0, "categorical_loss": 0.0}

        val_iterator = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_iterator):
                # Check batch structure and convert to expected format
                # if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                #     numeric_data, categorical_data = batch_data
                #     # Move to device
                #     numeric_data = numeric_data.to(device)
                #     categorical_data = categorical_data.to(device)
                #     batch = {"numeric": numeric_data, "categorical": categorical_data}
                # else:
                #     # Handle if it's already a dict
                #     batch = batch_data
                #     for key in batch:
                #         batch[key] = batch[key].to(device)
                #         # Ensure data type consistency
                #         batch[key] = batch[key].to(torch.float32)

                # Eval step
                batch = batch_data
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

        # Enhanced TensorBoard logging
        # Log training metrics
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

        # Early stopping and LR reduction logic
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
        print(
            f"Gradient issues this epoch: {gradient_issues}/{len(train_loader)} batches"
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
