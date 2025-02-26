import os
import time

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    num_workers=0,  # Changed default to 0 to avoid multiprocessing issues
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

    Returns:
        dict: Training history
    """
    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, learning_rate)

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize metric history
    history = create_metric_history()
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = time.time()

        # Training phase
        model.train()
        train_losses = {"loss": 0.0, "numeric_loss": 0.0, "categorical_loss": 0.0}

        train_iterator = tqdm(train_loader, desc="Training")
        for batch_idx, batch_data in enumerate(train_iterator):
            # Check batch structure and convert to expected format
            # TimeSeriesDS likely returns a tuple of (numeric, categorical)
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                numeric_data, categorical_data = batch_data
                # Move to device
                numeric_data = numeric_data.to(device)
                categorical_data = categorical_data.to(device)
                batch = {"numeric": numeric_data, "categorical": categorical_data}
            else:
                # Handle if it's already a dict
                batch = batch_data
                for key in batch:
                    batch[key] = batch[key].to(device)
                    # Ensure data type consistency
                    batch[key] = batch[key].to(torch.float32)

            # Train step
            batch_losses = train_step(model, batch, optimizer)

            # Update running loss
            for key in train_losses:
                train_losses[key] += batch_losses[key]

            # Update progress bar
            train_iterator.set_postfix(
                {
                    "loss": batch_losses["loss"],
                    "num_loss": batch_losses["numeric_loss"],
                    "cat_loss": batch_losses["categorical_loss"],
                }
            )

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
            for batch_idx, batch_data in enumerate(val_iterator):
                # Check batch structure and convert to expected format
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    numeric_data, categorical_data = batch_data
                    # Move to device
                    numeric_data = numeric_data.to(device)
                    categorical_data = categorical_data.to(device)
                    batch = {"numeric": numeric_data, "categorical": categorical_data}
                else:
                    # Handle if it's already a dict
                    batch = batch_data
                    for key in batch:
                        batch[key] = batch[key].to(device)
                        # Ensure data type consistency
                        batch[key] = batch[key].to(torch.float32)

                # Eval step
                batch_losses = eval_step(model, batch)

                # Update running loss
                for key in val_losses:
                    val_losses[key] += batch_losses[key]

                # Update progress bar
                val_iterator.set_postfix({"val_loss": batch_losses["loss"]})

        # Calculate average validation losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", train_losses["loss"], epoch)
        writer.add_scalar("Loss/val", val_losses["loss"], epoch)
        writer.add_scalar("NumericLoss/train", train_losses["numeric_loss"], epoch)
        writer.add_scalar("NumericLoss/val", val_losses["numeric_loss"], epoch)
        writer.add_scalar(
            "CategoricalLoss/train", train_losses["categorical_loss"], epoch
        )
        writer.add_scalar("CategoricalLoss/val", val_losses["categorical_loss"], epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        # Update history
        history["train_loss"].append(train_losses["loss"])
        history["train_numeric_loss"].append(train_losses["numeric_loss"])
        history["train_categorical_loss"].append(train_losses["categorical_loss"])
        history["val_loss"].append(val_losses["loss"])
        history["val_numeric_loss"].append(val_losses["numeric_loss"])
        history["val_categorical_loss"].append(val_losses["categorical_loss"])

        # Save best model
        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
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
                os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"),
            )

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        print(
            f"Train Loss: {train_losses['loss']:.4f}, Val Loss: {val_losses['loss']:.4f}"
        )
        print(
            f"Train Numeric Loss: {train_losses['numeric_loss']:.4f}, Val Numeric Loss: {val_losses['numeric_loss']:.4f}"
        )
        print(
            f"Train Cat Loss: {train_losses['categorical_loss']:.4f}, Val Cat Loss: {val_losses['categorical_loss']:.4f}"
        )
        print("-" * 50)

    writer.close()
    print("Training completed!")

    return history
