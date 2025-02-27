import os
from datetime import datetime as dt

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from hephaestus.models.decoder_patch import patch_time_series_decoder
from hephaestus.models.models import TimeSeriesDecoder
from hephaestus.training.training_loop import train_model


def run_training(train_ds, test_ds, time_series_config):
    """
    Run the training process with improved debugging and tensor board logging

    Args:
        train_ds: Training dataset
        test_ds: Test/validation dataset
        time_series_config: Configuration for the time series model

    Returns:
        dict: Training history
    """
    # Apply model patches first
    patch_time_series_decoder()

    # Set up the training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model with more stable initialization
    tabular_decoder = TimeSeriesDecoder(
        time_series_config,
        d_model=128,  # Further reduced for stability
        n_heads=4,  # Reduced number of attention heads
    )

    # Print model configuration info
    print(
        f"Model has {sum(p.numel() for p in tabular_decoder.parameters())} parameters"
    )
    print(f"Categorical dimensions: {tabular_decoder.categorical_dims}")

    # Verify categorical dimensions match dataset
    sample_batch = next(iter(DataLoader(train_ds, batch_size=1)))
    if "categorical" in sample_batch:
        for i in range(sample_batch["categorical"].size(2)):  # Iterate over features
            unique_classes = torch.unique(sample_batch["categorical"][:, :, i])
            max_class = unique_classes.max().item()
            print(
                f"Feature {i} has max class index {max_class}, model expects {tabular_decoder.categorical_dims[i]} classes"
            )
            if max_class >= tabular_decoder.categorical_dims[i]:
                print(
                    f"Warning: Feature {i} has class indices exceeding model's categorical dims!"
                )

    # Initialize weights with more conservative values
    def init_weights(m):
        if hasattr(m, "weight") and m.weight is not None:
            if len(m.weight.shape) > 1:
                # Use Kaiming initialization for better stability
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                # Scale down initial weights to prevent explosions
                m.weight.data *= 0.05
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # Apply custom weight initialization
    tabular_decoder.apply(init_weights)
    print("Applied conservative weight initialization")

    # Move model to device
    tabular_decoder.to(device)

    # Set up training parameters with much more conservative values
    learning_rate = 1e-5  # Reduced learning rate by 5x
    num_epochs = 10
    gradient_accumulation_steps = 4  # Increased for stability
    max_grad_norm = 0.1  # Much tighter gradient clipping

    # Add gradient explosion detection threshold
    max_gradient_norm_allowed = 10.0
    max_explosion_count = 5  # Allow this many explosions before reducing LR permanently

    timestamp = dt.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = f"runs/{timestamp}_planets_fixed_categorical"
    save_dir = "models/planets"

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("To view logs, run: tensorboard --logdir=runs")

    # Train the model with enhanced stability parameters
    history = train_model(
        model=tabular_decoder,
        train_dataset=train_ds,
        val_dataset=test_ds,
        batch_size=4,  # Reduced batch size for stability
        epochs=num_epochs,
        learning_rate=learning_rate,
        log_dir=log_dir,
        save_dir=save_dir,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        explosion_threshold=max_gradient_norm_allowed,
        max_explosions_per_epoch=max_explosion_count,
    )

    # Visualize training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_numeric_loss"], label="Train Numeric")
    plt.plot(history["train_categorical_loss"], label="Train Categorical")
    plt.plot(history["val_numeric_loss"], label="Val Numeric")
    plt.plot(history["val_categorical_loss"], label="Val Categorical")
    plt.title("Component Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{log_dir}/training_history.png")
    plt.show()

    # Load and evaluate the best model
    best_model_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        tabular_decoder.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}"
        )

    return history
