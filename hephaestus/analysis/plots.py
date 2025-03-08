import os

import matplotlib.pyplot as plt
import numpy as np

from hephaestus.analysis.analysis import DFComparison


def is_finite(data):
    """Check if data contains only finite values.

    Args:
        data: Array-like data to check

    Returns:
        bool: True if all values are finite, False otherwise
    """
    if isinstance(data, np.ndarray) or isinstance(data, list):
        return np.all(np.isfinite(data))
    elif hasattr(data, "values"):
        return np.all(np.isfinite(data.values))
    else:
        return np.all(np.isfinite(np.array(data)))


def safe_plot(x, y, *args, **kwargs):
    """Safely plot data by filtering out non-finite values.

    Args:
        x: X coordinates
        y: Y coordinates
        *args: Additional arguments for plt.plot
        **kwargs: Additional keyword arguments for plt.plot

    Returns:
        The result of plt.plot or None if data is invalid
    """
    if not is_finite(x) or not is_finite(y):
        # Create masks for finite values
        x_np = np.array(x)
        y_np = np.array(y)
        mask = np.isfinite(x_np) & np.isfinite(y_np)

        if not np.any(mask):
            print("Warning: No finite values to plot")
            return None

        # Filter data
        x_filtered = x_np[mask]
        y_filtered = y_np[mask]

        if len(x_filtered) > 0:
            return plt.plot(x_filtered, y_filtered, *args, **kwargs)
    else:
        return plt.plot(x, y, *args, **kwargs)


def plot_training_history(history, save_dir=None):
    """Plot training and validation loss history.

    Args:
        history: Dictionary containing loss history
        save_dir: Optional directory to save the plot
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)

    # Use safe_plot instead of plt.plot
    safe_plot(
        range(len(history["train_loss"])), history["train_loss"], label="Train Loss"
    )
    safe_plot(range(len(history["val_loss"])), history["val_loss"], label="Val Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    safe_plot(
        range(len(history["train_numeric_loss"])),
        history["train_numeric_loss"],
        label="Train Numeric",
    )
    safe_plot(
        range(len(history["train_categorical_loss"])),
        history["train_categorical_loss"],
        label="Train Categorical",
    )
    safe_plot(
        range(len(history["val_numeric_loss"])),
        history["val_numeric_loss"],
        label="Val Numeric",
    )
    safe_plot(
        range(len(history["val_categorical_loss"])),
        history["val_categorical_loss"],
        label="Val Categorical",
    )
    plt.title("Component Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "training_history.png"))

    plt.show()


def plot_col_error(dfs: DFComparison, col: str):
    plt.figure(figsize=(12, 6))

    # Plotting wind direction comparison
    plt.scatter(dfs.input_df[col], dfs.output_df[col], alpha=0.7)
    plt.xlabel(f"Actual {col.title()}")
    plt.ylabel("Predicted {col.title()}")
    plt.title("Actual vs Predicted")
    plt.grid(True, alpha=0.3)

    # Add a diagonal line representing perfect prediction
    min_val = min(dfs.input_df[col].min(), dfs.output_df[col].min())
    max_val = max(dfs.input_df[col].max(), dfs.output_df[col].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_col_comparison(dfs: DFComparison, col: str):
    plt.figure(figsize=(12, 6))

    # Plotting wind direction comparison
    plt.plot(dfs.input_df[col], label=f"Actual {col.title()}")
    plt.plot(dfs.output_df[col], label=f"Predicted {col.title()}")
    plt.xlabel("Time")
    plt.ylabel(col.title())
    plt.title(f"Actual vs Predicted {col.title()}")
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.tight_layout()
    plt.show()
