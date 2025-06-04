import csv
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as L


class MetricsLogger(L.Callback):
    """Custom callback to log metrics to CSV files and maintain DataFrames.

    This callback captures training and validation metrics at each epoch
    and saves them to CSV files, while also maintaining in-memory DataFrames
    for easy analysis.
    """

    def __init__(self, save_dir: str = "output_metrics"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage for metrics
        self.train_metrics = []
        self.val_metrics = []

        # CSV file paths
        self.train_csv_path = self.save_dir / "train_metrics.csv"
        self.val_csv_path = self.save_dir / "validation_metrics.csv"

        # DataFrames for easy access
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()

        # Initialize CSV files with headers
        self._initialize_csv_files()

    def _initialize_csv_files(self):
        """Initialize CSV files with appropriate headers."""
        train_headers = [
            "epoch",
            "step",
            "train_loss",
            "learning_rate",
            "timestamp",
        ]
        val_headers = [
            "epoch",
            "val_loss",
            "timestamp",
        ]

        # Create train CSV
        with open(self.train_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(train_headers)

        # Create validation CSV
        with open(self.val_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(val_headers)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called when the train epoch ends."""
        # Get current epoch metrics
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step

        # Extract training metrics
        train_loss = self._get_metric_value(metrics, "train_loss")

        # Get learning rate
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else 0.0

        # Create row data
        timestamp = pd.Timestamp.now()
        train_row = {
            "epoch": current_epoch,
            "step": global_step,
            "train_loss": train_loss,
            "learning_rate": lr,
            "timestamp": timestamp,
        }

        # Add to storage
        self.train_metrics.append(train_row)

        # Append to CSV
        with open(self.train_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    current_epoch,
                    global_step,
                    train_loss,
                    lr,
                    timestamp,
                ]
            )

        # Update DataFrame
        self.train_df = pd.DataFrame(self.train_metrics)

        print(
            f"✓ Saved training metrics for epoch {current_epoch} "
            f"to {self.train_csv_path}"
        )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called when the validation epoch ends."""
        # Get current epoch metrics
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch

        # Extract validation metrics
        val_loss = self._get_metric_value(metrics, "val_loss")

        # Create row data
        timestamp = pd.Timestamp.now()
        val_row = {
            "epoch": current_epoch,
            "val_loss": val_loss,
            "timestamp": timestamp,
        }

        # Add to storage
        self.val_metrics.append(val_row)

        # Append to CSV
        with open(self.val_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    current_epoch,
                    val_loss,
                    timestamp,
                ]
            )

        # Update DataFrame
        self.val_df = pd.DataFrame(self.val_metrics)

        print(
            f"✓ Saved validation metrics for epoch {current_epoch} "
            f"to {self.val_csv_path}"
        )

    def _get_metric_value(self, metrics: dict[str, Any], key: str) -> float:
        """Safely extract metric value from callback metrics."""
        if key in metrics:
            value = metrics[key]
            if hasattr(value, "item"):
                return value.item()
            return float(value)
        return 0.0

    def get_train_dataframe(self) -> pd.DataFrame:
        """Get the current training metrics as a DataFrame."""
        return self.train_df.copy()

    def get_validation_dataframe(self) -> pd.DataFrame:
        """Get the current validation metrics as a DataFrame."""
        return self.val_df.copy()

    def save_combined_metrics(self, filename: str = "combined_metrics.csv"):
        """Save a combined view of training and validation metrics."""
        if len(self.train_df) == 0 or len(self.val_df) == 0:
            print("No metrics to save yet.")
            return

        # Merge training and validation metrics on epoch
        combined = pd.merge(
            self.train_df[
                [
                    "epoch",
                    "train_numeric_loss",
                    "train_categorical_loss",
                    "train_categorical_accuracy",
                    "learning_rate",
                ]
            ],
            self.val_df[
                [
                    "epoch",
                    "val_numeric_loss",
                    "val_categorical_loss",
                    "val_categorical_accuracy",
                ]
            ],
            on="epoch",
            how="outer",
        ).sort_values("epoch")

        combined_path = self.save_dir / filename

        combined.to_csv(combined_path, index=False)
        print(f"✓ Saved combined metrics to {combined_path}")

        return combined
