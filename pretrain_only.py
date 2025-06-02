# %% [markdown]
# # Pre-training Only: Masked Tabular Modeling
# This notebook focuses solely on pre-training the model with masked reconstruction

# %%
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from icecream import ic
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

import hephaestus as hp
from hephaestus.timeseries_models.encoder_decoder import MaskedTabularPretrainer
from hephaestus.timeseries_models.encoder_decoder_dataset import (
    EncoderDecoderDataset,
    encoder_decoder_collate_fn,
)

torch.set_float32_matmul_precision("medium")


ic.disable()
# %%
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# %%
# Configuration
BATCH_SIZE = 32  # Larger batch size for more stable training
LEARNING_RATE = 5e-5  # Conservative learning rate
MAX_EPOCHS = 50
MASK_PROBABILITY = 0.15
D_MODEL = 64  # Slightly larger model
N_HEADS = 8
GRADIENT_CLIP = 1.0
ACCUMULATE_GRAD_BATCHES = 2  # Effective batch size = 64

# %% [markdown]
# ## Load and Preprocess Data

# %%
# Load the dataset
df = pd.read_parquet("data/combined_3w_real_sample.parquet")
print(f"Original dataset shape: {df.shape}")

# Take a smaller sample for faster iteration
df = df.head(1_000_000)  # 1M rows for debugging
df.drop(columns=["original_filename", "file_class_label", "system_id"], inplace=True)

# Process class labels
event_values = df["class"].unique()
event_values = np.sort(event_values)
events_names = {i: str(v) for i, v in enumerate(event_values)}
df["class"] = df["class"].astype("str")

print(f"Unique classes: {event_values}")
print(f"Dataset shape after sampling: {df.shape}")

# %%
# Filter out NaN classes
df = df.loc[df["class"].notna()]
print(f"Dataset shape after filtering: {df.shape}")

# Add time features
df["timestamp"] = pd.to_datetime(df["timestamp"])
seconds = (
    df["timestamp"].dt.hour * 3600
    + df["timestamp"].dt.minute * 60
    + df["timestamp"].dt.second
)
seconds_in_day = 24 * 60 * 60
df["timestamp_sin"] = np.sin(2 * np.pi * seconds / seconds_in_day)
df["timestamp_cos"] = np.cos(2 * np.pi * seconds / seconds_in_day)

day_of_year = df["timestamp"].dt.dayofyear
days_in_year = 366
df["dayofyear_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
df["dayofyear_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)

df = df.drop(columns=["timestamp"])

# %%
# Data normalization with debugging
scaler = RobustScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns

print("\n=== Data Normalization ===")
print(f"Numeric columns: {len(numeric_cols)}")
print(f"NaN values before normalization: {df[numeric_cols].isna().sum().sum():,}")

# Fill NaN values
df[numeric_cols] = df[numeric_cols].fillna(0)

# Check data statistics before scaling
print("\nBefore scaling:")
print(f"  Min: {df[numeric_cols].min().min():.2f}")
print(f"  Max: {df[numeric_cols].max().max():.2f}")
print(f"  Mean: {df[numeric_cols].mean().mean():.2f}")
print(f"  Std: {df[numeric_cols].std().mean():.2f}")

# Apply scaling
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Clip extreme values
df[numeric_cols] = df[numeric_cols].clip(-10, 10)

print("\nAfter scaling & clipping:")
print(f"  Min: {df[numeric_cols].min().min():.2f}")
print(f"  Max: {df[numeric_cols].max().max():.2f}")
print(f"  Mean: {df[numeric_cols].mean().mean():.2f}")
print(f"  Std: {df[numeric_cols].std().mean():.2f}")
print(f"  NaN values: {df[numeric_cols].isna().sum().sum()}")
print(f"  Inf values: {np.isinf(df[numeric_cols].values).sum()}")

# Add grouping index
df["idx"] = df.index // 512

# %% [markdown]
# ## Create Datasets and Data Loaders

# %%
# Generate config
time_series_config = hp.TimeSeriesConfig.generate(df=df, target="class")
print("\n=== Model Configuration ===")
print(f"Total tokens: {time_series_config.n_tokens}")
print(f"Categorical columns: {len(time_series_config.categorical_col_tokens)}")
print(f"Numeric columns: {len(time_series_config.numeric_col_tokens)}")

# Split data
train_idx = int(df.idx.max() * 0.8)
val_idx = int(df.idx.max() * 0.9)

train_df = df.loc[df.idx < train_idx].copy()
val_df = df.loc[(df.idx >= train_idx) & (df.idx < val_idx)].copy()
test_df = df.loc[df.idx >= val_idx].copy()

event_targets = {v: k for k, v in events_names.items()}

# Create datasets
train_ds = EncoderDecoderDataset(
    train_df, time_series_config, target_col="class", target_values=event_targets
)
val_ds = EncoderDecoderDataset(
    val_df, time_series_config, target_col="class", target_values=event_targets
)
test_ds = EncoderDecoderDataset(
    test_df, time_series_config, target_col="class", target_values=event_targets
)

print("\nDataset sizes:")
print(f"  Train: {len(train_ds):,}")
print(f"  Val: {len(val_ds):,}")
print(f"  Test: {len(test_ds):,}")

# %%
# Create data loaders
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=encoder_decoder_collate_fn,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=encoder_decoder_collate_fn,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

# %% [markdown]
# ## Custom Callback for Monitoring


# %%
class PretrainingMonitor(L.Callback):
    """Custom callback to monitor pre-training progress"""

    def __init__(self, check_every_n_epochs=1):
        super().__init__()
        self.check_every_n_epochs = check_every_n_epochs
        self.train_losses = []
        self.val_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:  # Log every 100 batches
            loss = (
                outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
            )
            self.train_losses.append(loss)

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None:
            self.val_losses.append(val_loss.item())

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.check_every_n_epochs == 0:
            print(f"\n=== Epoch {trainer.current_epoch} Summary ===")
            print(f"Train loss: {trainer.callback_metrics.get('train_loss', 'N/A')}")
            print(f"Val loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")

            # Check for NaN in model parameters
            for name, param in pl_module.named_parameters():
                if torch.isnan(param).any():
                    print(f"WARNING: NaN found in {name}")
                if torch.isinf(param).any():
                    print(f"WARNING: Inf found in {name}")


# %% [markdown]
# ## Create and Train Pre-training Model

# %%
# Create the pre-training model
pretrain_model = MaskedTabularPretrainer(
    time_series_config,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    learning_rate=LEARNING_RATE,
    mask_probability=MASK_PROBABILITY,
)

print("\n=== Model Architecture ===")
print(f"Model parameters: {sum(p.numel() for p in pretrain_model.parameters()):,}")
print(
    f"Trainable parameters: {sum(p.numel() for p in pretrain_model.parameters() if p.requires_grad):,}"
)

# %%
# Setup callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
    ),
    ModelCheckpoint(
        dirpath="checkpoints/pretrain_only",
        filename="pretrain_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
    ),
    LearningRateMonitor(logging_interval="step"),
    PretrainingMonitor(check_every_n_epochs=1),
    RichProgressBar(),
    RichModelSummary(max_depth=2),
]

# Setup logger
logger = TensorBoardLogger(
    "runs/pretrain_only",
    name=f"{dt.now().strftime('%Y%m%d_%H%M%S')}_pretrain",
)

# %%
# Create trainer
trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    logger=logger,
    callbacks=callbacks,
    gradient_clip_val=GRADIENT_CLIP,
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    precision="16-mixed" if torch.cuda.is_available() else 32,
    deterministic=True,
    log_every_n_steps=10,
    val_check_interval=0.25,  # Check validation 4 times per epoch
)

# %%
# Inspect a batch before training
print("\n=== Inspecting Training Batch ===")
batch = next(iter(train_dl))
inputs, targets = batch

print("Batch shapes:")
if inputs.numeric is not None:
    print(f"  Numeric inputs: {inputs.numeric.shape}")
    print(f"  Numeric range: [{inputs.numeric.min():.3f}, {inputs.numeric.max():.3f}]")
if inputs.categorical is not None:
    print(f"  Categorical inputs: {inputs.categorical.shape}")
    print(f"  Unique categorical values: {inputs.categorical.unique().shape[0]}")

# Test forward pass
with torch.no_grad():
    # Apply masking
    masked_numeric, masked_categorical, numeric_mask, categorical_mask = (
        pretrain_model.mask_inputs(inputs.numeric, inputs.categorical)
    )

    # Forward pass
    numeric_pred, categorical_pred = pretrain_model(
        input_numeric=masked_numeric,
        input_categorical=masked_categorical,
        numeric_mask=numeric_mask,
        categorical_mask=categorical_mask,
        deterministic=True,
    )

    if numeric_pred is not None:
        print(f"\nNumeric predictions shape: {numeric_pred.shape}")
        print(
            f"Numeric predictions range: [{numeric_pred.min():.3f}, {numeric_pred.max():.3f}]"
        )

# %% [markdown]
# ## Train the Model

# %%
print("\n=== Starting Pre-training ===")
trainer.fit(pretrain_model, train_dl, val_dl)

# %%
# Plot training curves if available
monitor_callback = None
for callback in trainer.callbacks:
    if isinstance(callback, PretrainingMonitor):
        monitor_callback = callback
        break

if monitor_callback and len(monitor_callback.train_losses) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(monitor_callback.train_losses, label="Train Loss", alpha=0.7)
    if len(monitor_callback.val_losses) > 0:
        val_x = np.linspace(
            0, len(monitor_callback.train_losses), len(monitor_callback.val_losses)
        )
        plt.plot(val_x, monitor_callback.val_losses, label="Val Loss", marker="o")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Pre-training Loss Curves")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pretrain_loss_curves.png", dpi=150)
    plt.show()

# %% [markdown]
# ## Evaluate Reconstruction Quality

# %%
# Load best model
best_model_path = trainer.checkpoint_callback.best_model_path
print(f"\nLoading best model from: {best_model_path}")

best_model = MaskedTabularPretrainer.load_from_checkpoint(
    best_model_path,
    config=time_series_config,
)
best_model.eval()

# %%
# Evaluate reconstruction quality on test set
print("\n=== Evaluating Reconstruction Quality ===")

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=encoder_decoder_collate_fn,
    num_workers=2,
)

total_numeric_error = 0
total_numeric_samples = 0
reconstruction_examples = []

with torch.no_grad():
    for batch_idx, (inputs, _) in enumerate(test_dl):
        if batch_idx >= 10:  # Evaluate on 10 batches
            break

        # Apply masking
        masked_numeric, masked_categorical, numeric_mask, categorical_mask = (
            best_model.mask_inputs(inputs.numeric, inputs.categorical)
        )

        # Forward pass
        numeric_pred, categorical_pred = best_model(
            input_numeric=masked_numeric,
            input_categorical=masked_categorical,
            numeric_mask=numeric_mask,
            categorical_mask=categorical_mask,
            deterministic=True,
        )

        if numeric_pred is not None and numeric_mask is not None:
            # Calculate reconstruction error on masked positions
            numeric_mask_transposed = numeric_mask.permute(0, 2, 1)
            numeric_transposed = inputs.numeric.permute(0, 2, 1)

            masked_true = numeric_transposed[numeric_mask_transposed]
            masked_pred = numeric_pred[numeric_mask_transposed]

            if masked_true.numel() > 0:
                error = torch.abs(masked_pred - masked_true).mean()
                total_numeric_error += error.item() * masked_true.numel()
                total_numeric_samples += masked_true.numel()

                # Store examples for visualization
                if batch_idx == 0:
                    reconstruction_examples.append(
                        {
                            "true": masked_true[:100].cpu().numpy(),
                            "pred": masked_pred[:100].cpu().numpy(),
                        }
                    )

if total_numeric_samples > 0:
    avg_error = total_numeric_error / total_numeric_samples
    print(f"Average numeric reconstruction error: {avg_error:.4f}")

# %%
# Visualize reconstruction examples
if reconstruction_examples:
    example = reconstruction_examples[0]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(example["true"], example["pred"], alpha=0.5, s=10)
    plt.plot([-10, 10], [-10, 10], "r--", label="Perfect reconstruction")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Reconstruction Scatter Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    errors = np.abs(example["true"] - example["pred"])
    plt.hist(errors, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    plt.title(
        f"Reconstruction Error Distribution\nMean: {np.mean(errors):.3f}, Std: {np.std(errors):.3f}"
    )
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reconstruction_quality.png", dpi=150)
    plt.show()

# %%
print("\n=== Pre-training Complete ===")
print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
print(f"Final validation loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")

# Save model info
model_info = {
    "config": time_series_config,
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "mask_probability": MASK_PROBABILITY,
    "best_checkpoint": trainer.checkpoint_callback.best_model_path,
    "final_val_loss": float(trainer.callback_metrics.get("val_loss", -1)),
}

torch.save(model_info, "checkpoints/pretrain_only/model_info.pt")
print("\nModel info saved to: checkpoints/pretrain_only/model_info.pt")
