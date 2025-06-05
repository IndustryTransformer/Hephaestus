# %% [markdown]
# # Pre-training with Efficient Attention
# This notebook uses efficient attention mechanisms for longer context windows

# %%
import os
from datetime import datetime as dt

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
from hephaestus.timeseries_models.encoder_decoder_dataset import (
    EncoderDecoderDataset,
    encoder_decoder_collate_fn,
)
from hephaestus.timeseries_models.time_series_utils import MetricsLogger
from hephaestus.training.masked_timeseries_training import (
    MaskedTabularPretrainer,
)

torch.set_float32_matmul_precision("medium")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
# Configuration for efficient attention
BATCH_SIZE = 14  # Reduced batch size for longer sequences
SEQUENCE_LENGTH = 1090  # Much longer context window!
LEARNING_RATE = 5e-5
MAX_EPOCHS = 50
MASK_PROBABILITY = 0.000001
D_MODEL = 128
N_HEADS = 8
GRADIENT_CLIP = 1.0
# Disable gradient accumulation so that step logging aligns with optimizer updates
ACCUMULATE_GRAD_BATCHES = 1

# Efficient attention configuration
ATTENTION_TYPE = (
    "flash"  # Options: "flash", "local", "sparse", "featurewise", "chunked"
)
ATTENTION_KWARGS = {
    # For local attention
    # "window_size": 256,
    # For sparse attention
    # "stride": 8,
    # "local_window": 128,
    # For chunked attention
    # "chunk_size": 256,
    # "overlap": 32,
    # For featurewise attention
    # "share_weights": True,
}

# %% [markdown]
# ## Load and Preprocess Data

# %%
# Load the dataset
df = pd.read_parquet("data/combined_3w_real_sample.parquet")
print(f"Original dataset shape: {df.shape}")

# Take a smaller sample for faster iteration
df = df.head(2_000_000)  # 2M rows to have enough for longer sequences
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

# %% [markdown]
# ## Configure Features and Normalize Data

# %%
# Define features
categorical_cols = ["class"]
drop_cols = ["timestamp"] + categorical_cols
# Only select columns that are actually numeric
numeric_cols = [
    col for col in df.select_dtypes(include=["number"]).columns if col not in drop_cols
]

# %%
# Normalize numeric columns
print("=== Data Normalization ===")
print(f"Numeric columns: {len(numeric_cols)}")
print(f"NaN values before normalization: {df[numeric_cols].isna().sum().sum():,}")

# Use RobustScaler which is less sensitive to outliers
scaler = RobustScaler()
numeric_data = df[numeric_cols].values.astype(np.float32)

# Replace inf with nan before scaling
numeric_data[np.isinf(numeric_data)] = np.nan

# Print statistics before scaling
print("\nBefore scaling:")
print(f"  Min: {np.nanmin(numeric_data):.2f}")
print(f"  Max: {np.nanmax(numeric_data):.2f}")
print(f"  Mean: {np.nanmean(numeric_data):.2f}")
print(f"  Std: {np.nanstd(numeric_data):.2f}")

# Fit and transform
scaled_data = scaler.fit_transform(numeric_data)

# Clip to reasonable range to prevent extreme values
scaled_data = np.clip(scaled_data, -10, 10)

# Replace NaN with 0 after scaling
scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=10.0, neginf=-10.0)

df[numeric_cols] = scaled_data

print("\nAfter scaling & clipping:")
print(f"  Min: {df[numeric_cols].min().min():.2f}")
print(f"  Max: {df[numeric_cols].max().max():.2f}")
print(f"  Mean: {df[numeric_cols].mean().mean():.2f}")
print(f"  Std: {df[numeric_cols].std().mean():.2f}")
print(f"  NaN values: {df[numeric_cols].isna().sum().sum()}")
print(f"  Inf values: {np.isinf(df[numeric_cols].values).sum()}")

# %% [markdown]
# ## Group Data into Longer Sequences

# %%
# Group into sequences with longer context
df.sort_values("timestamp", inplace=True)
df["idx"] = df.index // SEQUENCE_LENGTH  # Create sequences of SEQUENCE_LENGTH timesteps

# %% [markdown]
# ## Create Data Configuration and Datasets

# %%
# Create configuration
time_series_config = hp.TimeSeriesConfig.generate(df=df, target="class")

print("=== Model Configuration ===")
print(f"Total tokens: {time_series_config.n_tokens}")
print(f"Categorical columns: {len(time_series_config.categorical_col_tokens)}")
print(f"Numeric columns: {len(time_series_config.numeric_col_tokens)}")

# %%
# Split data by groups
train_groups = df.loc[df["idx"] < df["idx"].quantile(0.8), "idx"].unique()
val_groups = df.loc[
    (df["idx"] >= df["idx"].quantile(0.8)) & (df["idx"] < df["idx"].quantile(0.9)),
    "idx",
].unique()
test_groups = df.loc[df["idx"] >= df["idx"].quantile(0.9), "idx"].unique()

# Create event targets mapping
event_targets = {v: k for k, v in events_names.items()}

# Create datasets
train_ds = EncoderDecoderDataset(
    df[df["idx"].isin(train_groups)],
    time_series_config,
    target_col="class",
    target_values=event_targets,
)

val_ds = EncoderDecoderDataset(
    df[df["idx"].isin(val_groups)],
    time_series_config,
    target_col="class",
    target_values=event_targets,
)

test_ds = EncoderDecoderDataset(
    df[df["idx"].isin(test_groups)],
    time_series_config,
    target_col="class",
    target_values=event_targets,
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
    num_workers=4,
    collate_fn=encoder_decoder_collate_fn,
    persistent_workers=True,
)

val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=encoder_decoder_collate_fn,
    persistent_workers=True,
)

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=encoder_decoder_collate_fn,
    persistent_workers=True,
)

# %% [markdown]
# ## Create Efficient Pre-training Model

# %%
# Create the efficient pre-training model
pretrain_model = MaskedTabularPretrainer(
    time_series_config,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    learning_rate=LEARNING_RATE,
    mask_probability=MASK_PROBABILITY,
    attention_type=ATTENTION_TYPE,
    attention_kwargs=ATTENTION_KWARGS,
)

print("=== Model Architecture ===")
print(f"Model parameters: {sum(p.numel() for p in pretrain_model.parameters()):,}")
print(
    "Trainable parameters: "
    f"{sum(p.numel() for p in pretrain_model.parameters() if p.requires_grad):,}"
)
print(f"Attention type: {ATTENTION_TYPE}")
print(f"Sequence length: {SEQUENCE_LENGTH}")

# %% [markdown]
# ## Custom Metrics Logger for CSV Export


# %% [markdown]
# ## Inspect Training Batch

# %%
# Test forward pass with a batch
print("\n=== Inspecting Training Batch ===")
sample_batch = next(iter(train_dl))
inputs, targets = sample_batch

# Print batch information
print("Batch shapes:")
print(f"  Numeric inputs: {inputs.numeric.shape}")
print(f"  Numeric range: [{inputs.numeric.min():.3f}, {inputs.numeric.max():.3f}]")
print(f"  Categorical inputs: {inputs.categorical.shape}")
print(f"  Unique categorical values: {inputs.categorical.unique().numel()}")

# Test masking
with torch.no_grad():
    masked_numeric, masked_categorical, numeric_targets, categorical_targets = (
        pretrain_model.mask_inputs(
            inputs.numeric,
            inputs.categorical,
        )
    )

    # Forward pass
    numeric_predictions, categorical_predictions = pretrain_model(
        input_numeric=masked_numeric,
        input_categorical=masked_categorical,
        targets_numeric=None,
        targets_categorical=None,
        deterministic=False,
    )

print(f"\nNumeric predictions shape: {numeric_predictions.shape}")
print(
    "Numeric predictions range: "
    f"[{numeric_predictions.min():.3f}, {numeric_predictions.max():.3f}]"
)

# %% [markdown]
# ## Setup Training

# %%
# Callbacks
callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=2),
    LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(
        monitor="val_total_loss",  # Updated to use the correct metric name
        dirpath=f"checkpoints/efficient_pretrain_{ATTENTION_TYPE}",
        filename="pretrain-{epoch:02d}-{val_total_loss:.4f}",
        save_top_k=3,
        mode="min",
        save_weights_only=False,
    ),
    EarlyStopping(
        monitor="val_total_loss",  # Updated to use the correct metric name
        patience=10,
        mode="min",
        verbose=True,
    ),
]

# Create timestamp for consistent naming across metrics and tensorboard
model_timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

# Logger
logger = TensorBoardLogger(
    save_dir="runs",
    name=f"efficient_pretrain_{ATTENTION_TYPE}",
    version=model_timestamp,
)

# Trainer
trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    precision="16-mixed",
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    gradient_clip_val=GRADIENT_CLIP,
    callbacks=callbacks,
    logger=logger,
    log_every_n_steps=50,
    check_val_every_n_epoch=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
)

print("=== Starting Pre-training ===")

# Add metrics logger to the trainer
metrics_save_dir = f"output_metrics/{ATTENTION_TYPE}_{model_timestamp}"
metrics_logger = MetricsLogger(save_dir=metrics_save_dir)
trainer.callbacks.append(metrics_logger)

print(f"=== Metrics will be saved to: {metrics_save_dir} ===")

# %%
# Train the model
trainer.fit(pretrain_model, train_dl, val_dl)

# %%
# Access and save metrics after training
print("\n=== Saving Metrics to CSV ===")

# Get the metrics from our custom logger
train_metrics_df = metrics_logger.get_train_dataframe()
val_metrics_df = metrics_logger.get_validation_dataframe()

# Save combined metrics
combined_metrics_filename = f"combined_metrics_{ATTENTION_TYPE}_{model_timestamp}.csv"
combined_metrics_df = metrics_logger.save_combined_metrics(combined_metrics_filename)

# Display some metrics info
if len(train_metrics_df) > 0:
    print(f"Training metrics saved: {len(train_metrics_df)} epochs")
    print(f"Final train loss: {train_metrics_df['train_loss'].iloc[-1]:.4f}")

if len(val_metrics_df) > 0:
    print(f"Validation metrics saved: {len(val_metrics_df)} epochs")
    print(f"Final val loss: {val_metrics_df['val_loss'].iloc[-1]:.4f}")

# You can also access the DataFrames directly for analysis:
# print("\nFirst few training metrics:")
# print(train_metrics_df.head())
# print("\nFirst few validation metrics:")
# print(val_metrics_df.head())

# %%
# Save the final model
model_filename = (
    "pretrained_model_efficient_"
    f"{ATTENTION_TYPE}_{SEQUENCE_LENGTH}_{model_timestamp}.pt"
)
torch.save(
    {
        "model_state_dict": pretrain_model.state_dict(),
        "config": time_series_config,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "attention_type": ATTENTION_TYPE,
        "attention_kwargs": ATTENTION_KWARGS,
        "sequence_length": SEQUENCE_LENGTH,
    },
    model_filename,
)

print(f"\nPre-training completed! Model saved to {model_filename}")

# %%
# Print final metrics are now logged to TensorBoard.
# The print statements below are removed as they are redundant.

print("\\nMetrics have been logged to TensorBoard. To view them, run:")
tensorboard_path = f"runs/efficient_pretrain_{ATTENTION_TYPE}/{model_timestamp}"
print(f"  tensorboard --logdir {tensorboard_path}")
print(f"\\nRaw metrics saved to: {metrics_save_dir}")
# Further print statements about specific metrics are removed
# as the new logging structure in TensorBoard will make these clear.
# %%
