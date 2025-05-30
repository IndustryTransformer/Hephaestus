# %% [markdown]
# # 3W Anomaly Detection with Two-Stage Training
# [Data](https://github.com/ricardovvargas/3w_dataset) is a time series dataset for
# anomaly detection.
#
# This notebook implements a two-stage training approach:
# 1. Pre-training with masked modeling (20% masking)
# 2. Fine-tuning on the class prediction task
#
# ## Load Libraries
#
# %%
import os
from datetime import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from icecream import ic
from IPython.display import display
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import hephaestus as hp
from hephaestus.timeseries_models.encoder_decoder import (
    MaskedTabularPretrainer,
    TabularEncoderDecoder,
)
from hephaestus.timeseries_models.encoder_decoder_dataset import (
    EncoderDecoderDataset,
    encoder_decoder_collate_fn,
)

torch.set_float32_matmul_precision("medium")

# %%
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("MPS available")
else:
    print("CUDA not available. Checking why...")
    import os

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# %%
# Configure output settings
ic.disable()  # Disable icecream output
ic.configureOutput(includeContext=True, contextAbsPath=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% [markdown]
# ## Load and Preprocess Data

# %%
# Load the dataset
df = pd.read_parquet("data/combined_3w_real_sample.parquet")
df = df.head(100_000)  # Reduce dataset size for debugging
df.drop(columns=["original_filename", "file_class_label", "system_id"], inplace=True)

# %%
# Create event mapping
event_values = df["class"].unique()
event_values = np.sort(event_values)
events_names = {i: str(v) for i, v in enumerate(event_values)}
df["class"] = df["class"].astype("str")  # Ensure class is string type for mapping

# %%
# Filter and preprocess
print(f"DF shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
df = df.loc[df["class"].notna()]
print(f"DF shape after filtering: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

# Convert timestamp to datetime if not already
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Convert timestamp to cyclical features
seconds = (
    df["timestamp"].dt.hour * 3600
    + df["timestamp"].dt.minute * 60
    + df["timestamp"].dt.second
)
seconds_in_day = 24 * 60 * 60
df["timestamp_sin"] = np.sin(2 * np.pi * seconds / seconds_in_day)
df["timestamp_cos"] = np.cos(2 * np.pi * seconds / seconds_in_day)

# Day of year
day_of_year = df["timestamp"].dt.dayofyear
days_in_year = 366  # Leap years included for generality
df["dayofyear_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
df["dayofyear_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)

df = df.drop(columns=["timestamp"])

# %%
# Normalize numeric columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Add index column for grouping
df["idx"] = df.index // 512  # Group by 512 rows

# %% [markdown]
# ## Create Encoder-Decoder Dataset

# %%
# Generate TimeSeriesConfig (exclude 'class' from input features)
time_series_config = hp.TimeSeriesConfig.generate(df=df, target="class")

# Split into train and test sets
train_idx = int(df.idx.max() * 0.8)
train_df = df.loc[df.idx < train_idx].copy()
test_df = df.loc[df.idx >= train_idx].copy()

event_targets = {v: k for k, v in events_names.items()}
# Create encoder-decoder datasets
train_ds = EncoderDecoderDataset(
    train_df, time_series_config, target_col="class", target_values=event_targets
)
test_ds = EncoderDecoderDataset(
    test_df, time_series_config, target_col="class", target_values=event_targets
)

print(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

# %% [markdown]
# ## Stage 1: Pre-training with Masked Modeling

# %%
# Set hyperparameters
N_HEADS = 4
D_MODEL = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
MAX_EPOCHS_PRETRAIN = 10
MAX_EPOCHS_FINETUNE = 20
MASK_PROBABILITY = 0.2
MODEL_NAME = "two_stage_encoder_decoder"

# Create the pre-training model
pretrain_model = MaskedTabularPretrainer(
    time_series_config,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    learning_rate=LEARNING_RATE,
    mask_probability=MASK_PROBABILITY,
)

# Setup pre-training
logger_pretrain = TensorBoardLogger("runs", name=f"{dt.now()}_pretrain_{MODEL_NAME}")
early_stopping_pretrain = EarlyStopping(monitor="val_loss", patience=3, mode="min")
checkpoint_pretrain = ModelCheckpoint(
    dirpath="checkpoints",
    filename=f"pretrain_{MODEL_NAME}_{{epoch}}_{{val_loss:.2f}}",
    monitor="val_loss",
    save_top_k=1,
    mode="min",
)

trainer_pretrain = L.Trainer(
    max_epochs=MAX_EPOCHS_PRETRAIN,
    logger=logger_pretrain,
    callbacks=[early_stopping_pretrain, checkpoint_pretrain],
)

# Create data loaders
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=encoder_decoder_collate_fn,
    num_workers=0,
)

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=encoder_decoder_collate_fn,
    num_workers=0,
)

# %%
# Pre-train the model
print("Starting pre-training with masked modeling...")
trainer_pretrain.fit(pretrain_model, train_dl, test_dl)

# %% [markdown]
# ## Stage 2: Fine-tuning on Classification Task

# %%
# Load the best pre-trained model
best_pretrain_path = checkpoint_pretrain.best_model_path
print(f"Loading pre-trained model from: {best_pretrain_path}")
pretrained = MaskedTabularPretrainer.load_from_checkpoint(
    best_pretrain_path,
    config=time_series_config,
)

# Create fine-tuning model with pre-trained encoder
finetune_model = TabularEncoderDecoder(
    time_series_config,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    learning_rate=LEARNING_RATE / 10,  # Lower learning rate for fine-tuning
    classification_values=list(events_names.values()),
    pretrained_encoder=pretrained.encoder,  # Use the pre-trained encoder
)

# Freeze the encoder for fine-tuning
finetune_model.freeze_encoder()
print("Encoder frozen for fine-tuning")

# Setup fine-tuning
logger_finetune = TensorBoardLogger("runs", name=f"{dt.now()}_finetune_{MODEL_NAME}")
early_stopping_finetune = EarlyStopping(monitor="val_loss", patience=5, mode="min")
checkpoint_finetune = ModelCheckpoint(
    dirpath="checkpoints",
    filename=f"finetune_{MODEL_NAME}_{{epoch}}_{{val_loss:.2f}}",
    monitor="val_loss",
    save_top_k=1,
    mode="min",
)

trainer_finetune = L.Trainer(
    max_epochs=MAX_EPOCHS_FINETUNE,
    logger=logger_finetune,
    callbacks=[early_stopping_finetune, checkpoint_finetune],
)

# %%
# Fine-tune the model
print("Starting fine-tuning on classification task...")
trainer_finetune.fit(finetune_model, train_dl, test_dl)

# %% [markdown]
# ## Evaluate the Fine-tuned Model

# %%
# Use PyTorch Lightning's built-in prediction functionality
print("Starting prediction with PyTorch Lightning...")

# Load best fine-tuned model
best_finetune_path = checkpoint_finetune.best_model_path
print(f"Loading fine-tuned model from: {best_finetune_path}")
best_model = TabularEncoderDecoder.load_from_checkpoint(
    best_finetune_path,
    config=time_series_config,
    pretrained_encoder=pretrained.encoder,
)

# Predict
predictions_list = trainer_finetune.predict(best_model, test_dl)

# Process predictions
all_preds = []
all_targets = []

for pred_batch in predictions_list:
    batch_preds = pred_batch["predictions"]  # [batch, seq_len]
    batch_targets = pred_batch["targets"]  # [batch, seq_len]

    # Flatten predictions and targets
    batch_preds = batch_preds.reshape(-1).cpu().numpy()
    batch_targets = batch_targets.reshape(-1).cpu().numpy()

    # Filter out any NaN targets
    valid_mask = ~np.isnan(batch_targets)
    batch_preds = batch_preds[valid_mask]
    batch_targets = batch_targets[valid_mask]

    all_preds.extend(batch_preds)
    all_targets.extend(batch_targets)

print(f"Done predicting! Collected {len(all_preds):,} predictions.")

# Convert to numpy arrays
predictions = np.array(all_preds).astype(int)
targets = np.array(all_targets).astype(int)

# %%
# Calculate metrics
print(f"Total predictions: {len(predictions):,}")

# Create confusion matrix
cm = confusion_matrix(targets, predictions)

# Find the unique classes that are actually present in the data
unique_classes = np.unique(np.concatenate([predictions, targets]))
print(f"Unique classes in predictions and targets: {unique_classes}")

# Map numeric classes back to event names
class_names = [events_names[cls] for cls in unique_classes]
print(f"Classes present in confusion matrix: {class_names}")

# Calculate overall accuracy
accuracy = np.mean(predictions == targets)
print(f"\nOverall accuracy: {accuracy:.4f}")

# Get per-class metrics
print("\nClassification Report:")
print(classification_report(targets, predictions, target_names=class_names))

# %% [markdown]
# ## Visualize Results

# %%
# Create DataFrame for visualization
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df = cm_df.reset_index().melt(id_vars="index")
cm_df.columns = ["true", "predicted", "count"]

# Create heatmap using Altair
confusion_chart = (
    alt.Chart(cm_df)
    .mark_rect()
    .encode(
        x=alt.X("predicted:N", title="Predicted"),
        y=alt.Y("true:N", title="True"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["true", "predicted", "count"],
    )
    .properties(
        title="Confusion Matrix - Two-Stage Training",
        width=400,
        height=400,
    )
)

# Add text overlay
text_overlay = (
    alt.Chart(cm_df)
    .mark_text(color="black")
    .encode(
        x=alt.X("predicted:N"),
        y=alt.Y("true:N"),
        text=alt.Text("count:Q", format=",d"),
    )
)

# Combine charts
confusion_plot = confusion_chart + text_overlay
display(confusion_plot)

# %%
# Create normalized confusion matrix
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
cm_norm_df = cm_norm_df.reset_index().melt(id_vars="index")
cm_norm_df.columns = ["true", "predicted", "percentage"]

# Create normalized confusion matrix heatmap
norm_confusion_chart = (
    alt.Chart(cm_norm_df)
    .mark_rect()
    .encode(
        x=alt.X("predicted:N", title="Predicted"),
        y=alt.Y("true:N", title="True"),
        color=alt.Color("percentage:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["true", "predicted", alt.Tooltip("percentage:Q", format=".2%")],
    )
    .properties(
        title="Normalized Confusion Matrix - Two-Stage Training",
        width=400,
        height=400,
    )
)

# Add text overlay for percentage values
norm_text_overlay = (
    alt.Chart(cm_norm_df)
    .mark_text(color="black")
    .encode(
        x=alt.X("predicted:N"),
        y=alt.Y("true:N"),
        text=alt.Text("percentage:Q", format=".2%"),
    )
)

# Combine charts
norm_confusion_plot = norm_confusion_chart + norm_text_overlay
display(norm_confusion_plot)

# %%
print("\nTwo-stage training complete!")
print(f"Pre-training epochs: {trainer_pretrain.current_epoch}")
print(f"Fine-tuning epochs: {trainer_finetune.current_epoch}")
print(f"Final accuracy: {accuracy:.4f}")
