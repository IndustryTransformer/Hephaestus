# %% [markdown]
# # 3W Anomaly Detection with Encoder-Decoder Architecture
# [Data](https://github.com/ricardovvargas/3w_dataset) is a time series dataset for
# anomaly detection.
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
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import hephaestus as hp
from hephaestus.timeseries_models.encoder_decoder import TabularEncoderDecoder
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
# Define the columns


# Load the dataset
df = pd.read_parquet("data/combined_3w_real_sample.parquet")
df = df.head(100_000)  # Reduce dataset size for debugging
df.drop(columns=["original_filename", "file_class_label", "system_id"], inplace=True)
# %%
# %%
# %%
# Create a dictionary mapping numeric class values to event names
# events_names = {
#     0: "Normal",
#     1: "Abrupt Increase of BSW",
#     2: "Spurious Closure of DHSV",
#     3: "Severe Slugging",
#     4: "Flow Instability",
#     5: "Rapid Productivity Loss",
#     6: "Quick Restriction in PCK",
#     7: "Scaling in PCK",
#     8: "Hydrate in Production Line",
#     -1: "missing",
# }
event_values = df["class"].unique()
event_values = np.sort(event_values)
events_names = {i: str(v) for i, v in enumerate(event_values)}
df["class"] = df["class"].astype("str")  # Ensure class is string type for mapping

# %%
# Apply the mapping to the 'class' column
# df["class"] = df["class"].map(events_names)
print(f"DF shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
df = df.loc[df["class"].notna()]
print(f"DF shape after filtering: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
# df = df.sort_values("timestamp")
# Convert timestamp to datetime if not already
df["timestamp"] = pd.to_datetime(df["timestamp"])
plot_df = df.head(1000)
COLUMN_NAME = "T-TPT"

chart = (
    alt.Chart(plot_df)
    .mark_line()
    .encode(
        x=alt.X("timestamp:T", title="Timestamp"),
        y=alt.Y(f"{COLUMN_NAME}:Q", title=COLUMN_NAME),
        tooltip=["timestamp:T", f"{COLUMN_NAME}:Q"],
    )
    .properties(
        title="P-PDG vs Timestamp (First 500 Rows)",
        width=800,
        height=300,
    )
)
display(chart)
# Seconds in day
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
seconds_in_day = 24 * 60 * 60

df = df.drop(columns=["timestamp"])
# Plot "P-PDG" against timestamp, ordered by timestamp, first 500 rows using Altair

# %%
# Convert timestamp to cyclical features (sin/cos)
# Reload the original parquet to get the timestamp column


# Normalize numeric columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Add index column for grouping
df["idx"] = df.index // 512  # Group by 1024 rows (1 second intervals)

# %% [markdown]
# ## Create Encoder-Decoder Dataset

# %%
# Generate TimeSeriesConfig (exclude 'class' from input features)
time_series_config = hp.TimeSeriesConfig.generate(df=df, target="class")

# Split into train and test sets
train_idx = int(df.idx.max() * 0.8)
train_df = df.loc[df.idx < train_idx].copy()
# train_df = train_df.head(10000)  # Limit to 10k samples for faster training
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
# ## Create and Train the Encoder-Decoder Model

# %%
# Set hyperparameters
N_HEADS = 4
D_MODEL = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
MAX_EPOCHS = 20
MODEL_NAME = "smaller_encoder_decoder"
# Create the encoder-decoder model
encoder_decoder_model = TabularEncoderDecoder(
    time_series_config,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    learning_rate=LEARNING_RATE,
    classification_values=list(events_names.values()),
)

# Setup training with PyTorch Lightning
logger = TensorBoardLogger("runs", name=f"{dt.now()}_{MODEL_NAME}")
early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    logger=logger,
    callbacks=[early_stopping],
)

# Create data loaders
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=encoder_decoder_collate_fn,
    num_workers=0,  # Disable workers for debugging
    # persistent_workers=True,
)

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=encoder_decoder_collate_fn,
    num_workers=0,  # Disable workers for debugging
    # persistent_workers=True,
)


# %% [markdown]
# ## Test model forward step
# %%
inputs, targets = train_ds[0:32]
# Forward pass through the model
output = encoder_decoder_model(
    input_numeric=inputs.numeric,
    input_categorical=inputs.categorical,
    deterministic=True,  # Set to True for inference
)
# %%
# Train the model
trainer.fit(encoder_decoder_model, train_dl, test_dl)

# %% [markdown]
# ## Evaluate
# %%
# Use PyTorch Lightning's built-in prediction functionality for efficiency
print("Starting prediction with PyTorch Lightning...")

# Using the trainer's predict method is much faster than manual looping
predictions_list = trainer.predict(encoder_decoder_model, test_dl)

# Process all predictions at once
all_preds = []
all_targets = []

# Extract predictions and targets from each batch result
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
print(f"Total predictions: {len(predictions):,}")
# Create confusion matrix
cm = confusion_matrix(targets, predictions)

# Find the unique classes that are actually present in the data
unique_classes = np.unique(np.concatenate([predictions, targets]))
print(f"Unique classes in predictions and targets: {unique_classes}")

# Map numeric classes back to event names for better readability
# Use only the class names that are actually present in the data
class_names = [events_names[cls] for cls in unique_classes]
print(f"Classes present in confusion matrix: {class_names}")

# Create DataFrame for Altair visualization
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
        title="Confusion Matrix",
        width=400,
        height=400,
    )
)

# Add text overlay for cell values
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

# Display chart
print("Confusion Matrix created")
# To display the chart in the notebook, uncomment the line below
# display(confusion_plot)

# %% [markdown]
# ## Calculate Accuracy and Other Metrics

# %%
# Calculate overall accuracy
accuracy = np.mean(predictions == targets)
print(f"Overall accuracy: {accuracy:.4f}")

# Get per-class metrics

print("\nClassification Report:")
print(classification_report(targets, predictions, target_names=class_names))
# %%
# Create a normalized confusion matrix for better visualization
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
        title="Normalized Confusion Matrix",
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

# Display normalized confusion matrix
print("Normalized Confusion Matrix created")
# To display the chart in the notebook, uncomment the line below

# %%
norm_confusion_plot  # noqa B018

# %%
