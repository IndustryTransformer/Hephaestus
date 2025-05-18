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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as L
import seaborn as sns
import torch
from icecream import ic
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
columns = [
    "timestamp",
    "P-PDG",
    "P-TPT",
    "T-TPT",
    "P-MON-CKP",
    "T-JUS-CKP",
    "P-JUS-CKGL",
    "T-JUS-CKGL",
    "QGL",
    "class",
]

# Load the dataset
df = pd.read_parquet("data/3w_dataset/3w_dataset.parquet")

# Create a dictionary mapping numeric class values to event names
events_names = {
    0: "Normal",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
}

# Apply the mapping to the 'class' column
df["class"] = df["class"].map(events_names)

# %%
# Add some categorical columns for demonstration
df["dummy_category"] = "dummy"
df["dummy_category_2"] = "dummy2"

# Drop timestamp column
df = df.drop(columns=["timestamp"])  # Eventually convert to numeric with sin/cos

# Normalize numeric columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Add index column for grouping
df["idx"] = df.index // 128

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
# ## Create and Train the Encoder-Decoder Model

# %%
# Set hyperparameters
N_HEADS = 8 * 4
D_MODEL = 512
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 10

# Create the encoder-decoder model
encoder_decoder_model = TabularEncoderDecoder(
    time_series_config,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    learning_rate=LEARNING_RATE,
    classification_values=list(events_names.values()),
)

# Setup training with PyTorch Lightning
logger = TensorBoardLogger("runs", name=f"{dt.now()}_tabular_encoder_decoder")
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
    num_workers=1,
    persistent_workers=True,
)

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=encoder_decoder_collate_fn,
    num_workers=1,
    persistent_workers=True,
)

# Test a prediction before training
sample_batch = next(iter(train_dl))
encoder_decoder_model.predict_step(sample_batch, batch_idx=0)

# %%
# Train the model
trainer.fit(encoder_decoder_model, train_dl, test_dl)

# %% [markdown]
# ## Evaluate the Model

# %%    # Make predictions on test data
predictions = []
targets = []

for batch in test_dl:
    inputs, target_batch = batch
    pred_output = encoder_decoder_model.predict_step(batch, batch_idx=0)

    # Get class predictions for all elements in the sequence
    batch_preds = pred_output["class_predictions"].cpu().numpy()

    # Get target classes from categorical targets
    class_col_idx = None
    for i, col_name in enumerate(time_series_config.categorical_col_tokens):
        if col_name == "class":
            class_col_idx = i
            break

    batch_targets = target_batch.categorical[:, class_col_idx, :].cpu().numpy()

    # Flatten the predictions and targets
    predictions.extend(batch_preds.flatten())
    targets.extend(batch_targets.flatten())

# Convert token indices back to class names
id_to_class = {
    v: k for k, v in time_series_config.token_dict.items() if k in events_names.values()
}

pred_classes = [id_to_class.get(p) for p in predictions]
target_classes = [id_to_class.get(t) for t in targets]

# Print classification report
print("\nClassification Report:")
print(classification_report(target_classes, pred_classes))

# Plot confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(target_classes, pred_classes, normalize="true")
sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=events_names.values(),
    yticklabels=events_names.values(),
)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("result_images/anomaly_confusion_matrix.png")
plt.show()

# %% [markdown]
# ## Compare with Previous Decoder-Only Model

# %%
# Load a pre-trained decoder-only model if available
try:
    decoder_only_path = "checkpoints/tabular_decoder_anomaly.ckpt"
    decoder_only_model = hp.TabularDecoder.load_from_checkpoint(
        decoder_only_path, config=time_series_config
    )

    # Make predictions with decoder-only model
    decoder_predictions = []

    for batch in test_dl:
        inputs, targets = batch
        # Combine inputs for decoder-only model
        combined_inputs = hp.NumericCategoricalData(
            numeric=inputs.numeric, categorical=inputs.categorical
        )

        decoder_output = decoder_only_model.predict_step(combined_inputs)
        # Extract class predictions
        # This will depend on your decoder implementation

        # Add to predictions list
        decoder_predictions.extend(decoder_output.cpu().numpy().flatten())

    # Compare results
    print("\nEncoder-Decoder vs Decoder-Only Accuracy:")
    encoder_decoder_acc = (np.array(predictions) == np.array(targets)).mean()
    decoder_only_acc = (np.array(decoder_predictions) == np.array(targets)).mean()

    print(f"Encoder-Decoder: {encoder_decoder_acc:.4f}")
    print(f"Decoder-Only: {decoder_only_acc:.4f}")

    # Plot comparison
    plt.figure(figsize=(8, 6))
    models = ["Encoder-Decoder", "Decoder-Only"]
    accuracies = [encoder_decoder_acc, decoder_only_acc]

    plt.bar(models, accuracies, color=["#3498db", "#e74c3c"])
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.ylim(0, 1.0)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig("result_images/anomaly_model_comparison.png")
    plt.show()

except Exception as e:
    print(f"Could not load decoder-only model for comparison: {e}")
    print("Skipping comparison...")

# %%
