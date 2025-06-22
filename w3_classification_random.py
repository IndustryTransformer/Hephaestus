# %% [markdown]
# # Classification Model with Randomly Initialized Transformer
#
# This script creates a classification model using a randomly initialized
# TimeSeriesDecoder to classify the 'class' column, instead of trying to
# extract features from a pre-trained model.
#

# %%
import os
from collections import namedtuple
from datetime import datetime as dt

import icecream
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import hephaestus as hp
from hephaestus.timeseries_models import tabular_collate_fn

# Define namedtuples at module level for pickling
ClassificationBatch = namedtuple(
    "ClassificationBatch", ["numeric", "categorical", "labels"]
)
TempBatch = namedtuple("TempBatch", ["numeric", "categorical"])

# %%
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("medium")

# %%
icecream.install()
ic_disable = True  # Global variable to disable ic
if ic_disable:
    ic.disable()
ic.configureOutput(includeContext=True, contextAbsPath=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEQUENCE_LENGTH = 512

# %%
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")  # type: ignore
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("MPS available")
else:
    print("CUDA not available. Checking why...")
    import os

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# %% [markdown]
# ## Load Data with Class Column Preserved
#

# %%
df = pd.read_parquet("data/combined_3w_real_sample.parquet")
df = df.head(10_000_000)  # Much smaller dataset for debugging

# Separate class column before preprocessing and convert to string
class_labels = df["class"].astype(str).copy()

# Drop other metadata columns but keep class for now
df.drop(
    columns=[
        "original_filename",
        "file_class_label",
        "system_id",
        "timestamp",
        "instance_id",
        "state",
    ],
    inplace=True,
)

# Convert class to string type before processing
df["class"] = df["class"].astype(str)

# Drop all-NaN columns
df = df.dropna(axis=1, how="all")

# Drop constant columns (no variance)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].nunique() <= 1:
        df = df.drop(columns=[col])
        print(f"Dropped constant column: {col}")

# Fill NaN values first to prevent numerical instability
df = df.fillna(0)

# Normalize numeric columns using StandardScaler to prevent numerical instability
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col != "idx"]  # Don't normalize idx
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Check for any remaining NaN or inf values
numeric_df = df.select_dtypes(include=[np.number])
print(f"NaN values after preprocessing: {numeric_df.isna().sum().sum()}")
print(f"Inf values after preprocessing: {np.isinf(numeric_df).sum().sum()}")

# Create idx to group rows into sequences of SEQUENCE_LENGTH
df["idx"] = np.arange(len(df)) // SEQUENCE_LENGTH

# Store class labels for each sequence (use mode for each sequence)
class_df = pd.DataFrame({"idx": df["idx"], "class": df["class"]})
sequence_classes = class_df.groupby("idx")["class"].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
)

# Create class label encoding
unique_classes = sorted(sequence_classes.unique())
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

# Encode sequence classes as integers
sequence_classes_encoded = sequence_classes.map(class_to_idx)

# Remove class from the main dataframe for feature processing
df = df.drop(columns=["class"])

# %%
df.head()

# %%
# Get categorical columns
df_categorical = df.select_dtypes(include=["object"]).astype(str)
unique_values_per_column = df_categorical.apply(pd.Series.unique).values
flattened_unique_values = np.concatenate(unique_values_per_column).tolist()
unique_values = list(set(flattened_unique_values))
unique_values  # noqa: B018

# %%
df = df.reset_index(drop=True)

# %% [markdown]
# ## Simple Classification Model with Pooling
#


class SimpleClassificationModel(L.LightningModule):
    """
    Simple classification model that processes sequences and predicts class labels.
    Uses pooling over sequence dimension for classification.
    """

    def __init__(
        self,
        time_series_config,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        attention_type: str = "standard",
        attention_kwargs: dict = None,
        num_classes: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_type = attention_type
        self.attention_kwargs = attention_kwargs or {}
        self.num_classes = num_classes
        self.time_series_config = time_series_config

        # Create a randomly initialized transformer
        self.transformer = hp.timeseries_models.TimeSeriesTransformer(
            config=time_series_config,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            attention_type=self.attention_type,
            attention_kwargs=self.attention_kwargs,
        )

        # Classification head
        feature_dim = d_model * time_series_config.n_columns
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),  # Add layer norm for stability
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, batch):
        """Forward pass for classification."""
        # Check inputs for NaN/Inf
        if torch.isnan(batch.numeric).any() or torch.isinf(batch.numeric).any():
            print("NaN/Inf detected in numeric input!")

        # Get transformer embeddings
        embeddings = self.transformer(batch.numeric, batch.categorical)

        # embeddings.value_embeddings shape: (batch, n_features, seq_len, d_model)
        features = embeddings.value_embeddings

        # Check transformer output
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("NaN/Inf detected in transformer output!")
            print(f"Features stats: min={features.min()}, max={features.max()}")

        # Apply global average pooling over sequence dimension manually
        # Shape: (batch, n_features, d_model)
        pooled = features.mean(dim=2)

        # Flatten features: (batch, n_features * d_model)
        batch_size = pooled.shape[0]
        flattened = pooled.view(batch_size, -1)

        # Check pooled features
        if torch.isnan(flattened).any() or torch.isinf(flattened).any():
            print("NaN/Inf detected in pooled features!")

        # Apply classifier
        logits = self.classifier(flattened)

        # Check final output
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN/Inf detected in logits!")
            print(f"Logits stats: min={logits.min()}, max={logits.max()}")

        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)

        # Use class weights if available
        if hasattr(self, "class_weights"):
            weight = self.class_weights.to(logits.device)
            loss = F.cross_entropy(logits, batch.labels, weight=weight)
        else:
            loss = F.cross_entropy(logits, batch.labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)

        # Use class weights if available
        if hasattr(self, "class_weights"):
            weight = self.class_weights.to(logits.device)
            loss = F.cross_entropy(logits, batch.labels, weight=weight)
        else:
            loss = F.cross_entropy(logits, batch.labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Train the entire model with lower learning rate for stability
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# %% [markdown]
# ## Custom Dataset for Classification
#


class ClassificationTimeSeriesDS(torch.utils.data.Dataset):
    """Dataset that includes class labels for each sequence."""

    def __init__(self, df, time_series_config, sequence_classes_encoded):
        self.base_dataset = hp.TimeSeriesDS(df, time_series_config)
        self.sequence_classes = sequence_classes_encoded

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get the base item
        item = self.base_dataset[idx]

        # The base dataset returns sequences by unique index
        # Get the sequence index from the base dataset's unique_indices
        seq_idx = self.base_dataset.unique_indices[idx]

        # Get the class label for this sequence
        if seq_idx in self.sequence_classes.index:
            class_label = self.sequence_classes.loc[seq_idx]
        else:
            # Use the most common class as fallback
            class_label = (
                self.sequence_classes.mode().iloc[0]
                if len(self.sequence_classes) > 0
                else 0
            )

        # Return the classification batch
        return ClassificationBatch(
            numeric=item.numeric,
            categorical=item.categorical,
            labels=torch.tensor(class_label, dtype=torch.long),
        )


def classification_collate_fn(batch):
    """Custom collate function for classification dataset."""
    # Separate the components
    numeric_batch = [item.numeric for item in batch]
    categorical_batch = [item.categorical for item in batch]
    labels_batch = [item.labels for item in batch]

    # Use the original collate function for numeric and categorical
    temp_batch = [TempBatch(n, c) for n, c in zip(numeric_batch, categorical_batch)]
    collated = tabular_collate_fn(temp_batch)

    # Stack the labels
    labels = torch.stack(labels_batch)

    # Create final batch
    return ClassificationBatch(
        numeric=collated.numeric, categorical=collated.categorical, labels=labels
    )


# %%
# Get train test split at 80/20
time_series_config = hp.TimeSeriesConfig.generate(df=df)
train_idx = int(df.idx.max() * 0.8)
train_df = df.loc[df.idx < train_idx].copy()
test_df = df.loc[df.idx >= train_idx].copy()

# Split sequence classes accordingly
train_sequence_classes = sequence_classes_encoded.loc[
    sequence_classes_encoded.index < train_idx
]
test_sequence_classes = sequence_classes_encoded.loc[
    sequence_classes_encoded.index >= train_idx
]

# Create datasets
train_ds = ClassificationTimeSeriesDS(
    train_df, time_series_config, train_sequence_classes
)
test_ds = ClassificationTimeSeriesDS(test_df, time_series_config, test_sequence_classes)

print(f"Train dataset size: {len(train_ds)}")
print(f"Test dataset size: {len(test_ds)}")
print(f"Number of unique classes: {len(unique_classes)}")
print(f"Class mapping: {class_to_idx}")
print(f"Class distribution: {sequence_classes_encoded.value_counts().sort_index()}")

# Calculate class weights to handle imbalance
class_counts = sequence_classes_encoded.value_counts().sort_index()
total_samples = len(sequence_classes_encoded)
class_weights = total_samples / (len(unique_classes) * class_counts)
class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)
print(f"Class weights: {class_weights_tensor}")

# %%
# Create the classification model
N_HEADS = 4
num_classes = len(unique_classes)

classification_model = SimpleClassificationModel(
    time_series_config,
    d_model=64,
    n_heads=N_HEADS,
    n_layers=2,  # Shallower model for stability
    attention_type="standard",  # Use standard attention for stability
    num_classes=num_classes,
)

# Store class weights in the model for use in loss calculation
classification_model.class_weights = class_weights_tensor

# %%
# Training setup
logger = TensorBoardLogger("runs", name=f"{dt.now()}_simple_classification_3w")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
trainer = L.Trainer(
    max_epochs=20,
    logger=logger,
    callbacks=[early_stopping],
    enable_progress_bar=True,
    gradient_clip_val=0.5,  # More aggressive gradient clipping
    precision="16-mixed",  # Use mixed precision for stability
)

train_dl = DataLoader(
    train_ds,
    batch_size=16,  # Smaller batch size for stability
    shuffle=True,
    collate_fn=classification_collate_fn,
    num_workers=0,  # Disable multiprocessing to avoid pickling issues
    persistent_workers=False,
)

test_dl = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    collate_fn=classification_collate_fn,
    num_workers=0,
    persistent_workers=False,
)

# %%
# Train the classification model
trainer.fit(classification_model, train_dl, test_dl)

# %%
