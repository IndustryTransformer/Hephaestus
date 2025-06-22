# %% [markdown]
# # Classification Model using Penultimate Layer Features
#
# This script creates a classification model that uses the penultimate layer
# from the pre-trained w3_pre_train.py model to classify the 'class' column.
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
df = df.head(10_000_000)  # Reduce dataset size for debugging

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

# Normalize numeric columns using StandardScaler to prevent numerical instability
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [
    col for col in numeric_cols if col != "idx"
]  # Don't normalize idx
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

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
# ## Custom Classification Decoder
#


class ClassificationTabularDecoder(L.LightningModule):
    """
    TabularDecoder modified to extract penultimate layer features for classification.
    """

    def __init__(
        self,
        time_series_config,
        d_model,
        n_heads,
        n_layers: int = 4,
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

        # Create the base TimeSeriesDecoder (frozen pre-trained model)
        self.base_model = hp.timeseries_models.TimeSeriesDecoder(
            time_series_config,
            self.d_model,
            self.n_heads,
            n_layers=self.n_layers,
            attention_type=self.attention_type,
            attention_kwargs=self.attention_kwargs,
        )

        # Classification head that takes penultimate layer features
        # The penultimate layer output has shape (batch, seq, features, d_model)
        # After reshaping and pooling: (batch, features * d_model)
        feature_dim = d_model * time_series_config.n_columns

        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # Store the penultimate layer features
        self.penultimate_features = None

    def extract_penultimate_features(self, numeric, categorical):
        """Extract features from penultimate layer of the transformer."""
        # Get the transformer output before the final projection layers
        transformer_output = self.base_model.time_series_transformer(
            numeric, categorical
        )

        # transformer_output is a ProcessedEmbeddings object with value_embeddings
        # shape: (batch, n_features, seq_len, d_model)
        features = transformer_output.value_embeddings
        
        # Get the last sequence position for each feature
        # This represents the final state after processing the full sequence
        # Shape: (batch, n_features, d_model)
        last_hidden = features[:, :, -1, :]
        
        # Flatten features: (batch, n_features * d_model)
        batch_size, n_features, d_model = last_hidden.shape
        pooled_features = last_hidden.reshape(batch_size, n_features * d_model)

        return pooled_features

    def forward(self, batch):
        """Forward pass for classification."""
        # Extract penultimate layer features
        features = self.extract_penultimate_features(batch.numeric, batch.categorical)

        # Store features for inspection
        self.penultimate_features = features.detach()

        # Apply classification head
        logits = self.classification_head(features)

        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = F.cross_entropy(logits, batch.labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.labels).float().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = F.cross_entropy(logits, batch.labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch.labels).float().mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss

    def configure_optimizers(self):
        # Only train the classification head, freeze the base model
        optimizer = torch.optim.Adam(self.classification_head.parameters(), lr=1e-3)
        return optimizer


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

# %%
# Create the classification model
N_HEADS = 4
num_classes = len(unique_classes)

classification_model = ClassificationTabularDecoder(
    time_series_config,
    d_model=64,
    n_heads=N_HEADS,
    attention_type="flash",  # Use flash attention for efficiency
    num_classes=num_classes,
)

# %%
# Training setup
logger = TensorBoardLogger("runs", name=f"{dt.now()}_classification_3w_real_sample")
early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
trainer = L.Trainer(max_epochs=10, logger=logger, callbacks=[early_stopping])

train_dl = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    collate_fn=classification_collate_fn,
    num_workers=1,
    persistent_workers=True,
)

test_dl = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    collate_fn=classification_collate_fn,
    num_workers=1,
    persistent_workers=True,
)

# %%
# Train the classification model
trainer.fit(classification_model, train_dl, test_dl)

# %%
