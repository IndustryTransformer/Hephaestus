# ruff: noq: F402
# %% tags=["hide-input", "hide-output"]
from IPython.display import Markdown  # noqa: F401

Markdown("""# Branking Fraud Detection with tabNCT


This notebook demonstrates how to use the tabNCT model for banking fraud detection.

## Load Libraries and Prepare Data

""")
# %%
import os
from datetime import datetime as dt
from pathlib import Path

# ruff: noqa: E402
import numpy as np
import pandas as pd


import pytorch_lightning as L  # noqa: N812
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

import hephaestus.single_row_models as sr

# ... other imports ...

# Set precision for Tensor Cores (choose 'high' or 'medium')
torch.set_float32_matmul_precision("high")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


Markdown("""## tabNCT Parameters
We will use the following parameters for the tabNCT model:
""")

D_MODEL = 128
N_HEADS = 4
LR = 0.0001
BATCH_SIZE = 64  # Smaller batch sizes lead to better predictions because outliers are
# better trained on.
name = "BankingFraudDetection"
LOGGER_VARIANT_NAME = f"{name}_D{D_MODEL}_H{N_HEADS}_LR{LR}"
LABEL_RATIO = 1.0

# Load and preprocess the train_dataset (assuming you have a CSV file)

# Concatenate all dataframes
df = pd.read_csv("data/bankmarketing_train.csv")
df.columns = df.columns.str.lower()

# df = df.rename(columns={"nox": "target"})  # Not needed for banking fraud
# scale the non-target numerical columns
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[float, int]).columns
# numeric_cols = numeric_cols.drop("target")
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# df["target"] = df["target"] + 100
df.head()

# %%
Markdown("""### Model Initialization

Initialize the model and create dataloaders for training and validation.
""")
# %%
X_setup = df[df.columns.drop("y")]
# y = df["target"]

model_config_mtm = sr.SingleRowConfig.generate(X_setup)  # Full dataset - target
model_config_clf = sr.SingleRowConfig.generate(
    df, target="y", target_type="binary_classification"
)  # Full dataset with target for classification
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

X_train_sub_set = df_train.drop(columns=["y"])
y_train_sub_set = df_train["y"]
X_test = df_test.drop(columns=["y"])
y_test = df_test["y"]
train_dataset = sr.TabularDS(X_train_sub_set, model_config_mtm)
test_dataset = sr.TabularDS(X_test, model_config_mtm)


mtm_model = sr.MaskedTabularModeling(
    model_config_mtm, d_model=D_MODEL, n_heads=N_HEADS, lr=LR
)
mtm_model.predict_step(train_dataset[0:10].inputs)

# %%
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=sr.training.masked_tabular_collate_fn,
)
val_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=sr.training.masked_tabular_collate_fn,
)


# %%
Markdown("""### Model Training
Using PyTorch Lightning, we will train the model using the training and validation
""")

retrain_model = False
pretrained_model_dir = Path("checkpoints/banking_fraud")
pre_trained_models = list(pretrained_model_dir.glob("*.ckpt"))
# Check if a model with the exact name exists or if retraining is forced
if retrain_model or not any(LOGGER_VARIANT_NAME in p.stem for p in pre_trained_models):
    print("Retraining model or specified model not found.")
    run_trainer = True
else:
    print("Attempting to load pre-trained model.")
    run_trainer = False


if run_trainer:
    logger_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
    logger_name = f"{logger_time}_{LOGGER_VARIANT_NAME}"
    print(f"Using logger name: {logger_name}")
    logger = TensorBoardLogger(
        "runs",
        name=logger_name,
    )
    model_summary = ModelSummary(max_depth=3)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, min_delta=0.001, mode="min"
    )
    progress_bar = TQDMProgressBar(leave=False)
    trainer = L.Trainer(
        max_epochs=200,
        logger=logger,
        callbacks=[early_stopping, progress_bar, model_summary],
        log_every_n_steps=1,
    )
    trainer.fit(
        mtm_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.save_checkpoint(
        os.path.join(
            "checkpoints",
            "banking_fraud",
            f"{LOGGER_VARIANT_NAME}_{trainer.logger.name}.ckpt",
        )
    )

else:  # Find the checkpoint file matching the LOGGER_VARIANT_NAME prefix
    # Ensure the directory exists before searching
    pretrained_model_dir.mkdir(parents=True, exist_ok=True)

    found_checkpoints = list(pretrained_model_dir.glob(f"{LOGGER_VARIANT_NAME}*.ckpt"))

    if not found_checkpoints:
        # Handle the case where no matching checkpoint is found
        print(
            f"📭 No checkpoint found starting with {LOGGER_VARIANT_NAME} in {pretrained_model_dir}. Training model instead."
        )
        run_trainer = True  # Set to train if checkpoint not found
    elif len(found_checkpoints) > 1:
        # Handle ambiguity if multiple checkpoints match (e.g., load the latest)
        # For now, let's load the first one found as an example
        print(
            f"‼️ Warning: Found multiple checkpoints for {LOGGER_VARIANT_NAME}. Loading the first one: {found_checkpoints[0]}"
        )
        checkpoint_path = found_checkpoints[0]
        print(f"Loading checkpoint: {checkpoint_path}")
        mtm_model = sr.MaskedTabularModeling.load_from_checkpoint(
            checkpoint_path,
            model_config=model_config_mtm,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            lr=LR,
        )

    else:
        # Exactly one checkpoint found
        checkpoint_path = found_checkpoints[0]
        print(f"Loading checkpoint: {checkpoint_path}")
        mtm_model = sr.MaskedTabularModeling.load_from_checkpoint(checkpoint_path)

# %%

Markdown("### Init Classifier")
classifier = sr.TabularClassifier(
    model_config=model_config_clf,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_classes=model_config_clf.n_classes,  # Use the number of classes from config
    lr=LR,
)
# Transfer the pre-trained encoder weights from the masked tabular model
classifier.model.tabular_encoder = mtm_model.model.tabular_encoder

# %% Train Classifier
# Create datasets for classification training
train_dataset_clf = sr.TabularDS(df_train, model_config_clf)
test_dataset_clf = sr.TabularDS(df_test, model_config_clf)

train_dataloader_clf = torch.utils.data.DataLoader(
    train_dataset_clf,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=sr.training.tabular_collate_fn,  # Use classification-friendly collate function
)
val_dataloader_clf = torch.utils.data.DataLoader(
    test_dataset_clf,
    batch_size=BATCH_SIZE,
    collate_fn=sr.training.tabular_collate_fn,
)

# Train the classifier
retrain_classifier = False
classifier_model_dir = Path("checkpoints/banking_fraud_classifier")
classifier_pre_trained_models = list(classifier_model_dir.glob("*.ckpt"))
classifier_variant_name = f"{name}_classifier_D{D_MODEL}_H{N_HEADS}_LR{LR}"

# Check if a classifier model with the exact name exists or if retraining is forced
if retrain_classifier or not any(
    classifier_variant_name in p.stem for p in classifier_pre_trained_models
):
    print("Training classifier model or specified model not found.")
    run_classifier_trainer = True
else:
    print("Attempting to load pre-trained classifier model.")
    run_classifier_trainer = False

if run_classifier_trainer:
    logger_time = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
    logger_name = f"{logger_time}_{classifier_variant_name}"
    print(f"Using classifier logger name: {logger_name}")
    logger = TensorBoardLogger(
        "runs",
        name=logger_name,
    )
    model_summary = ModelSummary(max_depth=3)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, min_delta=0.001, mode="min"
    )
    progress_bar = TQDMProgressBar(leave=False)
    trainer = L.Trainer(
        max_epochs=100,  # Fewer epochs for fine-tuning
        logger=logger,
        callbacks=[early_stopping, progress_bar, model_summary],
        log_every_n_steps=1,
    )
    trainer.fit(
        classifier,
        train_dataloaders=train_dataloader_clf,
        val_dataloaders=val_dataloader_clf,
    )
    # Save the trained classifier
    classifier_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(
        os.path.join(
            "checkpoints",
            "banking_fraud_classifier",
            f"{classifier_variant_name}_{trainer.logger.name}.ckpt",
        )
    )
else:
    # Load pre-trained classifier
    classifier_model_dir.mkdir(parents=True, exist_ok=True)
    found_classifier_checkpoints = list(
        classifier_model_dir.glob(f"{classifier_variant_name}*.ckpt")
    )

    if not found_classifier_checkpoints:
        print(
            f"📭 No classifier checkpoint found starting with {classifier_variant_name}. Training instead."
        )
        run_classifier_trainer = True
    else:
        checkpoint_path = found_classifier_checkpoints[0]
        print(f"Loading classifier checkpoint: {checkpoint_path}")
        classifier = sr.TabularClassifier.load_from_checkpoint(
            checkpoint_path,
            model_config=model_config_clf,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_classes=model_config_clf.n_classes,
            lr=LR,
        )

# %% Evaluate on Test Set
Markdown("""## Test Set Evaluation

Calculate accuracy metrics on the test dataset.
""")

# Set model to evaluation mode
classifier.eval()

# Get predictions on test set
all_predictions = []
all_probabilities = []
all_targets = []

with torch.no_grad():
    for batch in val_dataloader_clf:
        X = batch.inputs
        y = batch.target
        y_hat = classifier.model(X.numeric, X.categorical)
        
        # Get predictions and probabilities
        probs = torch.softmax(y_hat, dim=1)
        preds = torch.argmax(y_hat, dim=1)
        
        # Convert to numpy and store
        all_predictions.extend(preds.cpu().numpy())
        all_probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        all_targets.extend(y.long().squeeze(-1).cpu().numpy())

# Convert to numpy arrays
y_true = np.array(all_targets)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)

# Calculate metrics
test_accuracy = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred)
test_auc = roc_auc_score(y_true, y_prob)

# Print results
print("=" * 50)
print("TEST SET EVALUATION RESULTS")
print("=" * 50)
print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"F1 Score: {test_f1:.4f}")
print(f"AUC-ROC:  {test_auc:.4f}")
print("=" * 50)
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['No Fraud', 'Fraud']))

# %%
