# %%
import ast
import os
import re
from datetime import datetime as dt

import icecream
import numpy as np
import pandas as pd
import torch

# Add these imports
from icecream import ic
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from transformers import BertTokenizerFast, FlaxBertModel

from hephaestus.models.models import TimeSeriesConfig, TimeSeriesDecoder, TimeSeriesDS
from hephaestus.training.training_loop import train_model  # Fixed import path

# %%
torch.set_default_dtype(torch.float32)
print("Hello")
# %%
icecream.install()
ic_disable = False  # Global variable to disable ic
if ic_disable:
    ic.disable()
ic.configureOutput(includeContext=True, contextAbsPath=True)
pd.options.mode.copy_on_write = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
model = FlaxBertModel.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)
batch_size = 16


# %%
def line2df(line, idx):
    data_rows = []
    line = ast.literal_eval(line)
    for i, time_step in enumerate(line["data"]):
        row = {"time_step": i}
        # Add position data for each planet
        for j, position in enumerate(time_step):
            row[f"planet{j}_x"] = position[0]
            row[f"planet{j}_y"] = position[1]
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    description = line.pop("description")
    step_size = description.pop("stepsize")
    for k, v in description.items():
        for k_prop, v_prop in v.items():
            df[f"{k}_{k_prop}"] = v_prop
    df["time_step"] = df["time_step"] * step_size
    df.insert(0, "idx", idx)

    return df


# %%
files = os.listdir("data")
if "planets.parquet" not in files:
    with open("data/planets.data") as f:
        data = f.read().splitlines()

        dfs = []
        for idx, line in enumerate(tqdm(data)):
            dfs.append(line2df(line, idx))
        df = pd.concat(dfs)
    df.to_parquet("data/planets.parquet")
else:
    df = pd.read_parquet("data/planets.parquet")


# Combine total mass of all planets into one column `planet<n>_m`
mass_regex = re.compile(r"planet(\d+)_m")
mass_cols = [col for col in df.columns if mass_regex.match(col)]
df["total_mass"] = df[mass_cols].sum(axis=1)
# Introduce categorical columns for the number of planets choose non null columns with mass
df["n_planets"] = df[mass_cols].notnull().sum(axis=1).astype("object")
df["n_planets"] = df["n_planets"].apply(lambda x: f"{x}_planets")
# Create category acceleration if the sum of plane/d_[x,y, z] is greater than 0
df["acceleration_x"] = df[
    [col for col in df.columns if "planet" in col and "_x" in col]
].sum(axis=1)
# Set acceleration_x to "increasing" if greater than 0 else "decreasing"
df["acceleration_x"] = (
    df["acceleration_x"]
    .apply(lambda x: "increasing" if x > 0 else "decreasing")
    .astype("object")
)
df["acceleration_y"] = df[
    [col for col in df.columns if "planet" in col and "_y" in col]
].sum(axis=1)
df["acceleration_y"] = df["acceleration_y"].apply(
    lambda x: "increasing" if x > 0 else "decreasing"
)


df.describe()

# %%
df.head()

# %%
df_categorical = df.select_dtypes(include=["object"]).astype(str)
unique_values_per_column = df_categorical.apply(
    pd.Series.unique
).values  # .flatten().tolist()
flattened_unique_values = np.concatenate(unique_values_per_column).tolist()
unique_values = list(set(flattened_unique_values))
unique_values

# %%
df.select_dtypes(include="object").groupby(
    df.select_dtypes(include="object").columns.tolist()
).size().reset_index(name="count")

# %%
df = df.reset_index(drop=True)

# %%
time_series_config = TimeSeriesConfig.generate(df=df)

# %%
time_series_config = TimeSeriesConfig.generate(df=df)
train_idx = int(df.idx.max() * 0.8)
train_df = df.loc[df.idx < train_idx].copy()
test_df = df.loc[df.idx >= train_idx].copy()
# del df
train_ds = TimeSeriesDS(train_df, time_series_config)
test_ds = TimeSeriesDS(test_df, time_series_config)
len(train_ds), len(test_ds)

# %%
train_ds[0]


def make_batch(ds: TimeSeriesDS, start: int, length: int):
    """Create a batch of data from the dataset.

    Args:
        ds (hp.TimeSeriesDS): The dataset to create a batch from.
        start (int): The starting index of the batch.
        length (int): The length of the batch.

    Returns:
        dict: Dictionary containing numeric and categorical data.
    """
    numeric = []
    categorical = []
    for i in range(start, length + start):
        numeric.append(ds[i][0])
        categorical.append(ds[i][1])

    # Convert to numpy arrays first to avoid the warning
    numeric_array = np.array(numeric)
    categorical_array = np.array(categorical)

    return {
        "numeric": torch.tensor(
            numeric_array, dtype=torch.float32
        ),  # Explicitly use float32
        "categorical": torch.tensor(
            categorical_array, dtype=torch.float32
        ),  # Explicitly use float32
    }


# %%
tabular_decoder = TimeSeriesDecoder(time_series_config, d_model=512, n_heads=8)


# After creating the model, add debugging function
def check_model_for_nans(model):
    """Check for NaNs in model parameters"""
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in parameter {name}")
            has_nan = True
    return has_nan


# %%
# Create a sample batch from the training dataset (batch size = 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tabular_decoder = tabular_decoder.to(device)  # Move model to device first

# Create a sample batch from the training dataset
example_batch = make_batch(train_ds, 0, 6)
numeric_data = example_batch["numeric"].to(device)
categorical_data = example_batch["categorical"].to(device)

# Add this just before the prediction
print("Checking model parameters for NaNs:")
has_nan_params = check_model_for_nans(tabular_decoder)
if has_nan_params:
    print("NaNs found in model parameters. Reinitializing...")
    # Reinitialize the model to try to resolve NaN issues
    tabular_decoder = TimeSeriesDecoder(time_series_config, d_model=512, n_heads=8)
    tabular_decoder.to(device)

# Before making a prediction, do a sanity check on inputs
print("Input shapes:", numeric_data.shape, categorical_data.shape)
print("Input contains NaN (numeric):", torch.isnan(numeric_data).any().item())
print("Input contains NaN (categorical):", torch.isnan(categorical_data).any().item())

# Make a prediction using the model
with torch.no_grad():
    # Use try/except to catch and debug any runtime errors
    try:
        prediction = tabular_decoder(numeric_data, categorical_data)

        # Move predictions back to CPU for numpy operations if needed
        prediction = {k: v.cpu() for k, v in prediction.items()}

        prediction["numeric"] = prediction["numeric"].transpose(1, 2)
        prediction["categorical"] = prediction["categorical"].permute(0, 2, 1, 3)

        # Handle error case differently
    except RuntimeError as e:
        print(f"Error during prediction: {e}")
        prediction = {
            "numeric": torch.zeros_like(numeric_data),
            "categorical": torch.zeros_like(categorical_data[:, :, 0]).unsqueeze(-1),
        }

        # Check if prediction contains NaNs
        if any(torch.isnan(tensor).any() for tensor in prediction.values()):
            print("Warning: NaNs in prediction. Applying nan_to_num...")
            for key in prediction:
                prediction[key] = torch.nan_to_num(prediction[key], nan=0.0)

# Print prediction summary instead of all values
print("Prediction numeric shape:", prediction["numeric"].shape)
print("Prediction categorical shape:", prediction["categorical"].shape)
print(
    "Prediction contains NaN (numeric):",
    torch.isnan(prediction["numeric"]).any().item(),
)
print(
    "Prediction contains NaN (categorical):",
    torch.isnan(prediction["categorical"]).any().item(),
)

# %%
# Modify the main execution code to use the proper multiprocessing pattern
ic.disable()


# Add this near the end, replacing your current training setup
def run_training():
    # Set up the training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model with more stable initialization
    tabular_decoder = TimeSeriesDecoder(
        time_series_config,
        d_model=256,  # Reduced from 512 to improve stability
        n_heads=8,
    )

    # Initialize weights with more conservative values
    def init_weights(m):
        if hasattr(m, "weight") and m.weight is not None:
            if len(m.weight.shape) > 1:
                # Use Kaiming initialization for better stability
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                # Scale down initial weights to prevent explosions
                m.weight.data *= 0.05
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # Apply custom weight initialization
    tabular_decoder.apply(init_weights)
    print("Applied conservative weight initialization")

    # Move model to device
    tabular_decoder.to(device)

    # Set up training parameters with much more conservative values
    learning_rate = 1e-5  # Reduced learning rate by 5x
    num_epochs = 10
    gradient_accumulation_steps = 4  # Increased for stability
    max_grad_norm = 0.1  # Much tighter gradient clipping

    # Add gradient explosion detection threshold
    max_gradient_norm_allowed = 10.0
    max_explosion_count = 5  # Allow this many explosions before reducing LR permanently

    timestamp = dt.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = f"runs/{timestamp}_WTF_IS_GOING_ON"
    save_dir = "models/planets"

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("To view logs, run: tensorboard --logdir=runs")

    # Train the model with enhanced stability parameters
    history = train_model(
        model=tabular_decoder,
        train_dataset=train_ds,
        val_dataset=test_ds,
        batch_size=4,  # Reduced batch size for stability
        epochs=num_epochs,
        learning_rate=learning_rate,
        log_dir=log_dir,
        save_dir=save_dir,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        explosion_threshold=max_gradient_norm_allowed,
        max_explosions_per_epoch=max_explosion_count,
    )

    # Visualize training history
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_numeric_loss"], label="Train Numeric")
    plt.plot(history["train_categorical_loss"], label="Train Categorical")
    plt.plot(history["val_numeric_loss"], label="Val Numeric")
    plt.plot(history["val_categorical_loss"], label="Val Categorical")
    plt.title("Component Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # When evaluating the best model, create the test_loader here
    best_model_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        tabular_decoder.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}"
        )

        # Create test loader for evaluation
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Evaluate on test set
        tabular_decoder.eval()
        test_losses = {"loss": 0.0, "numeric_loss": 0.0, "categorical_loss": 0.0}

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating test set"):
                # Move batch to device
                for key in batch:
                    batch[key] = batch[key].to(device)

                # Get predictions
                outputs = tabular_decoder(
                    numeric_inputs=batch["numeric"],
                    categorical_inputs=batch["categorical"],
                )

                # Calculate losses
                from hephaestus.training.training import categorical_loss, numeric_loss

                numeric_loss_val = numeric_loss(batch["numeric"], outputs["numeric"])
                categorical_loss_val = categorical_loss(
                    batch["categorical"], outputs["categorical"]
                )
                total_loss = numeric_loss_val + categorical_loss_val

                test_losses["loss"] += total_loss.item()
                test_losses["numeric_loss"] += numeric_loss_val.item()
                test_losses["categorical_loss"] += categorical_loss_val.item()

        # Average losses - this now works because test_loader is defined
        for key in test_losses:
            test_losses[key] /= len(test_loader)

        print(f"Test Loss: {test_losses['loss']:.4f}")
        print(f"Test Numeric Loss: {test_losses['numeric_loss']:.4f}")
        print(f"Test Categorical Loss: {test_losses['categorical_loss']:.4f}")
    return history


# Add this at the very end of the file
if __name__ == "__main__":
    # This is critical for multiprocessing with PyTorch DataLoader
    history = run_training()

# %%
