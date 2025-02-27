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

# from hephaestus.analysis.analysis import (
#     AutoRegressiveResults,
#     auto_regressive_predictions,
#     create_test_inputs_df,
#     plot_column_variants,
# )
from hephaestus.models import (
    TimeSeriesConfig,
    TimeSeriesDecoder,
    TimeSeriesDS,
    tabular_collate_fn,
)
from hephaestus.training.training_loop import train_model  # Fixed import path

# %%
torch.set_default_dtype(torch.float32)
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
df = df.head(5_000)
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


# %%
N_HEADS = 8 * 4
tabular_decoder = TimeSeriesDecoder(time_series_config, d_model=512, n_heads=N_HEADS)


# %%
# Create a sample batch from the training dataset (batch size = 1)
# Check for available devices including MPS (Metal Performance Shaders) for Macs with Apple Silicon
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Set to 0 to avoid multiprocessing issues
    collate_fn=tabular_collate_fn,
    pin_memory=True if torch.cuda.is_available() else False,
)
tabular_decoder = tabular_decoder.to(device)  # Move model to device first

# Create a sample batch from the training dataset
example_batch = train_ds[0:6]
numeric_data = example_batch.numeric.to(device)
categorical_data = example_batch.categorical.to(device)
# %%

# Before making a prediction, do a sanity check on inputs
print("Input shapes:", numeric_data.shape, categorical_data.shape)
print("Input contains NaN (numeric):", torch.isnan(numeric_data).any().item())
print("Input contains NaN (categorical):", torch.isnan(categorical_data).any().item())

# Make a prediction using the model
with torch.no_grad():
    prediction = tabular_decoder(numeric_data, categorical_data)

    # Move predictions back to CPU for numpy operations if needed
    prediction = prediction.to("cpu")

    prediction.numeric = prediction.numeric.transpose(1, 2)
    prediction.categorical = prediction.categorical.permute(0, 2, 1, 3)


# Print prediction summary instead of all values
print("Prediction numeric shape:", prediction.numeric.shape)
print("Prediction categorical shape:", prediction.categorical.shape)
print(
    "Prediction contains NaN (numeric):",
    torch.isnan(prediction.numeric).any().item(),
)
print(
    "Prediction contains NaN (categorical):",
    torch.isnan(prediction.categorical).any().item(),
)

# %%
# Modify the main execution code to use the proper multiprocessing pattern
ic.disable()


# Add this near the end, replacing your current training setup
def run_training():
    # Set up the training loop
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
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
    log_dir = f"runs/{timestamp}_Planets_Does_This_Work"
    save_dir = "images/planets"

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
        batch_size=batch_size,  # Reduced batch size for stability
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

    # Import the plotting functions from our new module
    from hephaestus.analysis.plots import (
        evaluate_planetary_predictions,
        plot_training_history,
    )

    # Visualize training history
    plot_training_history(history, save_dir)

    # Load best model for evaluation
    best_model_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        tabular_decoder.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}"
        )

        # Evaluate the model on the test data using our new function
        try:
            results, pred_df, actual_df = evaluate_planetary_predictions(
                model=tabular_decoder,
                test_ds=test_ds,
                time_series_config=time_series_config,
                test_sample_idx=0,
                n_steps=20,
                save_dir=save_dir,
                device=device,
            )

            # Print summary of evaluation results with better NaN handling
            if results and "mae_results" in results:
                # Filter out NaN values before calculating average
                valid_maes = [
                    v for v in results["mae_results"].values() if np.isfinite(v)
                ]

                if valid_maes:
                    avg_mae = sum(valid_maes) / len(valid_maes)
                    print(
                        "\nEvaluation complete. Average MAE across all positions:",
                        f"{avg_mae:.6f}",
                    )
                    print(
                        f"Valid MAE values: {len(valid_maes)}/{len(results['mae_results'])}"
                    )
                else:
                    print("\nEvaluation complete, but no valid MAE values were found.")
                    print(
                        "Check for numerical stability issues in your model predictions."
                    )

        except Exception as e:
            print(f"Error during evaluation: {e}")

    return history


# Add this at the very end of the file
if __name__ == "__main__":
    # This is critical for multiprocessing with PyTorch DataLoader
    history = run_training()

# %%
