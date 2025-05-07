# %% [markdown]
# # 3W Anomaly Detection
# [Data](https://github.com/ricardovvargas/3w_dataset) is a time series dataset for anomaly detection.
#
# ## Load Libs
#
# %%
import os

import icecream
import pandas as pd
import torch
from pathlib import Path

from icecream import ic


torch.set_float32_matmul_precision("medium")
# %%

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
icecream.install()
ic_disable = True  # Global variable to disable ic
if ic_disable:
    ic.disable()
ic.configureOutput(includeContext=True, contextAbsPath=True)
# pd.options.mode.copy_on_write = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# Define the columns

# Define the columns
columns = (
    ["timestamp"]
    + [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    + ["class"]
)


def class_and_file_generator(data_path, real=False, simulated=False, drawn=False):
    """
    Generates class codes and file paths based on directory structure and file names.

    Args:
        data_path (Path): The root directory containing the data.
        real (bool, optional): If True, only yields real data files. Defaults to False.
        simulated (bool, optional): If True, only yields simulated data files. Defaults to False.
        drawn (bool, optional): If True, only yields drawn data files. Defaults to False.

    Yields:
        tuple: (class_code, instance_path) for each qualifying file.
    """
    for class_path in data_path.iterdir():
        if class_path.is_dir():
            class_code = int(class_path.stem)
            for instance_path in class_path.iterdir():
                if instance_path.suffix == ".csv":
                    if (
                        (simulated and instance_path.stem.startswith("SIMULATED"))
                        or (drawn and instance_path.stem.startswith("DRAWN"))
                        or (
                            real
                            and (not instance_path.stem.startswith("SIMULATED"))
                            and (not instance_path.stem.startswith("DRAWN"))
                        )
                    ):
                        yield class_code, instance_path


def load_instance(instance_path):
    """
    Loads a single CSV file into a Pandas DataFrame.

    Args:
        instance_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.  Raises
        an exception if the columns in the CSV file do not match the expected
        columns.
    """
    try:
        df = pd.read_csv(instance_path, sep=",", header=0)
        # crucial check of the columns.
        assert (
            df.columns == columns
        ).all(), f"Invalid columns in the file {instance_path}: {df.columns.tolist()}"
        return df
    except Exception as e:
        raise Exception(f"Error reading file {instance_path}: {e}")


def create_raw_dataframe(data_path):
    """
    Loads all CSV files from the specified directory and its subdirectories
    into a single Pandas DataFrame.  It only loads "real" data (i.e., not
    simulated or drawn).  It converts the 'timestamp' column to datetime.

    Args:
        data_path (Path): Path to the root directory containing the data.

    Returns:
        pd.DataFrame: A DataFrame containing all the data, or None if no data
                      is found.
    """
    data = []
    for class_code, instance_path in class_and_file_generator(data_path, real=True):
        try:
            df_instance = load_instance(instance_path)
            data.append(df_instance)  # Append the DataFrame, not the path
        except Exception as e:
            print(f"Error processing {instance_path}: {e}")  # keep processing.

    if not data:
        print("No data found.")
        return None

    # Concatenate all the dataframes
    df = pd.concat(data, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# Example usage:
data_path = Path("./data/3w_dataset/data")  # Point to the extracted data directory
raw_df = create_raw_dataframe(data_path)
if raw_df is not None:
    print(raw_df.head())
    print(raw_df.info())
# %%
