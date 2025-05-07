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
import py7zr

from icecream import ic

import hephaestus as hp

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


def extract_7z_files(archive_dir, target_dir):
    """
    Extracts 7z archive files into the target directory.

    Args:
        archive_dir (Path): Directory containing 7z archive files
        target_dir (Path): Directory where files should be extracted

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Find all 7z archive parts
        archive_parts = sorted(archive_dir.glob("*.7z.*"))

        if not archive_parts:
            print(f"No 7z archive parts found in {archive_dir}")
            return False

        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Check if data is already extracted
        if any(target_dir.iterdir()):
            print(f"Data appears to be already extracted in {target_dir}")
            return True

        print(f"Extracting 7z archives from {archive_dir} to {target_dir}...")

        # With py7zr, we only need to open the first part
        first_part = str(archive_parts[0])
        with py7zr.SevenZipFile(first_part, "r") as archive:
            archive.extractall(path=target_dir)

        print("Extraction completed successfully")
        return True

    except Exception as e:
        print(f"Error extracting 7z archives: {e}")
        return False


data_path = Path("./data/3w_dataset")  # Base path to the data
extracted_data_path = data_path / "extracted"  # Path for extracted data

# Extract 7z archives if needed
if not extract_7z_files(data_path, extracted_data_path):
    print("Failed to extract 7z archives. Please check the archive files.")
else:
    # Use the extracted data path
    raw_df = create_raw_dataframe(extracted_data_path)
    if raw_df is not None:
        print(raw_df.head())
        print(raw_df.info())

    df = raw_df.copy() if raw_df is not None else None

# %%
time_series_config = hp.TimeSeriesConfig.generate(df=df)
# %%
train_idx = int(df.idx.max() * 0.8)
train_df = df.loc[df.idx < train_idx].copy()
test_df = df.loc[df.idx >= train_idx].copy()
# %%
train_ds = hp.TimeSeriesDS(train_df, time_series_config)
test_ds = hp.TimeSeriesDS(test_df, time_series_config)
len(train_ds), len(test_ds)
# %%
train_ds[0]
# %%
