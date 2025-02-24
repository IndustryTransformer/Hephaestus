import subprocess

import jax.numpy as jnp

import hephaestus as hp


def make_batch(ds: hp.TimeSeriesDS, start: int, length: int):
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
    return {"numeric": jnp.array(numeric), "categorical": jnp.array(categorical)}


def get_git_commit_hash():
    """Get the current git commit hash.

    Returns:
        str: The current git commit hash, or 'unknown' if an error occurs.
    """
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode()
        )
        return commit_hash
    except Exception:
        return "unknown"
