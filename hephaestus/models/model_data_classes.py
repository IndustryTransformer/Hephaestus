from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Import tokenizer functions directly from module to avoid circular imports
from hephaestus.utils.tokenizer import convert_object_to_int_tokens, split_complex_word


@dataclass
class TimeSeriesConfig:
    """
    Configuration class for time series decoder.

    Attributes:
        numeric_token (str): Token for numeric embedding.
        numeric_mask (str): Token for numeric mask.
        numeric_col_tokens (list): List of tokens for numeric columns.
        categorical_col_tokens (list): List of tokens for categorical columns.
        tokens (list): List of all tokens.
        token_dict (dict): Dictionary mapping tokens to indices.
        token_decoder_dict (dict): Dictionary mapping indices to tokens.
        n_tokens (int): Number of tokens.
        numeric_indices (torch.Tensor): Tensor of indices for numeric columns.
        categorical_indices (torch.Tensor): Tensor of indices for categorical columns.
        object_tokens (list): List of unique values in categorical columns.
        numeric_mask_token (int): Index of numeric mask token.
        reservoir_vocab (list): List of words in custom vocabulary.
        reservoir_encoded (torch.Tensor): Encoded reservoir tokens.
        tokenizer (AutoTokenizer): Tokenizer for encoding tokens.
    """

    numeric_token: str = None
    numeric_mask: str = None
    numeric_col_tokens: list = None
    categorical_col_tokens: list = None
    tokens: list = None
    token_dict: dict = None
    token_decoder_dict: dict = None
    n_tokens: int = None
    numeric_indices: torch.Tensor = None
    categorical_indices: torch.Tensor = None
    object_tokens: list = None
    numeric_mask_token: int = None
    reservoir_vocab: list = None
    reservoir_encoded: torch.Tensor = None
    tokenizer: AutoTokenizer = None
    vocab_size: int = None
    ds_length: int = None
    n_columns: int = None

    @classmethod
    def generate(cls, df: pd.DataFrame) -> "TimeSeriesConfig":
        """
        Generate a TimeSeriesConfig object based on the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            TimeSeriesConfig: The generated TimeSeriesConfig object.
        """

        # Set df.idx to start from 0
        df.idx = df.idx - df.idx.min()
        ds_length = df.groupby("idx").size().max()
        df = df.set_index("idx")
        df.index.name = None

        df_categorical = df.select_dtypes(include=["object"]).astype(str)
        numeric_token = "[NUMERIC_EMBEDDING]"
        cls_dict = {}
        cls_dict["numeric_token"] = numeric_token
        special_tokens = [
            "[PAD]",
            "[NUMERIC_MASK]",
            "[MASK]",
            "[UNK]",
            numeric_token,
        ]
        cls_dict["numeric_mask"] = "[NUMERIC_MASK]"
        numeric_mask = cls_dict["numeric_mask"]
        # Remove check on idx
        cls_dict["numeric_col_tokens"] = [
            col_name for col_name in df.select_dtypes(include="number").columns
        ]
        cls_dict["categorical_col_tokens"] = [
            col_name for col_name in df.select_dtypes(include="object").columns
        ]
        # Get all the unique values in the categorical columns and add them to the tokens
        unique_values_per_column = df_categorical.apply(pd.Series.unique).values
        flattened_unique_values = np.concatenate(unique_values_per_column).tolist()
        object_tokens = list(set(flattened_unique_values))
        cls_dict["object_tokens"] = object_tokens

        cls_dict["tokens"] = (
            special_tokens
            + cls_dict["numeric_col_tokens"]
            + cls_dict["object_tokens"]
            + cls_dict["categorical_col_tokens"]
        )
        tokens = cls_dict["tokens"]
        numeric_col_tokens = cls_dict["numeric_col_tokens"]
        categorical_col_tokens = cls_dict["categorical_col_tokens"]
        token_dict = {token: i for i, token in enumerate(tokens)}
        cls_dict["token_dict"] = token_dict
        token_decoder_dict = {i: token for i, token in enumerate(tokens)}
        cls_dict["token_decoder_dict"] = token_decoder_dict
        n_tokens = len(cls_dict["tokens"])
        cls_dict["n_tokens"] = n_tokens

        # Convert to PyTorch tensors
        numeric_indices = torch.tensor([tokens.index(i) for i in numeric_col_tokens])
        cls_dict["numeric_indices"] = numeric_indices
        categorical_indices = torch.tensor(
            [tokens.index(i) for i in categorical_col_tokens]
        )
        cls_dict["categorical_indices"] = categorical_indices

        numeric_mask_token = tokens.index(numeric_mask)
        cls_dict["numeric_mask_token"] = numeric_mask_token
        # Make custom vocab by splitting on snake case, camel case, spaces and numbers
        reservoir_vocab = [split_complex_word(word) for word in token_dict.keys()]
        # flatten the list, make a set and then list again
        reservoir_vocab = list(
            set([item for sublist in reservoir_vocab for item in sublist])
        )
        # Get reservoir embedding tokens
        reservoir_tokens_list = [
            token_decoder_dict[i] for i in range(len(token_decoder_dict))
        ]  # ensures they are in the same order
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls_dict["tokenizer"] = tokenizer

        # Convert to PyTorch tensor
        encoded = tokenizer(
            reservoir_tokens_list,
            padding="max_length",
            max_length=8,  # TODO Make this dynamic
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        cls_dict["reservoir_encoded"] = encoded

        cls_dict["reservoir_vocab"] = reservoir_vocab
        cls_dict["ds_length"] = ds_length

        cls_dict["n_columns"] = len(df.columns)
        cls_dict["vocab_size"] = tokenizer.vocab_size
        df_categorical = convert_object_to_int_tokens(df_categorical, token_dict)

        return cls(**cls_dict)


@dataclass
class TimeSeriesInputs:
    """
    Class to hold time series inputs.
    """
    numeric: torch.Tensor
    categorical: Optional[torch.Tensor] = None
    
    def to(self, device):
        """Move tensors to the specified device"""
        self.numeric = self.numeric.to(device)
        if self.categorical is not None:
            self.categorical = self.categorical.to(device)
        return self


@dataclass
class TimeSeriesOutput:
    """
    Data class to store time series model outputs.

    Attributes:
        numeric: Numeric outputs from the model.
        categorical: Categorical outputs from the model, may be None if no categorical data exists.
    """

    numeric: torch.Tensor
    categorical: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        """
        Move tensors to the specified device.

        Args:
            device: The device to move tensors to (e.g., 'cuda', 'cpu', or torch.device)

        Returns:
            TimeSeriesOutput: Self with tensors moved to the specified device
        """
        self.numeric = self.numeric.to(device)
        if self.categorical is not None:
            self.categorical = self.categorical.to(device)
        return self

    def items(self):
        """
        Return a dictionary-like items view for backward compatibility.

        Returns:
            An items iterator similar to dict.items()
        """
        result = {"numeric": self.numeric}
        if self.categorical is not None:
            result["categorical"] = self.categorical
        return result.items()

    def __getitem__(self, key):
        """
        Support dictionary-style access for backward compatibility.

        Args:
            key: The key to access ('numeric' or 'categorical')

        Returns:
            The corresponding tensor

        Raises:
            KeyError: If key is not 'numeric' or 'categorical'
        """
        if key == "numeric":
            return self.numeric
        elif key == "categorical":
            return self.categorical
        else:
            raise KeyError(f"Invalid key: {key}")

    def values(self):
        """
        Return values for backward compatibility with dictionary.

        Returns:
            List of tensor values
        """
        result = [self.numeric]
        if self.categorical is not None:
            result.append(self.categorical)
        return result


class TimeSeriesDS(Dataset):
    """
    Dataset class for time series data.

    Args:
        df (pd.DataFrame): The DataFrame containing the time series data.
        config (TimeSeriesConfig): Configuration for the time series.

    Attributes:
        max_seq_len (int): Maximum sequence length.
        df_categorical (pd.DataFrame): DataFrame of categorical columns.
        df_numeric (pd.DataFrame): DataFrame of numeric columns.
        batch_size (int): Batch size for the dataset.
    """

    def __init__(self, df: pd.DataFrame, config: TimeSeriesConfig):
        self.max_seq_len = df.groupby("idx").size().max()
        # Set df.idx to start from 0
        df.idx = df.idx - df.idx.min()
        df = df.set_index("idx")
        df.index.name = None

        def convert_object_to_int_tokens(df, token_dict):
            """Converts object columns to integer tokens using a token dictionary."""
            df = df.copy()
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].map(token_dict)
            return df

        self.df_categorical = df.select_dtypes(include=["object"]).astype(str)
        self.df_categorical = convert_object_to_int_tokens(
            self.df_categorical, config.token_dict
        )
        self.df_numeric = df.select_dtypes(include="number")
        self.batch_size = self.max_seq_len
        self.unique_indices = sorted(df.index.unique())

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.unique_indices)

    def get_data(self, df_name, set_idx):
        """Gets self.df_<df_name> for a given index."""
        df = getattr(self, df_name)

        batch = df.loc[df.index == set_idx, :]
        # Convert DataFrame to torch.Tensor directly
        batch = torch.tensor(batch.values, dtype=torch.float32)

        # Add padding
        batch_len, n_cols = batch.shape
        pad_len = self.max_seq_len - batch_len
        # Use torch.full instead of np.full
        padding = torch.full((pad_len, n_cols), float("nan"), dtype=torch.float32)
        # Use torch.cat instead of np.concatenate
        batch = torch.cat([batch, padding], dim=0)
        # Use torch.permute instead of np.swapaxes
        batch = batch.permute(1, 0)
        return batch

    def __getitem__(self, idx):
        """
        Get item(s) from the dataset.
        Supports both integer indexing and slice indexing.

        Args:
            idx: Integer index or slice object

        Returns:
            For integer index: A TimeSeriesInputs object
            For slice: A collated batch of TimeSeriesInputs formatted like DataLoader output
        """
        # Handle slice indexing
        if isinstance(idx, slice):
            # Get the actual indices from the slice
            indices = self.unique_indices[idx]

            # Create a list of TimeSeriesInputs
            items = [self._get_single_item(i) for i in indices]

            # Collate the items into a batch
            return self._collate_batch(items)

        # Handle integer indexing
        return self._get_single_item(self.unique_indices[idx])

    def _get_single_item(self, set_idx):
        """Get a single item from the dataset by actual index value."""
        if self.df_categorical.empty:
            categorical_inputs = None
        else:
            categorical_inputs = self.get_data("df_categorical", set_idx)
        numeric_inputs = self.get_data("df_numeric", set_idx)

        return TimeSeriesInputs(numeric=numeric_inputs, categorical=categorical_inputs)

    def _collate_batch(self, items):
        """
        Collate a list of TimeSeriesInputs into a batch.

        Args:
            items: List of TimeSeriesInputs objects

        Returns:
            A TimeSeriesInputs object with batched tensors
        """
        # All items should have numeric inputs
        # Use torch.stack instead of np.stack
        numeric_batch = torch.stack([item.numeric for item in items])

        # Not all items might have categorical inputs
        categorical_batch = None
        if items[0].categorical is not None:
            categorical_batch = torch.stack([item.categorical for item in items])

        return TimeSeriesInputs(numeric=numeric_batch, categorical=categorical_batch)

    def get_batch(self, batch_size=None, start_idx=0):
        """
        Get a batch of specified size starting from a specific index.

        Args:
            batch_size: Size of the batch to retrieve
            start_idx: Starting index position

        Returns:
            TimeSeriesInputs object with batched data
        """
        if batch_size is None:
            batch_size = self.batch_size

        end_idx = min(start_idx + batch_size, len(self))
        return self[start_idx:end_idx]


@dataclass
class ProcessedEmbeddings:
    """
    Data class to store processed embeddings.
    """

    column_embeddings: Optional[torch.Tensor] = None
    value_embeddings: Optional[torch.Tensor] = None


def tabular_collate_fn(batch: List[TimeSeriesInputs]) -> TimeSeriesInputs:
    """
    Custom collate function for TimeSeriesInputs.
    
    Args:
        batch: A list of TimeSeriesInputs objects
        
    Returns:
        A batched TimeSeriesInputs object
    """
    # Collect numeric and categorical tensors separately
    numeric_tensors = [item.numeric for item in batch]
    
    # Stack numeric tensors along a new batch dimension
    numeric_batch = torch.stack(numeric_tensors, dim=0)
    
    # Handle categorical data if present
    if batch[0].categorical is not None:
        categorical_tensors = [item.categorical for item in batch]
        categorical_batch = torch.stack(categorical_tensors, dim=0)
    else:
        categorical_batch = None
    
    # Create and return a new TimeSeriesInputs with batched data
    return TimeSeriesInputs(numeric=numeric_batch, categorical=categorical_batch)
