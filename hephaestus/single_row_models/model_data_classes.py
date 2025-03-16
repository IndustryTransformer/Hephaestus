from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from hephaestus.utils import (
    NumericCategoricalData,
    convert_object_to_int_tokens,
    split_complex_word,
)


@dataclass
class InputsTarget:
    """
    Dataclass for inputs and outputs.

    Attributes:
        inputs (NumericCategoricalData): The inputs.
        target (torch.Tensor): The Target.
    """

    inputs: NumericCategoricalData
    target: torch.Tensor


@dataclass
class SingleRowConfig:
    """
    Configuration class for single row encoder.

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
    n_columns: int = None
    n_columns_no_target: int = None
    n_numeric_cols: int = None
    n_cat_cols: int = None
    target: str = None

    @classmethod
    def generate(cls, df: pd.DataFrame, target: str = "target") -> "SingleRowConfig":
        """
        Generate a TimeSeriesConfig object based on the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            SingleRowConfig: The generated TimeSeriesConfig object.
        """

        cls_dict = {}

        df_categorical = df.select_dtypes(include=["object"]).astype(str)
        numeric_token = "[NUMERIC_EMBEDDING]"
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
            col_name
            for col_name in df.select_dtypes(include="number").columns
            if col_name != target  # Remove target column
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

        cls_dict["n_columns"] = len(df.columns)
        cls_dict["n_columns_no_target"] = len(df.columns) - 1
        cls_dict["vocab_size"] = tokenizer.vocab_size
        cls_dict["target"] = target

        cls_dict["n_numeric_cols"] = len(numeric_col_tokens)
        cls_dict["n_cat_cols"] = len(categorical_col_tokens) - 1
        df_categorical = convert_object_to_int_tokens(df_categorical, token_dict)

        return cls(**cls_dict)


# %%
class TabularDS(Dataset):
    """
    Dataset class for time series data.

    Args:
        df (pd.DataFrame): The DataFrame containing the time series data.
        config (TimeSeriesConfig): Configuration for the time series.

    Attributes:
        df_categorical (pd.DataFrame): DataFrame of categorical columns.
        df_numeric (pd.DataFrame): DataFrame of numeric columns.
        batch_size (int): Batch size for the dataset.
    """

    def __init__(self, df: pd.DataFrame, config: SingleRowConfig):
        # Set df.idx to start from 0

        def convert_object_to_int_tokens(df, token_dict):
            """Converts object columns to integer tokens using a token dictionary."""
            df = df.copy()
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].map(token_dict)
            return df

        df = df.copy()  # Prevents changing the original DataFrame
        self.len = len(df)
        self.target_df = pd.DataFrame(df.pop(config.target))
        self.df_categorical = df.select_dtypes(include=["object"]).astype(str)
        self.df_categorical = convert_object_to_int_tokens(
            self.df_categorical, config.token_dict
        )
        self.df_numeric = df.select_dtypes(include=["number"])

    def __len__(self):
        """Return the length of the dataset."""
        return self.len

    def get_data(self, df_name, set_idx):
        """Gets self.df_<df_name> for a given index."""
        df = getattr(self, df_name)

        batch = df.iloc[set_idx, :]
        # Convert DataFrame to torch.Tensor directly
        batch = torch.tensor(batch.values, dtype=torch.float32)

        # Add padding
        # batch_len, n_cols = batch.shape
        batch = batch.squeeze(0)
        # TODO Add padding for columns
        # batch = batch.permute(1, 0)
        return batch

    def __getitem__(self, idx):
        """
        Get item(s) from the dataset.
        Supports both integer indexing and slice indexing.

        Args:
            idx: Integer index or slice object

        Returns:
            For integer index: A NumericCategoricalData object
            For slice: A collated batch of NumericCategoricalData formatted like DataLoader output
        """
        # Handle slice indexing
        if isinstance(idx, slice):
            # Get the actual indices from the slice
            indices = range(*idx.indices(len(self)))

            # Create a list of NumericCategoricalData
            items = [self._get_single_item(i) for i in indices]

            # Collate the items into a batch
            return self._collate_batch(items)

        # Handle integer indexing
        return self._get_single_item(idx)

    def _get_single_item(self, set_idx):
        """Get a single item from the dataset by actual index value."""
        if self.df_categorical.empty:
            categorical_inputs = None
        else:
            categorical_inputs = self.get_data("df_categorical", set_idx)
        numeric_inputs = self.get_data("df_numeric", set_idx)
        if self.target_df is not None:
            target = self.get_data("target_df", set_idx)
        else:
            target = None
        numeric_cat = NumericCategoricalData(
            numeric=numeric_inputs, categorical=categorical_inputs
        )
        return InputsTarget(inputs=numeric_cat, target=target)

    def _collate_batch(self, items):
        """
        Collate a list of NumericCategoricalData into a batch.

        Args:
            items: List of NumericCategoricalData objects

        Returns:
            A NumericCategoricalData object with batched tensors
        """
        # All items should have numeric inputs
        # Use torch.stack instead of np.stack
        numeric_batch = torch.stack([item.inputs.numeric for item in items])

        # Not all items might have categorical inputs
        categorical_batch = None
        if items[0].inputs.categorical is not None:
            categorical_batch = torch.stack([item.inputs.categorical for item in items])
        if items[0].target is not None:
            target_batch = torch.stack([item.target for item in items])
        else:
            target_batch = None
        obj = InputsTarget(
            inputs=NumericCategoricalData(
                numeric=numeric_batch, categorical=categorical_batch
            ),
            target=target_batch,
        )
        return obj
