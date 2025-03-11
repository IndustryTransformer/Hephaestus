from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from hephaestus.utils.tokenizer import convert_object_to_int_tokens, split_complex_word


@dataclass
class TabularConfig:
    """
    Configuration class for tabular encoder with masking.

    Similar to TimeSeriesConfig but adapted for non-time series tabular data.

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
        vocab_size (int): Size of the vocabulary.
        n_columns (int): Number of columns in the dataset.
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

    @classmethod
    def generate(cls, df: pd.DataFrame) -> "TabularConfig":
        """
        Generate a TabularConfig object based on the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            TabularConfig: The generated TabularConfig object.
        """
        cls_dict = {}
        df_categorical = df.select_dtypes(include=["object"]).astype(str)

        # Define tokens
        numeric_token = "[NUMERIC_EMBEDDING]"
        cls_dict["numeric_token"] = numeric_token
        special_tokens = [
            "[PAD]",
            "[NUMERIC_MASK]",
            "[CATEGORICAL_MASK]",
            "[MASK]",
            "[UNK]",
            numeric_token,
        ]
        cls_dict["numeric_mask"] = "[NUMERIC_MASK]"

        # Get column tokens
        cls_dict["numeric_col_tokens"] = [
            col_name for col_name in df.select_dtypes(include="number").columns
        ]
        cls_dict["categorical_col_tokens"] = [
            col_name for col_name in df.select_dtypes(include="object").columns
        ]

        # Get unique values from categorical columns
        unique_values_per_column = df_categorical.apply(pd.Series.unique).values
        flattened_unique_values = (
            np.concatenate(unique_values_per_column).tolist()
            if len(unique_values_per_column) > 0
            else []
        )
        object_tokens = list(set(flattened_unique_values))
        cls_dict["object_tokens"] = object_tokens

        # Combine all tokens
        cls_dict["tokens"] = (
            special_tokens
            + cls_dict["numeric_col_tokens"]
            + cls_dict["object_tokens"]
            + cls_dict["categorical_col_tokens"]
        )

        # Create token dictionaries
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
        numeric_indices = (
            torch.tensor([tokens.index(i) for i in numeric_col_tokens])
            if numeric_col_tokens
            else torch.tensor([])
        )
        cls_dict["numeric_indices"] = numeric_indices
        categorical_indices = (
            torch.tensor([tokens.index(i) for i in categorical_col_tokens])
            if categorical_col_tokens
            else torch.tensor([])
        )
        cls_dict["categorical_indices"] = categorical_indices

        numeric_mask_token = tokens.index(cls_dict["numeric_mask"])
        cls_dict["numeric_mask_token"] = numeric_mask_token

        # Create reservoir vocabulary
        reservoir_vocab = [split_complex_word(word) for word in token_dict.keys()]
        reservoir_vocab = list(
            set([item for sublist in reservoir_vocab for item in sublist])
        )
        cls_dict["reservoir_vocab"] = reservoir_vocab

        # Setup tokenizer
        reservoir_tokens_list = [
            token_decoder_dict[i] for i in range(len(token_decoder_dict))
        ]
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls_dict["tokenizer"] = tokenizer

        # Encode the tokens
        encoded = tokenizer(
            reservoir_tokens_list,
            padding="max_length",
            max_length=8,  # Set a fixed length for token encoding
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        cls_dict["reservoir_encoded"] = encoded

        cls_dict["vocab_size"] = tokenizer.vocab_size
        cls_dict["n_columns"] = len(df.columns)

        return cls(**cls_dict)


@dataclass
class TabularInputs:
    """
    Data class to store tabular batch data.

    Attributes:
        numeric: Numeric inputs for the model.
        categorical: Categorical inputs for the model, may be None if no categorical data exists.
    """

    numeric: torch.Tensor
    categorical: Optional[torch.Tensor] = None
    numeric_mask: Optional[torch.Tensor] = None
    categorical_mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        """
        Move tensors to the specified device.

        Args:
            device: The device to move tensors to (e.g., 'cuda', 'cpu', or torch.device)

        Returns:
            TabularInputs: Self with tensors moved to the specified device
        """
        self.numeric = self.numeric.to(device)
        if self.categorical is not None:
            self.categorical = self.categorical.to(device)
        if self.numeric_mask is not None:
            self.numeric_mask = self.numeric_mask.to(device)
        if self.categorical_mask is not None:
            self.categorical_mask = self.categorical_mask.to(device)
        return self


@dataclass
class TabularOutput:
    """
    Data class to store tabular model outputs.

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
            TabularOutput: Self with tensors moved to the specified device
        """
        self.numeric = self.numeric.to(device)
        if self.categorical is not None:
            self.categorical = self.categorical.to(device)
        return self


@dataclass
class ProcessedEmbeddings:
    """
    Data class to store processed embeddings.
    Similar to TimeSeriesOutput but for tabular data.
    """

    column_embeddings: Optional[torch.Tensor] = None
    value_embeddings: Optional[torch.Tensor] = None


class TabularDS(Dataset):
    """
    Dataset class for tabular data with masking capabilities.

    Args:
        df (pd.DataFrame): The DataFrame containing the tabular data.
        config (TabularConfig): Configuration for the tabular data.
        mask_prob (float): Probability of masking a value (default: 0.15).

    Attributes:
        df_categorical (pd.DataFrame): DataFrame of categorical columns.
        df_numeric (pd.DataFrame): DataFrame of numeric columns.
    """

    def __init__(
        self, df: pd.DataFrame, config: TabularConfig, mask_prob: float = 0.15
    ):
        self.df = df.copy()
        self.config = config
        self.mask_prob = mask_prob

        # Process categorical data
        self.has_categorical = len(df.select_dtypes(include=["object"]).columns) > 0
        if self.has_categorical:
            self.df_categorical = df.select_dtypes(include=["object"]).astype(str)
            self.df_categorical = convert_object_to_int_tokens(
                self.df_categorical, config.token_dict
            )
        else:
            self.df_categorical = pd.DataFrame()

        # Process numeric data
        self.df_numeric = df.select_dtypes(include="number")

        # Get numeric mask token
        self.numeric_mask_token = config.numeric_mask_token
        if "categorical_mask_token" in config.token_dict:
            self.categorical_mask_token = config.token_dict["[CATEGORICAL_MASK]"]
        else:
            self.categorical_mask_token = config.token_dict[
                "[NUMERIC_MASK]"
            ]  # Fallback

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single item from the dataset with optional masking."""
        # Get numeric data
        numeric_data = self.df_numeric.iloc[idx].values
        numeric_data = torch.tensor(numeric_data, dtype=torch.float32)

        # Create numeric mask
        numeric_mask = torch.zeros_like(numeric_data, dtype=torch.bool)
        if self.mask_prob > 0:
            # Random masking
            mask_indices = torch.rand(numeric_data.shape) < self.mask_prob
            numeric_mask = mask_indices

        # Store original numeric values before masking
        numeric_target = numeric_data.clone()

        # Apply masking to numeric data (replace with zeros, the model will handle this specially)
        numeric_data = torch.where(
            numeric_mask, torch.tensor(float("nan"), dtype=torch.float32), numeric_data
        )

        # Get and process categorical data if available
        if not self.df_categorical.empty:
            categorical_data = self.df_categorical.iloc[idx].values
            categorical_data = torch.tensor(categorical_data, dtype=torch.float32)

            # Create categorical mask
            categorical_mask = torch.zeros_like(categorical_data, dtype=torch.bool)
            if self.mask_prob > 0:
                # Random masking for categorical data
                cat_mask_indices = torch.rand(categorical_data.shape) < self.mask_prob
                categorical_mask = cat_mask_indices

            # Store original categorical values before masking
            categorical_target = categorical_data.clone()

            # Apply masking to categorical data
            categorical_data = torch.where(
                categorical_mask,
                torch.tensor(self.categorical_mask_token, dtype=torch.float32),
                categorical_data,
            )
        else:
            categorical_data = None
            categorical_mask = None
            categorical_target = None

        return TabularInputs(
            numeric=numeric_data,
            categorical=categorical_data,
            numeric_mask=numeric_mask,
            categorical_mask=categorical_mask,
        ), TabularOutput(
            numeric=numeric_target,
            categorical=categorical_target,
        )

    def collate_fn(self, batch):
        """Custom collate function to handle batches of TabularInputs and TabularOutput."""
        inputs, targets = zip(*batch)

        # Stack numeric inputs and masks
        numeric_inputs = torch.stack([item.numeric for item in inputs])
        numeric_masks = torch.stack([item.numeric_mask for item in inputs])
        numeric_targets = torch.stack([item.numeric for item in targets])

        # Handle categorical data if present
        if self.has_categorical:
            categorical_inputs = torch.stack([item.categorical for item in inputs])
            categorical_masks = torch.stack([item.categorical_mask for item in inputs])
            categorical_targets = torch.stack([item.categorical for item in targets])
        else:
            categorical_inputs = None
            categorical_masks = None
            categorical_targets = None

        return (
            TabularInputs(
                numeric=numeric_inputs,
                categorical=categorical_inputs,
                numeric_mask=numeric_masks,
                categorical_mask=categorical_masks,
            ),
            TabularOutput(
                numeric=numeric_targets,
                categorical=categorical_targets,
            ),
        )
