# %%
import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def split_complex_word(word):
    """
    Splits a complex word into its individual parts.

    Args:
        word (str): The complex word to be split.

    Returns:
        list: A list of individual parts of the complex word.

    Example:
        >>> split_complex_word("myComplexWord")
        ['my', 'Complex', 'Word']
    """

    # Step 1: Split by underscore, preserving content within square brackets
    parts = re.split(r"(_|\[.*?\])", word)
    parts = [p for p in parts if p]  # Remove empty strings

    # Step 2: Split camelCase for parts not in square brackets
    def split_camel_case(s):
        """
        Splits a camel case string into a list of words.

        Args:
            s (str): The camel case string to be split.

        Returns:
            list: A list of words obtained from the camel case string.

        Examples:
            >>> split_camel_case("helloWorld")
            ['hello', 'World']
            >>> split_camel_case("thisIsATest")
            ['this', 'Is', 'A', 'Test']
        """

        if s.startswith("[") and s.endswith("]"):
            return [s]
        return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", s)

    # Step 3: Apply camelCase splitting to each part and flatten the result
    result = [item for part in parts for item in split_camel_case(part)]

    return result


def convert_object_to_int_tokens(df, token_dict):
    """
    Converts object columns to integer tokens using a token dictionary.

    Args:
        df (pandas.DataFrame): The DataFrame containing the object columns to be converted.
        token_dict (dict): A dictionary mapping object values to integer tokens.

    Returns:
        pandas.DataFrame: The DataFrame with object columns converted to integer tokens.
    """

    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(token_dict)
    return df


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

    def __len__(self):
        """Return the length of the dataset."""
        return self.df_numeric.index.nunique()

    def get_data(self, df_name, set_idx):
        """Gets self.df_<df_name> for a given index."""
        df = getattr(self, df_name)

        batch = df.loc[df.index == set_idx, :]
        batch = np.array(batch.values)

        # Add padding
        batch_len, n_cols = batch.shape
        pad_len = self.max_seq_len - batch_len
        padding = np.full((pad_len, n_cols), np.nan)
        batch = np.concatenate([batch, padding], axis=0)
        batch = np.swapaxes(batch, 0, 1)
        return batch

    def __getitem__(self, set_idx):
        """Get item from the dataset."""
        if self.df_categorical.empty:
            categorical_inputs = None
        else:
            categorical_inputs = self.get_data("df_categorical", set_idx)
        numeric_inputs = self.get_data("df_numeric", set_idx)

        return numeric_inputs, categorical_inputs


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward neural network module.

    Args:
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feed-forward layer.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.dense1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(d_model, d_model)

    def forward(self, x, deterministic: bool = False):
        """Forward pass of the feed-forward network."""
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x) if not deterministic else x
        x = self.dense2(x)
        x = self.dropout(x) if not deterministic else x
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block module.

    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feed-forward layer.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, num_heads: int, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward_network = FeedForwardNetwork(
            d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        deterministic: bool = False,
        mask: torch.Tensor = None,
    ):
        """Forward pass of the transformer block."""
        if mask is not None:
            mask_shape = mask.shape
        else:
            mask_shape = None
        ic("Transformer Block", q.shape, k.shape, v.shape, mask_shape)

        # PyTorch's MultiheadAttention has a different signature than Flax
        attention_output, _ = self.multi_head_attention(
            query=q, key=k, value=v, attn_mask=mask, need_weights=False
        )

        out = q + attention_output
        out = self.layer_norm1(out)

        # Feed Forward Network
        ffn = self.feed_forward_network(out, deterministic=deterministic)
        out = out + ffn
        out = self.layer_norm2(out)
        return out


class ReservoirEmbedding(nn.Module):
    """
    A module for performing reservoir embedding on a given input.

    Args:
        config (TimeSeriesConfig): Configuration for the time series.
        features (int): The number of features in the embedding.
        frozen_index (int, optional): The index of the embedding to freeze. Defaults to 0.
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
        features: int,
        frozen_index: int = 0,
    ):
        super().__init__()
        self.config = config
        self.features = features
        self.frozen_index = frozen_index

        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size, embedding_dim=self.features
        )

    def forward(self, base_indices: torch.Tensor):
        """
        Perform reservoir embedding on the given input.
        """
        # Get the encoded tokens from config
        token_reservoir_lookup = self.config.reservoir_encoded

        # Convert base_indices to PyTorch device
        if not isinstance(base_indices, torch.Tensor):
            base_indices = torch.tensor(
                base_indices, device=self.embedding.weight.device
            )
        else:
            base_indices = base_indices.to(self.embedding.weight.device)

        # Get reservoir indices for the provided base_indices
        reservoir_indices = token_reservoir_lookup[base_indices.long()]

        # Embed the reservoir indices
        return_embed = self.embedding(reservoir_indices)

        # Sum across the embedding dimension
        return_embed = torch.sum(return_embed, dim=-2, dtype=torch.float32)
        return return_embed


@dataclass
class ProcessedEmbeddings:
    """
    Data class to store processed embeddings.
    """

    column_embeddings: Optional[torch.Tensor] = None
    value_embeddings: Optional[torch.Tensor] = None


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for time series data.
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
        d_model: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads
        self.time_window = 10000

        self.embedding = ReservoirEmbedding(config=self.config, features=self.d_model)

        self.transformer_block_0 = TransformerBlock(
            num_heads=self.n_heads, d_model=self.d_model, d_ff=64, dropout_rate=0.1
        )

        self.transformer_block_chain = nn.ModuleList(
            [
                TransformerBlock(
                    num_heads=self.n_heads,
                    d_model=self.d_model,
                    d_ff=64,
                    dropout_rate=0.1,
                )
                for i in range(1, 4)
            ]
        )

    def process_numeric(self, numeric_inputs: torch.Tensor) -> ProcessedEmbeddings:
        """Processes the numeric inputs for the transformer model."""
        # Create a nan mask for the numeric inputs
        nan_mask = torch.isnan(numeric_inputs).detach()

        # Replace NaN values with zeros
        numeric_inputs = torch.where(
            nan_mask, torch.zeros_like(numeric_inputs), numeric_inputs
        )

        # Process numeric indices
        if not isinstance(self.config.numeric_indices, torch.Tensor):
            numeric_indices = torch.tensor(
                self.config.numeric_indices, device=next(self.parameters()).device
            )
        else:
            numeric_indices = self.config.numeric_indices.to(
                next(self.parameters()).device
            )

        repeated_numeric_indices = numeric_indices.repeat(
            numeric_inputs.shape[2], 1
        ).t()
        numeric_col_embeddings = self.embedding(repeated_numeric_indices)

        # Nan Masking
        numeric_col_embeddings = numeric_col_embeddings.unsqueeze(0).expand(
            numeric_inputs.shape[0], -1, -1, -1
        )

        ic("col_token type", numeric_col_embeddings.dtype)
        numeric_embedding = self.embedding(
            torch.tensor(
                self.config.token_dict[self.config.numeric_token],
                device=next(self.parameters()).device,
            )
        )
        ic(numeric_embedding.shape)

        # Multiply numeric values with embedding
        numeric_embedding = numeric_inputs.unsqueeze(-1) * numeric_embedding
        ic(numeric_embedding.shape)

        # Replace NaN values with mask token embedding
        mask_token = torch.tensor(
            self.config.numeric_mask_token, device=next(self.parameters()).device
        )
        mask_embedding = self.embedding(mask_token)
        numeric_embedding = torch.where(
            nan_mask.unsqueeze(-1),
            mask_embedding.expand_as(numeric_embedding),
            numeric_embedding,
        )

        ic(numeric_embedding.shape)
        return ProcessedEmbeddings(
            column_embeddings=numeric_col_embeddings,
            value_embeddings=numeric_embedding,
        )

    def process_categorical(
        self, categorical_inputs: Optional[torch.Tensor]
    ) -> ProcessedEmbeddings:
        """Processes the categorical inputs for the transformer model."""
        if categorical_inputs is None:
            return ProcessedEmbeddings(None, None)

        # Make sure nans are set to <NAN> token
        nan_mask = torch.isnan(categorical_inputs)
        mask_token = torch.tensor(
            self.config.token_dict["[NUMERIC_MASK]"],
            device=next(self.parameters()).device,
        )
        categorical_inputs = torch.where(
            nan_mask, mask_token.expand_as(categorical_inputs), categorical_inputs
        )

        # Convert to long for embedding
        categorical_inputs = categorical_inputs.long()

        # Get categorical embeddings
        categorical_embeddings = self.embedding(categorical_inputs)

        # Process categorical indices
        if not isinstance(self.config.categorical_indices, torch.Tensor):
            categorical_indices = torch.tensor(
                self.config.categorical_indices, device=next(self.parameters()).device
            )
        else:
            categorical_indices = self.config.categorical_indices.to(
                next(self.parameters()).device
            )

        ic(
            "Issue here",
            categorical_inputs.shape,
            "args",
            categorical_inputs.shape[2],
            categorical_indices.shape,
        )

        repeated_categorical_indices = categorical_indices.repeat(
            categorical_inputs.shape[2], 1
        ).t()
        ic(repeated_categorical_indices.shape)

        categorical_col_embeddings = self.embedding(repeated_categorical_indices)
        ic("Extra dim here?", categorical_col_embeddings.shape)

        categorical_col_embeddings = categorical_col_embeddings.unsqueeze(0).expand(
            categorical_inputs.shape[0], -1, -1, -1
        )
        ic(categorical_col_embeddings.shape)

        return ProcessedEmbeddings(
            column_embeddings=categorical_col_embeddings,
            value_embeddings=categorical_embeddings,
        )

    def combine_inputs(
        self, numeric: ProcessedEmbeddings, categorical: ProcessedEmbeddings
    ) -> ProcessedEmbeddings:
        """Combines numeric and categorical embeddings."""
        if (
            numeric.value_embeddings is not None
            and categorical.value_embeddings is not None
        ):
            ic(numeric.value_embeddings.shape, categorical.value_embeddings.shape)
            ic(numeric.column_embeddings.shape, categorical.column_embeddings.shape)
            value_embeddings = torch.cat(
                [numeric.value_embeddings, categorical.value_embeddings],
                dim=1,
            )
            column_embeddings = torch.cat(
                [
                    numeric.column_embeddings,
                    categorical.column_embeddings,
                ],
                dim=1,
            )
        elif numeric.value_embeddings is not None:
            value_embeddings = numeric.value_embeddings
            column_embeddings = numeric.column_embeddings
        elif categorical.value_embeddings is not None:
            value_embeddings = categorical.value_embeddings
            column_embeddings = categorical.column_embeddings
        else:
            raise ValueError("No numeric or categorical inputs provided.")

        return ProcessedEmbeddings(
            value_embeddings=value_embeddings, column_embeddings=column_embeddings
        )

    def causal_mask(
        self,
        numeric_inputs: Optional[torch.Tensor],
        categorical_inputs: Optional[torch.Tensor],
    ):
        """Generates a causal mask for the inputs."""
        if numeric_inputs is not None and categorical_inputs is not None:
            mask_input = torch.cat([numeric_inputs, categorical_inputs], dim=1)
        elif numeric_inputs is not None:
            mask_input = numeric_inputs
        elif categorical_inputs is not None:
            mask_input = categorical_inputs
        else:
            raise ValueError("No numeric or categorical inputs provided.")

        # Create causal mask (lower triangular)
        seq_len = mask_input.size(2)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(next(self.parameters()).device)

        # Create padding mask (for nan values)
        if torch.is_floating_point(mask_input):
            pad_mask = torch.isnan(mask_input).any(dim=1)
        else:
            # For categorical data
            pad_mask = (mask_input == self.config.token_dict["[PAD]"]).any(dim=1)

        # Combine masks
        # In PyTorch attention, True values are masked out
        combined_mask = mask
        if pad_mask is not None:
            # Expand pad_mask to match dimensions
            expanded_pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
            combined_mask = combined_mask | expanded_pad_mask

        return combined_mask

    def forward(
        self,
        numeric_inputs: Optional[torch.Tensor] = None,
        categorical_inputs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        causal_mask: bool = True,
        encoder_mask: bool = False,
    ):
        """Forward pass of the transformer model."""
        # Convert inputs to PyTorch tensors if they aren't already
        if numeric_inputs is not None and not isinstance(numeric_inputs, torch.Tensor):
            numeric_inputs = torch.tensor(
                numeric_inputs,
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )

        if categorical_inputs is not None and not isinstance(
            categorical_inputs, torch.Tensor
        ):
            categorical_inputs = torch.tensor(
                categorical_inputs,
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )

        ic(numeric_inputs.shape, categorical_inputs.shape)
        processed_numeric = self.process_numeric(numeric_inputs)
        processed_categorical = self.process_categorical(categorical_inputs)

        combined_inputs = self.combine_inputs(processed_numeric, processed_categorical)

        mask = (
            self.causal_mask(
                numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
            )
            if causal_mask
            else None
        )

        ic(
            combined_inputs.value_embeddings.shape,
            combined_inputs.column_embeddings.shape,
        )

        out = self.transformer_block_0(
            q=combined_inputs.value_embeddings,
            k=combined_inputs.column_embeddings,
            v=combined_inputs.value_embeddings,
            deterministic=deterministic,
            mask=mask,
        )

        for transformer_block_iter in self.transformer_block_chain:
            out = transformer_block_iter(
                q=out,
                k=combined_inputs.column_embeddings,
                v=out,
                deterministic=deterministic,
                mask=mask,
            )

        return out


class TimeSeriesDecoder(nn.Module):
    """
    Decoder module for time series data.
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
        d_model: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads

        self.time_series_transformer = TimeSeriesTransformer(
            config=self.config, d_model=self.d_model, n_heads=self.n_heads
        )

        self.numeric_linear1 = nn.Linear(
            d_model * self.config.n_columns, self.d_model * 2
        )

        self.numeric_linear2 = nn.Linear(
            d_model * 2, len(self.config.numeric_col_tokens)
        )

        self.categorical_dense1 = nn.Linear(
            self.d_model, len(self.config.token_decoder_dict.items())
        )

        self.categorical_dense2 = nn.Linear(
            self.config.n_columns, len(self.config.categorical_col_tokens)
        )

    def forward(
        self,
        numeric_inputs: torch.Tensor,
        categorical_inputs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        causal_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the decoder."""
        # Convert inputs to PyTorch tensors if they aren't already
        if not isinstance(numeric_inputs, torch.Tensor):
            numeric_inputs = torch.tensor(
                numeric_inputs,
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )

        if categorical_inputs is not None and not isinstance(
            categorical_inputs, torch.Tensor
        ):
            categorical_inputs = torch.tensor(
                categorical_inputs,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )

        out = self.time_series_transformer(
            numeric_inputs=numeric_inputs,
            categorical_inputs=categorical_inputs,
            deterministic=deterministic,
            causal_mask=causal_mask,
        )

        numeric_out = out.transpose(1, 2)
        numeric_out = numeric_out.reshape(
            numeric_out.shape[0], numeric_out.shape[1], -1
        )

        ic("Starting numeric output processing")
        ic(numeric_out.shape, self.config.ds_length)
        numeric_out = self.numeric_linear1(numeric_out)
        numeric_out = self.numeric_linear2(numeric_out)

        categorical_out = self.categorical_dense1(out)
        categorical_out = self.categorical_dense2(categorical_out)

        return {
            "numeric": numeric_out,
            "categorical": categorical_out,
        }


class PositionalEncoding(nn.Module):
    """
    Positional encoding module.

    Args:
        max_len (int): Maximum length of the input sequences.
        d_pos_encoding (int): Dimensionality of the positional encoding.

    Attributes:
        max_len (int): Maximum length of the input sequences.
        d_pos_encoding (int): Dimensionality of the positional encoding.
    """

    def __init__(self, max_len: int, d_pos_encoding: int):
        super().__init__()
        self.max_len = max_len  # Maximum length of the input sequences
        self.d_pos_encoding = d_pos_encoding  # Dimensionality of the embeddings/inputs

    def forward(self, x):
        """
        Forward pass of the positional encoding. Adds positional encoding to
        the input.

        Args:
            x: Input data. Shape: (batch_size, n_columns, seq_len, d_model)

        Returns:
            Output with positional encoding added. Shape: (batch_size, n_columns, seq_len, d_model)
        """
        n_epochs, n_columns, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} is larger than the "
                f"maximum length {self.max_len}"
            )

        # Calculate positional encoding
        position = torch.arange(
            self.max_len, dtype=torch.float, device=x.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_pos_encoding, 2, dtype=torch.float, device=x.device)
            * (-math.log(10000.0) / self.d_pos_encoding)
        )

        pe = torch.zeros((self.max_len, self.d_pos_encoding), device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Slice to required sequence length
        pe = pe[:seq_len, :]

        # Reshape for broadcasting
        pe = pe.unsqueeze(0).unsqueeze(3)  # Shape: (1, seq_len, d_pos_encoding, 1)

        # Repeat to match batch and columns dimensions
        pe = pe.repeat(
            n_epochs, 1, 1, n_columns
        )  # Shape: (n_epochs, seq_len, d_pos_encoding, n_columns)

        # Permute to match input dimensions
        pe = pe.permute(
            0, 3, 1, 2
        )  # Shape: (n_epochs, n_columns, seq_len, d_pos_encoding)

        # Add to input
        result = x + pe

        return result
