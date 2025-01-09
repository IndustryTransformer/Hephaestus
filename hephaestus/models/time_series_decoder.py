# %%
# import jax
import re
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx
from flax.struct import dataclass
from icecream import ic
from jax.lax import stop_gradient
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
    Parameters:
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
        numeric_indices (jnp.array): Array of indices for numeric columns.
        categorical_indices (jnp.array): Array of indices for categorical columns.
        object_tokens (list): List of unique values in categorical columns.
        numeric_mask_token (int): Index of numeric mask token.
        reservoir_vocab (list): List of words in custom vocabulary.
        reservoir_encoded (jnp.array): Encoded reservoir tokens.
        tokenizer (AutoTokenizer): Tokenizer for encoding tokens.
    Methods:
        generate(df: pd.DataFrame) -> TimeSeriesConfig:
            Generates a TimeSeriesConfig object based on the given DataFrame.


        Generates a TimeSeriesConfig object based on the given DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame containing the time series data.
        Returns:
            TimeSeriesConfig: The generated TimeSeriesConfig object.
    """

    numeric_token: str = None
    numeric_mask: str = None
    numeric_col_tokens: list = None
    categorical_col_tokens: list = None
    tokens: list = None
    token_dict: dict = None
    token_decoder_dict: dict = None
    n_tokens: int = None
    numeric_indices: nnx.Variable = None  # jnp.array = None
    categorical_indices: nnx.Variable = None  #  jnp.array = None
    object_tokens: list = None
    numeric_mask_token: int = None
    reservoir_vocab: list = None
    reservoir_encoded: nnx.Variable = None  #  jnp.array = None
    tokenizer: AutoTokenizer = None
    vocab_size: int = None
    ds_length: int = None
    n_columns: int = None

    @classmethod
    def generate(cls, df: pd.DataFrame) -> "TimeSeriesConfig":
        """
        Generate a TimeSeriesConfig object based on the given DataFrame.
        Args:
            cls (class): The class to instantiate.
            df (pd.DataFrame): The DataFrame containing the data.
        Returns:
            TimeSeriesConfig: The generated TimeSeriesConfig object.
        """

        # max_seq_len = df.groupby("idx").count().time_step.max()
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
        unique_values_per_column = df_categorical.apply(
            pd.Series.unique
        ).values  # .flatten().tolist()
        flattened_unique_values = np.concatenate(unique_values_per_column).tolist()
        object_tokens = list(set(flattened_unique_values))
        cls_dict["object_tokens"] = object_tokens
        # cls_dict["object_tokens"] = cls_dict["object_tokens"]

        # print(f'Type: {cls_dict["numeric_col_tokens"]=}')
        # print(f'Type: {cls_dict["categorical_col_tokens"]=}')
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
        numeric_indices = nnx.Variable(
            jnp.array([tokens.index(i) for i in numeric_col_tokens])
        )
        # numeric_indices = jnp.array([tokens.index(i) for i in numeric_col_tokens])
        cls_dict["numeric_indices"] = numeric_indices
        categorical_indices = nnx.Variable(
            jnp.array([tokens.index(i) for i in categorical_col_tokens])
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
        reservoir_encoded = nnx.Variable(
            tokenizer(
                reservoir_tokens_list,
                padding="max_length",
                max_length=8,  # TODO Make this dynamic
                truncation=True,
                return_tensors="jax",
                add_special_tokens=False,
            )["input_ids"]
        )  # TODO make this custom to reduce dictionary size
        cls_dict["reservoir_encoded"] = reservoir_encoded
        cls_dict["reservoir_vocab"] = reservoir_vocab
        cls_dict["ds_length"] = ds_length
        cls_dict["n_columns"] = len(df.columns)
        cls_dict["vocab_size"] = tokenizer.vocab_size
        df_categorical = convert_object_to_int_tokens(df_categorical, token_dict)

        return cls(**cls_dict)


class TimeSeriesDS(Dataset):
    def __init__(self, df: pd.DataFrame, config: TimeSeriesConfig):
        # Add nan padding to make sure all sequences are the same length
        # use the idx column to group by
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
        # return self.df.idx.max() + 1  # probably should be max idx + 1 thanks
        return self.df_numeric.index.nunique()

    def get_data(self, df_name, set_idx):
        """Gets self.df_<df_name> for a given index"""
        df = getattr(self, df_name)

        batch = df.loc[df.index == set_idx, :]
        batch = np.array(batch.values)

        # Add padding

        batch_len, n_cols = batch.shape
        pad_len = self.max_seq_len - batch_len
        padding = np.full((pad_len, n_cols), np.nan)
        batch = np.concatenate([batch, padding], axis=0)
        batch = np.swapaxes(batch, 0, 1)
        # if df_name == "df_categorical":
        #     # Cast to int
        #     batch = batch.astype(int)
        return batch

    def __getitem__(self, set_idx):
        if self.df_categorical.empty:
            categorical_inputs = None
        else:
            categorical_inputs = self.get_data("df_categorical", set_idx)
        numeric_inputs = self.get_data("df_numeric", set_idx)

        return numeric_inputs, categorical_inputs


class FeedForwardNetwork(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, rngs: nnx.Rngs):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.dense1 = nnx.Linear(in_features=512, out_features=512, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=d_model, out_features=512, rngs=rngs)
        # self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, deterministic: bool):
        # Feed Forward Network
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.dense2(x)
        # ic("About to call dropout2")
        x = self.dropout(x, deterministic=deterministic)
        # ic("Finished calling dropout2")
        return x


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.multi_head_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            # qkv_features=d_model,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs,
        )
        self.layer_norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.feed_forward_network = FeedForwardNetwork(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            rngs=rngs,
        )
        self.layer_norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(
        self,
        q: jnp.array,
        k: jnp.array,
        v: jnp.array,
        deterministic: bool,
        mask: jnp.array = None,
    ):
        if mask is not None:
            mask_shape = mask.shape
        else:
            mask_shape = None
        ic("Transformer Block", q.shape, k.shape, v.shape, mask_shape)
        attention = self.multi_head_attention(
            q, k, v, deterministic=deterministic, mask=mask
        )
        out = q + attention
        out = self.layer_norm1(out)
        # Feed Forward Network
        ffn = self.feed_forward_network(out, deterministic=deterministic)
        out = out + ffn
        out = self.layer_norm2(out)
        return out


class ReservoirEmbedding(nnx.Module):
    """
    A module for performing reservoir embedding on a given input.

    Args:
        dataset (SimpleDS): The dataset used for embedding.
        features (int): The number of features in the embedding.
        frozen_index (int, optional): The index of the embedding to freeze. Defaults to 0.
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
        features: int,
        rngs: nnx.Rngs,
        frozen_index: int = 0,
    ):
        self.config = config
        self.features = features
        self.frozen_index = frozen_index

        self.embedding = nnx.Embed(
            num_embeddings=self.config.vocab_size, features=self.features, rngs=rngs
        )

    # def __init__(self, config: TimeSeriesConfig, features: int, frozen_index: int = 0, rngs: nnx.Rngs):
    #     self.config = config
    #     self.features = features
    #     self.frozen_index = frozen_index

    #     # Modified to properly handle the random key initialization
    #     self.embedding = nnx.Param(
    #         "embedding",
    #         lambda key: nnx.initializers.normal(stddev=0.02)(
    #             key,
    #             (self.config.tokenizer.vocab_size, self.features),
    #         ),
    #         rngs=rngs  # Pass the rngs through
    #     )

    def __call__(self, base_indices: jnp.array):
        """
        Perform reservoir embedding on the given input.

        Args:
            base_indices (jnp.array): The base indices for embedding.

        Returns:
            jnp.array: The ultimate embedding after reservoir embedding.
        """
        token_reservoir_lookup = self.config.reservoir_encoded
        reservoir_indices = token_reservoir_lookup[base_indices]

        return_embed = self.embedding(reservoir_indices)
        return_embed = jnp.sum(return_embed, axis=-2)
        return return_embed

        # Create a mask for the frozen embedding
        # frozen_mask = jnp.arange(self.config.tokenizer.vocab_size) == self.frozen_index

        # # Set the frozen embedding to zero
        # frozen_embedding = jnp.where(
        #     frozen_mask[:, None], 0.0, self.embedding.embedding
        # )

        # # Stop gradient for the frozen embedding
        # penultimate_embedding = stop_gradient(frozen_embedding) + jnp.where(
        #     frozen_mask[:, None], 0.0, self.embedding.embedding - frozen_embedding
        # )
        # token_reservoir_lookup = self.config.reservoir_encoded
        # reservoir_indices = token_reservoir_lookup[base_indices]

        # ultimate_embedding = penultimate_embedding[reservoir_indices]
        # ultimate_embedding = jnp.sum(ultimate_embedding, axis=-2)
        # ic(ultimate_embedding.shape)

        # return ultimate_embedding


@dataclass
class ProcessedEmbeddings:
    column_embeddings: Optional[jnp.array] = None
    value_embeddings: Optional[jnp.array] = None


class TimeSeriesTransformer(nnx.Module):
    """
    Transformer-based model for time series data.

    Args:
        dataset (SimpleDS): The dataset object containing the time series data.
        d_model (int, optional): The dimensionality of the model. Defaults to 64.
        n_heads (int, optional): The number of attention heads. Defaults to 4.
        time_window (int, optional): The maximum length of the time window. Defaults to 10000.

    Methods:
        __call__(self, numeric_inputs: jnp.array, deterministic: bool, causal_mask: bool = True) -> jnp.array:
            Applies the transformer model to the input time series data.

    Attributes:
        dataset (SimpleDS): The dataset object containing the time series data.
        d_model (int): The dimensionality of the model.
        n_heads (int): The number of attention heads.
        time_window (int): The maximum length of the time window.
    """

    def __init__(
        self,
        config: TimeSeriesConfig,
        rngs=nnx.Rngs,
        d_model: int = 64,
        n_heads: int = 4,
    ):
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads
        self.time_window = 10000
        # self.embedding = nnx.Embed(
        #     num_embeddings=self.config.n_tokens, features=self.d_model, rngs=rngs
        # )
        self.embedding = ReservoirEmbedding(
            config=self.config, features=self.d_model, rngs=rngs
        )
        # ic(
        #     "Embedding Test Value",
        #     self.embedding(jnp.array([0])),
        #     self.embedding(jnp.array([0])).shape,
        # )
        self.transformer_block_0 = TransformerBlock(
            num_heads=self.n_heads,
            d_model=self.d_model,
            d_ff=64,
            dropout_rate=0.1,
            rngs=rngs,
        )
        self.transformer_block_chain = [
            TransformerBlock(
                num_heads=self.n_heads,
                d_model=self.d_model,
                d_ff=64,
                dropout_rate=0.1,
                rngs=rngs,
            )
            for i in range(1, 4)
        ]

    def process_numeric(self, numeric_inputs: jnp.array) -> ProcessedEmbeddings:
        """
        Processes the numeric inputs for the transformer model.

        Args:
            numeric_inputs (jnp.array): The numeric inputs to be processed.

        Returns:
            jnp.array: The processed numeric inputs.
        """
        # Create a nan mask for the numeric inputs
        nan_mask = stop_gradient(jnp.isnan(numeric_inputs))
        # ic(
        #     "Second Embedding Dim",
        #     self.embedding(jnp.array([0])),
        #     self.embedding(jnp.array([0])).shape,
        # )
        # Replace NaN values with zeros
        numeric_inputs = jnp.where(nan_mask, 0.0, numeric_inputs)
        repeated_numeric_indices = jnp.tile(
            self.config.numeric_indices, (numeric_inputs.shape[2], 1)
        )
        # repeated_numeric_indices = jnp.swapaxes(repeated_numeric_indices, 0, 1)
        repeated_numeric_indices = repeated_numeric_indices.T
        numeric_col_embeddings = self.embedding(repeated_numeric_indices)
        # Nan Masking
        numeric_col_embeddings = jnp.tile(
            # jnp.squeeze(
            numeric_col_embeddings[None, :, :, :],  # ),
            (numeric_inputs.shape[0], 1, 1, 1),
        )
        ic("col_token type", numeric_col_embeddings.dtype)
        numeric_embedding = self.embedding(
            jnp.array(self.config.token_dict[self.config.numeric_token])
        )
        ic(numeric_embedding.shape)

        numeric_embedding = numeric_inputs[:, :, :, None] * numeric_embedding
        ic(numeric_embedding.shape)

        numeric_embedding = jnp.where(
            nan_mask[:, :, :, None],
            self.embedding(jnp.array(self.config.numeric_mask_token)),
            numeric_embedding,
        )
        # End Nan Masking
        ic(numeric_embedding.shape)
        # numeric_embedding = self.embedding(numeric_embedding.astype(jnp.int32))
        return ProcessedEmbeddings(
            column_embeddings=numeric_col_embeddings,
            value_embeddings=numeric_embedding,
        )

    def process_categorical(
        self, categorical_inputs: Optional[jnp.array]
    ) -> ProcessedEmbeddings:
        """
        Processes the categorical inputs for the transformer model.

        Args:
            categorical_inputs (Optional[jnp.array]): The categorical inputs to be processed.

        Returns:
            jnp.array: The processed categorical inputs.
        """
        if categorical_inputs is None:
            return None, None
        # Make sure nans are set to <NAN> token
        categorical_inputs = jnp.where(
            jnp.isnan(categorical_inputs),
            jnp.array(self.config.token_dict["[NUMERIC_MASK]"]),
            categorical_inputs,
        )
        categorical_embeddings = self.embedding(categorical_inputs)
        ic(
            "Issue here",
            categorical_inputs.shape,
            "args",
            categorical_inputs.shape[2],
            self.config.categorical_indices.shape,
        )
        repeated_categorical_indices = jnp.tile(
            self.config.categorical_indices,
            # jnp.squeeze(self.config.categorical_indices),
            (categorical_inputs.shape[2], 1),
        )
        ic(repeated_categorical_indices.shape)
        repeated_categorical_indices = repeated_categorical_indices.T
        ic(repeated_categorical_indices.shape)
        categorical_col_embeddings = self.embedding(repeated_categorical_indices)
        ic("Extra dim here?", categorical_col_embeddings.shape)
        categorical_col_embeddings = jnp.tile(
            categorical_col_embeddings[None, :, :, :],
            (categorical_inputs.shape[0], 1, 1, 1),
        )
        ic(categorical_col_embeddings.shape)
        return ProcessedEmbeddings(
            column_embeddings=categorical_col_embeddings,
            value_embeddings=categorical_embeddings,
        )

    def combine_inputs(
        self, numeric: ProcessedEmbeddings, categorical: ProcessedEmbeddings
    ) -> ProcessedEmbeddings:
        """
        Combines numeric and categorical embeddings into a single ProcessedEmbeddings object.

        Args:
            numeric (ProcessedEmbeddings): The numeric embeddings to combine.
            categorical (ProcessedEmbeddings): The categorical embeddings to combine.

        Returns:
            ProcessedEmbeddings: A new ProcessedEmbeddings object containing the combined embeddings.

        Raises:
            ValueError: If neither numeric nor categorical embeddings are provided.

        """
        if (
            numeric.value_embeddings is not None
            and categorical.value_embeddings is not None
        ):
            ic(numeric.value_embeddings.shape, categorical.value_embeddings.shape)
            ic(numeric.column_embeddings.shape, categorical.column_embeddings.shape)
            value_embeddings = jnp.concatenate(
                [numeric.value_embeddings, categorical.value_embeddings],
                axis=1,
            )
            column_embeddings = jnp.concatenate(
                [
                    numeric.column_embeddings,
                    categorical.column_embeddings,
                ],
                axis=1,
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
        numeric_inputs: Optional[jnp.array],
        categorical_inputs: Optional[jnp.array],
    ):
        """
        Generates a causal mask for the given numeric and categorical inputs.
        Args:
            numeric_inputs (Optional[jnp.array]): Numeric inputs.
            categorical_inputs (Optional[jnp.array]): Categorical inputs.
        Returns:
            jnp.array: The generated causal mask.
        Raises:
            ValueError: If no numeric or categorical inputs are provided.
        """

        if numeric_inputs is not None and categorical_inputs is not None:
            mask_input = jnp.concatenate([numeric_inputs, categorical_inputs], axis=1)
        elif numeric_inputs is not None:
            mask_input = numeric_inputs
        elif categorical_inputs is not None:
            mask_input = categorical_inputs
        else:
            raise ValueError("No numeric or categorical inputs provided.")
        causal_mask = nnx.make_causal_mask(mask_input)
        pad_mask = nnx.make_attention_mask(mask_input, mask_input)
        mask = nnx.combine_masks(causal_mask, pad_mask)
        return mask

    def __call__(
        self,
        numeric_inputs: Optional[jnp.array] = None,
        categorical_inputs: Optional[jnp.array] = None,
        deterministic: bool = False,
        causal_mask: bool = True,
        encoder_mask: bool = False,
    ):
        ic(numeric_inputs.shape, categorical_inputs.shape)
        processed_numeric = self.process_numeric(numeric_inputs)
        processed_categorical = self.process_categorical(categorical_inputs)

        combined_inputs = self.combine_inputs(processed_numeric, processed_categorical)

        if causal_mask:
            mask = self.causal_mask(
                numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs
            )
        else:
            mask = None
        # pos_dim = 0 # TODO Add this back in
        ic(
            combined_inputs.value_embeddings.shape,
            combined_inputs.column_embeddings.shape,
        )
        out = self.transformer_block_0(
            q=combined_inputs.value_embeddings,
            k=combined_inputs.column_embeddings,
            v=combined_inputs.value_embeddings,
            deterministic=deterministic,
            # decode=False,
            mask=mask,
        )
        for transformer_block_iter in self.transformer_block_chain:
            out = transformer_block_iter(
                q=out,
                k=combined_inputs.column_embeddings,
                v=out,
                deterministic=deterministic,
                # decode=False,
                mask=mask,
            )

        return out


class TimeSeriesDecoder(nnx.Module):
    def __init__(
        self,
        config: TimeSeriesConfig,
        rngs: nnx.Rngs,
        d_model: int = 64,
        n_heads: int = 4,
    ):
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads
        self.time_series_transformer = TimeSeriesTransformer(
            config=self.config, d_model=self.d_model, n_heads=self.n_heads, rngs=rngs
        )
        # self.sequential = nnx.Sequential(
        #     nnx.Linear(
        #         in_features=self.d_model, out_features=self.d_model * 2, rngs=rngs
        #     ),
        #     nnx.relu,
        #     nnx.Linear(
        #         in_features=d_model * 2,
        #         out_features=len(self.config.numeric_indices),
        #         rngs=rngs,
        #     ),
        # )
        self.numeric_linear1 = nnx.Linear(
            in_features=d_model * self.config.n_columns,  # self.config.ds_length,
            out_features=self.d_model * 2,
            rngs=rngs,
        )
        self.numeric_linear2 = nnx.Linear(
            in_features=d_model * 2,
            out_features=len(self.config.numeric_col_tokens),
            rngs=rngs,
        )
        self.categorical_dense1 = nnx.Linear(
            in_features=self.d_model,
            out_features=len(self.config.token_decoder_dict.items()),
            rngs=rngs,
        )
        self.categorical_dense2 = nnx.Linear(
            in_features=self.config.n_columns,
            out_features=len(self.config.categorical_col_tokens),
            rngs=rngs,
        )

    # config: TimeSeriesConfig
    # d_model: int = 64 * 10
    # n_heads: int = 4

    def __call__(
        self,
        numeric_inputs: jnp.array,
        categorical_inputs: Optional[jnp.array] = None,
        deterministic: bool = False,
        causal_mask: bool = True,
    ) -> jnp.array:
        """ """
        out = self.time_series_transformer(
            numeric_inputs=numeric_inputs,
            categorical_inputs=jnp.astype(categorical_inputs, jnp.int32),
            deterministic=deterministic,
            causal_mask=causal_mask,
        )

        numeric_out = out.swapaxes(1, 2)
        numeric_out = numeric_out.reshape(
            numeric_out.shape[0], numeric_out.shape[1], -1
        )  # TODO This is wrong. Make this
        #  TODO WORK HERE!!!!! be of shape (batch_size, )

        # numeric_out = self.sequential(numeric_out)
        ic("Starting shit")
        ic(numeric_out.shape, self.config.ds_length)
        numeric_out = self.numeric_linear1(numeric_out)
        ic(numeric_out.shape)
        numeric_out = nnx.relu(numeric_out)
        ic(numeric_out.shape)
        numeric_out = self.numeric_linear2(numeric_out)
        ic(numeric_out.shape)
        numeric_out = numeric_out.swapaxes(1, 2)

        if categorical_inputs is not None:
            categorical_out = out.copy()
            ic(categorical_out.shape)
            categorical_out = self.categorical_dense1(categorical_out)

            categorical_out = nnx.relu(categorical_out)

            # categorical_out = categorical_out.swapaxes(1, 3)
            ic(
                "Categorical after dense1",
                categorical_out.shape,
            )
            categorical_out = categorical_out.swapaxes(1, 3)
            ic("Categorical out after first swap", categorical_out.shape)
            categorical_out = self.categorical_dense2(categorical_out)
            ic("Categorical after dense2", categorical_out.shape)
            categorical_out = categorical_out.swapaxes(1, 3)
            ic("Categorical after swap", categorical_out.shape)

        else:
            categorical_out = None

        return {"numeric_out": numeric_out, "categorical_out": categorical_out}


class PositionalEncoding(nnx.Module):
    """ """

    def __init__(self, max_len: int, d_pos_encoding: int):
        self.max_len = max_len  # Maximum length of the input sequences
        self.d_pos_encoding = d_pos_encoding  # Dimensionality of the embeddings/inputs

    def __call__(self, x):
        """
        Forward pass of the positional encoding. Concatenates positional encoding to
        the input.

        Args:
            x: Input data. Shape: (batch_size, seq_len, d_model)

        Returns:
            Output with positional encoding added. Shape: (batch_size, seq_len, d_model)
        """
        n_epochs, n_columns, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} is larger than the",
                f"maximum length {self.max_len}",
            )

        # Calculate positional encoding
        position = jnp.arange(self.max_len)[:, jnp.newaxis]
        div_term = jnp.exp(
            jnp.arange(0, self.d_pos_encoding, 2)
            * -(jnp.log(10000.0) / self.d_pos_encoding)
        )
        pe = jnp.zeros((self.max_len, self.d_pos_encoding))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[:seq_len, :]
        pe = pe[None, :, :, None]

        pe = jnp.tile(pe, (n_epochs, 1, 1, n_columns))

        pe = pe.transpose((0, 3, 1, 2))  #

        result = x + pe

        return result


# %%
