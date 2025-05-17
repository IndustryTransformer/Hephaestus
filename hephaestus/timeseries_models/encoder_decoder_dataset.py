import pandas as pd
import torch
from torch.utils.data import Dataset

from hephaestus.timeseries_models.model_data_classes import TimeSeriesConfig
from hephaestus.utils.numeric_categorical import NumericCategoricalData


class EncoderDecoderDataset(Dataset):
    """Dataset class for encoder-decoder models with feature inputs and classification targets.

    This dataset separates inputs (features) from targets (class labels), allowing
    for encoder-decoder architectures where the encoder processes features and the
    decoder predicts the class with causal masking.

    Args:
        df (pd.DataFrame): The DataFrame containing the time series data
        config (TimeSeriesConfig): Configuration for the time series
        target_col (str): The name of the target column for classification

    Attributes:
        max_seq_len (int): Maximum sequence length
        df_input_categorical (pd.DataFrame): DataFrame of input categorical columns
        df_input_numeric (pd.DataFrame): DataFrame of input numeric columns
        df_target_categorical (pd.DataFrame): DataFrame of target categorical columns
        df_target_numeric (pd.DataFrame): DataFrame of target numeric columns

    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: TimeSeriesConfig,
        target_col: str = "class",
    ):
        self.max_seq_len = df.groupby("idx").size().max()
        self.target_col = target_col

        # Set df.idx to start from 0
        df.idx = df.idx - df.idx.min()
        df = df.set_index("idx")
        df.index.name = None

        # Convert categorical columns to integer tokens
        df_categorical = df.select_dtypes(include=["object"]).astype(str)
        self.df_categorical = self._convert_object_to_int_tokens(
            df_categorical, config.token_dict
        )

        # All numeric columns except the target become input features
        self.df_numeric = df.select_dtypes(include="number")

        self.classification_values = df[
            self.target_col
        ].unique()  # TODO Fix this for regression as
        # Separate inputs and targets
        self._separate_inputs_targets()

        # Store unique indices and config
        self.unique_indices = sorted(df.index.unique())
        self.config = config

    def _convert_object_to_int_tokens(self, df, token_dict):
        """Converts object columns to integer tokens using a token dictionary."""
        df = df.copy()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].map(token_dict)
        return df

    def _separate_inputs_targets(self):
        """Separates the inputs (features) from targets (class labels)."""
        # Initialize input dataframes as copies of the full numeric/categorical dfs
        self.df_input_numeric = self.df_numeric.copy()
        self.df_input_categorical = self.df_categorical.copy()

        # Initialize target dataframes. They will be populated if the target is found.
        # Ensure they have the same index as the main dataframes.
        # If self.df_numeric or self.df_categorical is empty, their index will be empty,
        # leading to empty target dataframes, which is handled by _get_tensor_from_df.
        self.df_target_numeric = pd.DataFrame(index=self.df_numeric.index)
        self.df_target_categorical = pd.DataFrame(index=self.df_categorical.index)

        if self.target_col in self.df_categorical.columns:
            # Target is categorical
            self.df_target_categorical = self.df_categorical[[self.target_col]].copy()
            # Remove target column from input categorical features, if it exists there
            if self.target_col in self.df_input_categorical.columns:
                self.df_input_categorical = self.df_input_categorical.drop(
                    columns=[self.target_col]
                )
        elif self.target_col in self.df_numeric.columns:
            # Target is numeric
            self.df_target_numeric = self.df_numeric[[self.target_col]].copy()
            # Remove target column from input numeric features, if it exists there
            if self.target_col in self.df_input_numeric.columns:
                self.df_input_numeric = self.df_input_numeric.drop(
                    columns=[self.target_col]
                )
        else:
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame columns. "
                f"Available numeric columns: {self.df_numeric.columns.tolist()}. "
                f"Available categorical columns: {self.df_categorical.columns.tolist()}."
            )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.unique_indices)

    def _get_tensor_from_df(self, df_name, set_idx):
        """Gets self.df_<df_name> for a given index and converts to tensor with padding."""
        # Skip if the DataFrame doesn't exist or is empty
        if not hasattr(self, df_name) or getattr(self, df_name).empty:
            return None

        df = getattr(self, df_name)

        # Get batch for the given index
        batch = df.loc[df.index == set_idx, :]

        # Convert DataFrame to torch.Tensor
        batch = torch.tensor(batch.values, dtype=torch.float32)

        # Add padding if needed
        batch_len, n_cols = batch.shape
        pad_len = self.max_seq_len - batch_len

        if pad_len > 0:
            padding = torch.full((pad_len, n_cols), float("nan"), dtype=torch.float32)
            batch = torch.cat([batch, padding], dim=0)

        # Transpose to shape: (n_cols, seq_len)
        batch = batch.permute(1, 0)

        return batch

    def __getitem__(self, idx) -> tuple[NumericCategoricalData, NumericCategoricalData]:
        """
        Get item(s) from the dataset.

        Args:
            idx: Integer index or slice object

        Returns:
            Tuple containing (inputs, targets), where both are NumericCategoricalData
            objects
        """
        # Handle slice indexing
        if isinstance(idx, slice):
            # Get the actual indices from the slice
            indices = self.unique_indices[idx]

            # Create a list of (inputs, targets) tuples
            items = [self._get_single_item(i) for i in indices]

            # Collate the items into a batch
            return self._collate_batch(items)

        # Handle integer indexing
        return self._get_single_item(self.unique_indices[idx])

    def _get_single_item(
        self, set_idx
    ) -> tuple[NumericCategoricalData, NumericCategoricalData]:
        """Get a single item from the dataset by actual index value."""
        # Get input features
        input_numeric = self._get_tensor_from_df("df_input_numeric", set_idx)
        input_categorical = self._get_tensor_from_df("df_input_categorical", set_idx)

        # Get target values
        target_numeric = self._get_tensor_from_df("df_target_numeric", set_idx)
        target_categorical = self._get_tensor_from_df("df_target_categorical", set_idx)

        # Create NumericCategoricalData objects for inputs and targets
        inputs = NumericCategoricalData(
            numeric=input_numeric,
            categorical=input_categorical,
        )

        targets = NumericCategoricalData(
            numeric=target_numeric,
            categorical=target_categorical,
        )

        return inputs, targets

    def _collate_batch(
        self, items
    ) -> tuple[NumericCategoricalData, NumericCategoricalData]:
        """
        Collate a list of (inputs, targets) tuples into a batch.

        Args:
            items: List of (inputs, targets) tuples

        Returns:
            Tuple containing (inputs, targets) batches
        """
        # Split inputs and targets
        input_items = [item[0] for item in items]
        target_items = [item[1] for item in items]

        # Collate inputs
        input_numeric_batch = (
            torch.stack([item.numeric for item in input_items])
            if input_items[0].numeric is not None
            else None
        )
        input_categorical_batch = (
            torch.stack([item.categorical for item in input_items])
            if input_items[0].categorical is not None
            else None
        )

        # Collate targets
        target_numeric_batch = (
            torch.stack([item.numeric for item in target_items])
            if target_items[0].numeric is not None
            else None
        )
        target_categorical_batch = (
            torch.stack([item.categorical for item in target_items])
            if target_items[0].categorical is not None
            else None
        )

        # Create batched NumericCategoricalData objects
        inputs_batch = NumericCategoricalData(
            numeric=input_numeric_batch,
            categorical=input_categorical_batch,
        )

        targets_batch = NumericCategoricalData(
            numeric=target_numeric_batch,
            categorical=target_categorical_batch,
        )

        return inputs_batch, targets_batch


def encoder_decoder_collate_fn(batch):
    """Custom collate function for EncoderDecoderDataset batches."""
    # Split inputs and targets
    input_items = [item[0] for item in batch]
    target_items = [item[1] for item in batch]

    # Collate inputs
    input_numeric_batch = (
        torch.stack([item.numeric for item in input_items])
        if input_items[0].numeric is not None
        else None
    )
    input_categorical_batch = (
        torch.stack([item.categorical for item in input_items])
        if input_items[0].categorical is not None
        else None
    )

    # Collate targets
    target_numeric_batch = (
        torch.stack([item.numeric for item in target_items])
        if target_items[0].numeric is not None
        else None
    )
    target_categorical_batch = (
        torch.stack([item.categorical for item in target_items])
        if target_items[0].categorical is not None
        else None
    )

    # Create batched NumericCategoricalData objects
    inputs_batch = NumericCategoricalData(
        numeric=input_numeric_batch,
        categorical=input_categorical_batch,
    )

    targets_batch = NumericCategoricalData(
        numeric=target_numeric_batch,
        categorical=target_categorical_batch,
    )

    return inputs_batch, targets_batch
