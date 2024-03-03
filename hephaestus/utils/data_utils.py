import time
from dataclasses import dataclass, field
from itertools import chain

import jax
import jax.numpy as jnp
import pandas as pd
from flax import struct  # Flax dataclasses
from icecream import ic
from jax import random
from jaxlib.xla_extension import ArrayImpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TabularTimeSeriesData(Dataset):
    def __init__(
        self,
        df,
        target_column=None,
        batch_size=32,
        train=True,
        scale_target=True,
    ):
        df = df
        self.batch_size = batch_size
        self.target_column = target_column
        self.scale_target = scale_target
        self.special_tokens = ["[PAD]", "[NUMERIC_MASK]", "[MASK]"]

        self.cat_mask = "[MASK]"

        self.category_columns = df.select_dtypes(include=["object"]).columns.tolist()
        ic(self.category_columns)
        if self.target_column == []:
            raise ValueError("target_column cannot be an empty list")
        self.numeric_columns = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.tokens = list(
            chain(
                self.special_tokens,
                df.columns.to_list(),
                list(set(df[self.category_columns].values.flatten().tolist())),
            )
        )
        self.token_dict = {token: i for i, token in enumerate(self.tokens)}
        self.token_decoder_dict = {i: token for i, token in enumerate(self.tokens)}
        self.cat_mask_token = self.token_dict[self.cat_mask]
        self.scaler = StandardScaler()
        if self.target_column is None and not scale_target:
            raise ValueError("If target_column is None, scale_target must be False")
        if self.scale_target:
            self.scaler_target = StandardScaler()
            df[self.target_column] = self.scaler_target.fit_transform(
                df[[self.target_column]]
            )
        else:
            self.scaler_target = None

        if self.target_column is not None:
            self.numeric_columns.remove(self.target_column)
            self.target_column = [self.target_column]

        self.col_tokens = self.category_columns + self.numeric_columns
        self.n_cat_cols = len(self.category_columns)
        # self.numeric_columns = self.numeric_columns.remove(self.target_column[0])
        numeric_scaled = self.scaler.fit_transform(df[self.numeric_columns])
        df[self.numeric_columns] = numeric_scaled
        self.n_numeric_cols = len(self.numeric_columns)
        for col in self.category_columns:
            df[col] = df[col].map(self.token_dict)

        self.numeric_indices = jnp.array(
            [self.tokens.index(col) for col in self.numeric_columns]
        )
        self.numeric_mask_token = jnp.array(self.token_dict["[NUMERIC_MASK]"])
        if self.target_column is not None:
            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                shuffle=False,
            )
        else:
            X = df
            y = None
            self.X_train, self.X_test = train_test_split(
                X,
                test_size=0.2,
                shuffle=False,
            )
            self.y_train, self.y_test = None, None

        self.n_tokens = len(self.tokens)
        self.numeric_col_tokens = jnp.array(
            [self.token_dict[i] for i in self.numeric_columns]
        )
        self.col_indices = jnp.array(
            [self.tokens.index(col) for col in self.col_tokens]
        )
        if train:
            self.X_numeric = self.X_train[self.numeric_columns]
            self.X_categorical = self.X_train[self.category_columns]
            self.y = self.y_train
        else:
            self.X_numeric = self.X_test[self.numeric_columns]
            self.X_categorical = self.X_test[self.category_columns]
            self.y = self.y_test
        # Because pandas is the worst
        self.X_numeric = self.X_numeric.reset_index(drop=True)
        self.X_categorical = self.X_categorical.reset_index(drop=True)
        if self.target_column is not None:
            self.y = self.y.reset_index(drop=True)

    def __len__(self):
        return len(self.X_categorical) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        categorical_values = jnp.array(
            self.X_categorical.iloc[start_idx:end_idx, :].values
        )
        # print(categorical_values.shape)
        numeric_targets = jnp.array(self.X_numeric.iloc[start_idx:end_idx, :].values)
        if self.target_column is not None:
            y = jnp.array(self.y.iloc[start_idx:end_idx, :].values)
        else:
            y = jnp.nan
        return categorical_values, numeric_targets, y


@dataclass
class TabularDS:
    df: pd.DataFrame = field(repr=False)
    target_column: str = field(default=None)
    seed: int = 42
    special_tokens: list = field(
        default_factory=lambda: ["[PAD]", "[NUMERIC_MASK]", "[MASK]"]
    )
    cat_mask: str = "[MASK]"
    cat_mask_token: int = field(init=False)
    n_tokens: int = field(init=False)
    n_cat_cols: int = field(init=False)
    n_numeric_cols: int = field(init=False)
    numeric_col_tokens: jnp.array = field(init=False)
    category_columns: list = field(init=False)
    col_tokens: list = field(init=False)
    numeric_mask_token: list = field(init=False)
    numeric_indices: jnp.array = field(init=False)
    col_indices: jnp.array = field(init=False)
    shuffle: bool = True

    def __post_init__(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )  # This is where randomness is introduced
        self.category_columns = self.df.select_dtypes(
            include=["object"]
        ).columns.tolist()
        self.numeric_columns = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.tokens = list(
            chain(
                self.special_tokens,
                self.df.columns.to_list(),
                list(set(self.df[self.category_columns].values.flatten().tolist())),
            )
        )
        self.token_dict = {token: i for i, token in enumerate(self.tokens)}
        self.token_decoder_dict = {i: token for i, token in enumerate(self.tokens)}
        self.cat_mask_token = self.token_dict[self.cat_mask]
        self.scaler = StandardScaler()

        if self.target_column is not None:
            self.numeric_columns.remove(self.target_column)
            self.target_column = [self.target_column]

        self.col_tokens = self.category_columns + self.numeric_columns
        self.n_cat_cols = len(self.category_columns)
        # self.numeric_columns = self.numeric_columns.remove(self.target_column[0])
        numeric_scaled = self.scaler.fit_transform(self.df[self.numeric_columns])
        self.df[self.numeric_columns] = numeric_scaled
        self.n_numeric_cols = len(self.numeric_columns)
        for col in self.category_columns:
            self.df[col] = self.df[col].map(self.token_dict)

        self.numeric_indices = jnp.array(
            [self.tokens.index(col) for col in self.numeric_columns]
        )
        self.numeric_mask_token = jnp.array(self.token_dict["[NUMERIC_MASK]"])
        if self.target_column is not None:
            X = self.df.drop(self.target_column, axis=1)
            y = self.df[self.target_column]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2
            )
        else:
            X = self.df
            y = None
            self.X_train, self.X_test = train_test_split(X, test_size=0.2)
            self.y_train, self.y_test = None, None

        self.n_tokens = len(self.tokens)
        self.numeric_col_tokens = jnp.array(
            [self.token_dict[i] for i in self.numeric_columns]
        )
        self.col_indices = jnp.array(
            [self.tokens.index(col) for col in self.col_tokens]
        )
        X_train_numeric = self.X_train[self.numeric_columns]
        X_train_categorical = self.X_train[self.category_columns]
        X_test_numeric = self.X_test[self.numeric_columns]
        X_test_categorical = self.X_test[self.category_columns]

        self.X_train_numeric = jnp.array(X_train_numeric.values)

        self.X_train_categorical = jnp.array(X_train_categorical.values)

        self.X_test_numeric = jnp.array(X_test_numeric.values)

        self.X_test_categorical = jnp.array(X_test_categorical.values)

        if self.target_column is not None:
            self.y_train = jnp.array(self.y_train.values)

            self.y_test = jnp.array(self.y_test.values)
        else:
            self.y_train = None
            self.y_test = None


@struct.dataclass
class TimeSeriesRegression:
    categorical_inputs: jnp.ndarray
    numeric_inputs: jnp.ndarray
    y: jnp.ndarray = None
    time_window: int = 100


@struct.dataclass
class MTMModelInputs:
    categorical_mask: jnp.ndarray
    numeric_mask: jnp.ndarray
    numeric_targets: jnp.ndarray
    categorical_targets: jnp.ndarray


@struct.dataclass
class TimeSeriesModelInputs:
    categorical_mask: jnp.ndarray
    numeric_mask: jnp.ndarray
    numeric_targets: jnp.ndarray
    categorical_targets: jnp.ndarray


def create_mtm_model_inputs(
    dataset: TabularDS,
    idx: int = None,
    batch_size: int = None,
    set: str = "train",
    probability=0.8,
    device: jax.Device = None,
):
    if device is None:
        device = jax.devices()[0]
    if set == "train":
        categorical_values = dataset.X_train_categorical
        numeric_targets = dataset.X_train_numeric
    elif set == "test":
        categorical_values = dataset.X_test_categorical
        numeric_targets = dataset.X_test_numeric
    else:
        raise ValueError("set must be either 'train' or 'test'")

    if idx is None:
        idx = 0
    if batch_size is None:
        batch_size = numeric_targets.shape[0]

    categorical_values = categorical_values[idx : idx + batch_size, :]
    numeric_targets = numeric_targets[idx : idx + batch_size, :]

    categorical_mask = mask_tensor(categorical_values, dataset, probability=probability)
    numeric_mask = mask_tensor(numeric_targets, dataset, probability=probability)

    numeric_col_tokens = dataset.numeric_col_tokens.clone()
    repeated_numeric_col_tokens = jnp.tile(
        numeric_col_tokens, (categorical_values.shape[0], 1)
    )
    categorical_targets = jnp.concatenate(
        [
            categorical_values,
            repeated_numeric_col_tokens,
        ],
        axis=1,
    )
    categorical_targets = categorical_targets.at[jnp.isnan(categorical_targets)].set(
        dataset.cat_mask_token
    )

    numeric_targets = numeric_targets.at[jnp.isnan(numeric_targets)].set(0.0)

    mi = MTMModelInputs(
        categorical_mask=jax.device_put(categorical_mask, device=device),
        numeric_mask=jax.device_put(numeric_mask, device=device),
        numeric_targets=jax.device_put(numeric_targets, device=device),
        categorical_targets=jax.device_put(categorical_targets, device=device),
    )
    return mi


@struct.dataclass
class TRMModelInputs:
    categorical_inputs: jnp.ndarray
    numeric_inputs: jnp.ndarray
    y: jnp.ndarray = None


def create_trm_model_inputs(
    dataset: TabularDS,
    idx: int = None,
    batch_size: int = None,
    set: str = "train",
    device: jax.Device = None,
):
    if device is None:
        device = jax.devices()[0]
    if set == "train":
        categorical_values = dataset.X_train_categorical
        numeric_values = dataset.X_train_numeric
        y = dataset.y_train
    elif set == "test":
        categorical_values = dataset.X_test_categorical
        numeric_values = dataset.X_test_numeric
        y = dataset.y_test
    else:
        raise ValueError("set must be either 'train' or 'test'")

    if idx is None:
        idx = 0
    if batch_size is None:
        batch_size = numeric_values.shape[0]

    categorical_values = categorical_values[idx : idx + batch_size, :]
    numeric_values = numeric_values[idx : idx + batch_size, :]
    y = y[idx : idx + batch_size, :]

    mi = TRMModelInputs(
        categorical_inputs=jax.device_put(categorical_values, device=device),
        numeric_inputs=jax.device_put(numeric_values, device=device),
        y=jax.device_put(y, device=device),
    )
    return mi


def create_masked_time_series_model_inputs(
    dataset: TabularDS,
    idx: int = None,
    batch_size: int = None,
    set: str = "train",
    device: jax.Device = None,
):
    if device is None:
        device = jax.devices()[0]
    if set == "train":
        categorical_values = dataset.X_train_categorical
        numeric_values = dataset.X_train_numeric
        y = dataset.y_train
    elif set == "test":
        categorical_values = dataset.X_test_categorical
        numeric_values = dataset.X_test_numeric
        y = dataset.y_test
    else:
        raise ValueError("set must be either 'train' or 'test")

    if idx is None:
        idx = 0
    if batch_size is None:
        batch_size = numeric_values.shape[0]

    categorical_values = categorical_values[idx : idx + batch_size, :]
    numeric_values = numeric_values[idx : idx + batch_size, :]
    y = y[idx : idx + batch_size, :]

    mi = TimeSeriesModelInputs(
        categorical_mask=jax.device_put(categorical_values, device=device),
        numeric_mask=jax.device_put(numeric_values, device=device),
        numeric_targets=jax.device_put(y, device=device),
        categorical_targets=jax.device_put(y, device=device),
    )
    return mi


def show_mask_pred(params, model, i, dataset, probability=0.8, set="train"):
    mi = create_mtm_model_inputs(
        dataset, idx=i, batch_size=1, set=set, probability=probability
    )

    logits, numeric_preds = model.apply(
        {"params": params}, mi.categorical_mask, mi.numeric_mask
    )
    cat_preds = logits.argmax(axis=-1)

    # Get the words from the tokens
    decoder_dict = dataset.token_decoder_dict
    cat_preds = [decoder_dict[i.item()] for i in cat_preds[0]]

    results_dict = {k: cat_preds[i] for i, k in enumerate(dataset.col_tokens)}
    for i, k in enumerate(dataset.col_tokens[dataset.n_cat_cols :]):
        results_dict[k] = numeric_preds[0][i].item()
    # Get the masked values
    categorical_masked = [decoder_dict[i.item()] for i in mi.categorical_mask[0]]
    numeric_masked = mi.numeric_mask[0].tolist()
    masked_values = categorical_masked + numeric_masked
    # zip the masked values with the column names
    masked_dict = dict(zip(dataset.col_tokens, masked_values))
    # Get the original values
    categorical_values = [
        decoder_dict[i.item()]
        for i in mi.categorical_targets[0][0 : dataset.n_cat_cols]
    ]

    numeric_values = mi.numeric_targets.tolist()[0]
    original_values = categorical_values
    original_values.extend(numeric_values)
    # zip the original values with the column names
    original_dict = dict(zip(dataset.col_tokens, original_values))

    result_dict = {
        "masked": masked_dict,
        "actual": original_dict,
        "pred": results_dict,
    }

    return result_dict


def mask_tensor(tensor, dataset, probability=0.8, prng_key: ArrayImpl = None):
    if tensor.dtype == "float32" or tensor.dtype == "float64":
        is_numeric = True
    elif tensor.dtype == "int32" or tensor.dtype == "int64":
        is_numeric = False
    else:
        raise ValueError(f"Task {tensor.dtype} not supported.")

    tensor = tensor.copy()
    if prng_key is None:
        seed = int(time.time() * 1000000)
        prng_key = random.PRNGKey(seed)
        Warning(f"Using seed {seed}, consider passing a seed to mask_tensor")
    bit_mask = random.normal(key=prng_key, shape=tensor.shape) > probability
    if is_numeric:
        # tensor = tensor.at[bit_mask].set(float("nan"))
        tensor = jnp.where(bit_mask, jnp.nan, tensor)
    else:
        # tensor = tensor.at[bit_mask].set(dataset.cat_mask_token)
        tensor = jnp.where(bit_mask, dataset.cat_mask_token, tensor)
    return tensor
