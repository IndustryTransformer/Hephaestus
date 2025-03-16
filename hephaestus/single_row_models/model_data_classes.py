from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# %%
# @dataclass
# class TabularDS:
#     df: pd.DataFrame = field(repr=False)
#     target_column: str
#     seed: int = 42
#     device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # special_tokens: list =
#     special_tokens: list = field(
#         default_factory=lambda: ["[PAD]", "[NUMERIC_MASK]", "[MASK]"]
#     )

#     def __post_init__(self):
#         self.category_columns = self.df.select_dtypes(
#             include=["object"]
#         ).columns.tolist()
#         self.numeric_columns = self.df.select_dtypes(
#             include=["int64", "float64"]
#         ).columns.tolist()
#         self.target_column = [self.target_column]
#         self.tokens = list(
#             chain(
#                 self.special_tokens,
#                 self.df.columns.to_list(),
#                 list(set(self.df[self.category_columns].values.flatten().tolist())),
#             )
#         )

#         self.token_dict = {token: i for i, token in enumerate(self.tokens)}
#         self.token_decoder_dict = {i: token for i, token in enumerate(self.tokens)}

#         self.scaler = StandardScaler()
#         self.numeric_columns.remove(self.target_column[0])
#         # self.numeric_columns = self.numeric_columns.remove(self.target_column[0])
#         numeric_scaled = self.scaler.fit_transform(self.df[self.numeric_columns])
#         self.df[self.numeric_columns] = numeric_scaled
#         for col in self.category_columns:
#             self.df[col] = self.df[col].map(self.token_dict)


@dataclass
class TabularDS:
    df: pd.DataFrame = field(repr=False)
    target_column: str
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # special_tokens: list =
    special_tokens: list = field(
        default_factory=lambda: ["[PAD]", "[NUMERIC_MASK]", "[MASK]"]
    )

    def __post_init__(self):
        self.category_columns = self.df.select_dtypes(
            include=["object"]
        ).columns.tolist()
        self.numeric_columns = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.target_column = [self.target_column]
        self.tokens = list(
            chain(
                self.special_tokens,
                self.df.columns.to_list(),
                list(set(self.df[self.category_columns].values.flatten().tolist())),
            )
        )

        self.token_dict = {token: i for i, token in enumerate(self.tokens)}
        self.token_decoder_dict = {i: token for i, token in enumerate(self.tokens)}

        # self.scaler = StandardScaler()
        # self.numeric_columns.remove(self.target_column[0])
        print(self.numeric_columns, "numeric_columns1")

        self.numeric_columns.remove(self.target_column[0])
        print(self.numeric_columns, "numeric_columns2")
        # numeric_scaled = self.scaler.fit_transform(self.df[self.numeric_columns])
        # self.df[self.numeric_columns] = numeric_scaled
        # for col in self.category_columns:
        #     self.df[col] = self.df[col].map(self.token_dict)


class TabularDataset(Dataset):
    def __init__(self, config: TabularDS, df: pd.DataFrame, mode: str):
        self.config = config

        self.df = df
        self.scaler = StandardScaler()

        self._create_train_test()
        if mode == "train":
            self.X_numeric = self.X_train_numeric
            self.X_categorical = self.X_train_categorical
            self.y = self.y_train
        else:
            self.X_numeric = self.X_test_numeric
            self.X_categorical = self.X_test_categorical
            self.y = self.y_test

    def _create_train_test(self):
        X = self.df.drop(self.config.target_column, axis=1)
        y = self.df[self.config.target_column]
        X[self.config.numeric_columns] = self.scaler.fit_transform(
            X[self.config.numeric_columns]
        )
        for col in self.config.category_columns:
            X[col] = X[col].map(self.config.token_dict)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(X, y, test_size=0.001)

        X_train_numeric = self.X_train[self.config.numeric_columns]
        X_train_categorical = self.X_train[self.config.category_columns]
        X_test_numeric = self.X_test[self.config.numeric_columns]
        X_test_categorical = self.X_test[self.config.category_columns]

        self.X_train_numeric = np.array(X_train_numeric.values, dtype=np.float32)

        self.X_train_categorical = np.array(X_train_categorical.values, dtype=np.long)

        self.X_test_numeric = np.array(X_test_numeric.values, dtype=np.float32)

        self.X_test_categorical = np.array(X_test_categorical.values, dtype=np.long)

        self.y_train = np.array(self.y_train.values, dtype=np.float32)

        self.y_test = np.array(self.y_test.values, dtype=np.float32)

    def __len__(self):
        return self.X_numeric.shape[0]

    def __getitem__(self, idx):
        return {
            "X_numeric": torch.tensor(self.X_numeric[idx]),
            "X_categorical": torch.tensor(self.X_categorical[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }
