from dataclasses import dataclass, field
from itertools import chain

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

        self.scaler = StandardScaler()
        self.numeric_columns.remove(self.target_column[0])
        # self.numeric_columns = self.numeric_columns.remove(self.target_column[0])
        numeric_scaled = self.scaler.fit_transform(self.df[self.numeric_columns])
        self.df[self.numeric_columns] = numeric_scaled
        for col in self.category_columns:
            self.df[col] = self.df[col].map(self.token_dict)


class TabularDataset(Dataset):
    def __init__(self, config: TabularDS, mode: str):
        self.config = config
        # if mode == "train":
        #     self.X_numeric = config.X_train_numeric
        #     self.X_categorical = config.X_train_categorical
        #     self.y = config.y_train
        # else:
        #     self.X_numeric = config.X_test_numeric
        #     self.X_categorical = config.X_test_categorical
        #     self.y = config.y_test
        self.df = config.df
        self._create_train_test()
        if mode == "train":
            self.X_numeric = self.config.X_train_numeric
            self.X_categorical = self.config.X_train_categorical
            self.y = self.config.y_train
        else:
            self.X_numeric = self.config.X_test_numeric
            self.X_categorical = self.config.X_test_categorical
            self.y = self.config.y_test

    #     self.y = df[config.target_column].values
    #     df = df.drop(columns=config.target_column)
    #     X_numeric = df[config.numeric_columns].values
    #     self.X_numeric = torch.tensor(X_numeric, dtype=torch.float32)

    #     for col in config.category_columns:
    #         df[col] = df[col].map(config.token_dict)
    #     X_categorical = df[config.category_columns].values

    #     self.X_categorical = torch.tensor(X_categorical, dtype=torch.long)
    def _create_train_test(self):
        X = self.config.df.drop(self.config.target_column, axis=1)
        y = self.config.df[self.config.target_column]

        (
            self.config.X_train,
            self.config.X_test,
            self.config.y_train,
            self.config.y_test,
        ) = train_test_split(X, y, test_size=0.2)

        X_train_numeric = self.config.X_train[self.config.numeric_columns]
        X_train_categorical = self.config.X_train[self.config.category_columns]
        X_test_numeric = self.config.X_test[self.config.numeric_columns]
        X_test_categorical = self.config.X_test[self.config.category_columns]

        self.config.X_train_numeric = torch.tensor(
            X_train_numeric.values, dtype=torch.float
        )

        self.config.X_train_categorical = torch.tensor(
            X_train_categorical.values, dtype=torch.long
        )

        self.config.X_test_numeric = torch.tensor(
            X_test_numeric.values, dtype=torch.float
        )

        self.config.X_test_categorical = torch.tensor(
            X_test_categorical.values, dtype=torch.long
        )

        self.config.y_train = torch.tensor(
            self.config.y_train.values, dtype=torch.float
        )

        self.config.y_test = torch.tensor(self.config.y_test.values, dtype=torch.float)

    def __len__(self):
        return self.X_numeric.shape[0]

    def __getitem__(self, idx):
        return {
            "X_numeric": torch.tensor(self.X_numeric[idx]),
            "X_categorical": torch.tensor(self.X_categorical[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }
