from dataclasses import dataclass, field
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import torch


# %%
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
        self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )  # This is where randomness is introduced
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

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2
        )

        X_train_numeric = self.X_train[self.numeric_columns]
        X_train_categorical = self.X_train[self.category_columns]
        X_test_numeric = self.X_test[self.numeric_columns]
        X_test_categorical = self.X_test[self.category_columns]

        self.X_train_numeric = torch.tensor(
            X_train_numeric.values, dtype=torch.float
        ).to(self.device)

        self.X_train_categorical = torch.tensor(
            X_train_categorical.values, dtype=torch.long
        ).to(self.device)

        self.X_test_numeric = torch.tensor(X_test_numeric.values, dtype=torch.float).to(
            self.device
        )

        self.X_test_categorical = torch.tensor(
            X_test_categorical.values, dtype=torch.long
        ).to(self.device)

        self.y_train = torch.tensor(self.y_train.values, dtype=torch.float).to(
            self.device
        )

        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float).to(
            self.device
        )
