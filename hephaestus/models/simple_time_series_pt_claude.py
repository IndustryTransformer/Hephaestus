import math
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from icecream import ic
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def split_complex_word(word):
    parts = re.split(r"(_|\[.*?\])", word)
    parts = [p for p in parts if p]

    def split_camel_case(s):
        if s.startswith("[") and s.endswith("]"):
            return [s]
        return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", s)

    result = [item for part in parts for item in split_camel_case(part)]
    return result


def convert_object_to_int_tokens(df, token_dict):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(token_dict)
    return df


class SimpleDS(Dataset):
    def __init__(self, df, device):
        self.device = device
        self.max_seq_len = df.groupby("idx").count().time_step.max()
        df.idx = df.idx - df.idx.min()
        df = df.set_index("idx")
        df.index.name = None

        self.df_categorical = df.select_dtypes(include=["object"]).astype(str)
        self.df_numeric = df.select_dtypes(include="number")
        self.batch_size = self.max_seq_len
        self.numeric_token = "[NUMERIC_EMBEDDING]"
        self.special_tokens = [
            "[PAD]",
            "[NUMERIC_MASK]",
            "[MASK]",
            "[UNK]",
            self.numeric_token,
        ]
        self.numeric_mask = "[NUMERIC_MASK]"
        self.numeric_col_tokens = [
            col_name for col_name in df.select_dtypes(include="number").columns
        ]
        self.categorical_col_tokens = [
            col_name for col_name in df.select_dtypes(include="object").columns
        ]
        self.object_tokens = (
            self.df_categorical.apply(pd.Series.unique).values.flatten().tolist()
        )
        self.tokens = (
            self.special_tokens
            + self.numeric_col_tokens
            + self.object_tokens
            + self.categorical_col_tokens
        )

        self.token_dict = {token: i for i, token in enumerate(self.tokens)}
        self.token_decoder_dict = {i: token for i, token in enumerate(self.tokens)}
        self.n_tokens = len(self.tokens)
        self.numeric_indices = torch.tensor(
            [self.tokens.index(i) for i in self.numeric_col_tokens], device=self.device
        )
        self.categorical_indices = torch.tensor(
            [self.tokens.index(i) for i in self.categorical_col_tokens],
            device=self.device,
        )

        self.numeric_mask_token = self.tokens.index(self.numeric_mask)
        reservoir_vocab = [split_complex_word(word) for word in self.token_dict.keys()]
        self.reservoir_vocab = list(
            set([item for sublist in reservoir_vocab for item in sublist])
        )
        reservoir_tokens_list = [
            self.token_decoder_dict[i] for i in range(len(self.token_decoder_dict))
        ]
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.reservoir_encoded = self.tokenizer(
            reservoir_tokens_list,
            padding="max_length",
            max_length=8,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)

        self.df_categorical = convert_object_to_int_tokens(
            self.df_categorical, self.token_dict
        )

    def __len__(self):
        return self.df_numeric.index.nunique()

    def get_data(self, df_name, set_idx):
        df = getattr(self, df_name)
        batch = df.loc[df.index == set_idx, :]
        batch = np.array(batch.values).astype(np.float32)

        batch_len, n_cols = batch.shape
        pad_len = self.max_seq_len - batch_len
        padding = np.full((pad_len, n_cols), np.nan)
        batch = np.concatenate([batch, padding], axis=0)
        batch = np.swapaxes(batch, 0, 1)
        batch = batch.astype(np.float32)
        return torch.tensor(batch, device=self.device, dtype=torch.float32)

    def __getitem__(self, set_idx):
        if self.df_categorical.empty:
            categorical_inputs = None
        else:
            categorical_inputs = self.get_data("df_categorical", set_idx)
        numeric_inputs = self.get_data("df_numeric", set_idx)

        return numeric_inputs, categorical_inputs


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self, dataset, d_model=64, n_heads=4, time_window=10_000, device="cpu"
    ):
        super().__init__()
        self.dataset = dataset
        self.d_model = d_model
        self.n_heads = n_heads
        self.time_window = time_window
        self.device = device
        self.embedding = ReservoirEmbedding(
            self.dataset, features=self.d_model, device=self.device
        )
        self.pos_encoding = PositionalEncoding(
            max_len=self.time_window, d_pos_encoding=self.d_model
        )
        self.transformer_block = TransformerBlock(
            num_heads=self.n_heads, d_model=self.d_model, d_ff=64, dropout_rate=0.1
        )
        self.to(self.device)

    def forward(
        self,
        numeric_inputs: torch.Tensor,
        categorical_inputs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        mask_data: bool = True,
    ):
        numeric_inputs = numeric_inputs.to(self.device)
        nan_mask = torch.isnan(numeric_inputs)
        numeric_inputs = torch.where(
            nan_mask, torch.zeros_like(numeric_inputs), numeric_inputs
        )

        if categorical_inputs is not None:
            categorical_inputs = categorical_inputs.to(self.device)
            categorical_inputs = torch.where(
                torch.isnan(categorical_inputs),
                torch.tensor(
                    self.dataset.token_dict["[NUMERIC_MASK]"],
                    device=self.device,
                    dtype=categorical_inputs.dtype,
                ),
                categorical_inputs,
            )
            categorical_embeddings = self.embedding(categorical_inputs.long())
        else:
            categorical_embeddings = None

        repeated_numeric_indices = self.dataset.numeric_indices.repeat(
            numeric_inputs.shape[2], 1
        ).T
        numeric_col_embeddings = self.embedding(repeated_numeric_indices)
        numeric_col_embeddings = numeric_col_embeddings.unsqueeze(0).repeat(
            numeric_inputs.shape[0], 1, 1, 1
        )

        if categorical_embeddings is not None:
            repeated_categorical_indices = self.dataset.categorical_indices.repeat(
                categorical_inputs.shape[2], 1
            ).T
            categorical_col_embeddings = self.embedding(repeated_categorical_indices)
            categorical_col_embeddings = categorical_col_embeddings.unsqueeze(0).repeat(
                categorical_inputs.shape[0], 1, 1, 1
            )
        else:
            categorical_col_embeddings = None

        numeric_embedding = self.embedding(
            torch.tensor(
                self.dataset.token_dict[self.dataset.numeric_token], device=self.device
            )
        )
        numeric_broadcast = numeric_inputs.unsqueeze(-1) * numeric_embedding

        numeric_broadcast = torch.where(
            nan_mask.unsqueeze(-1),
            self.embedding(
                torch.tensor(self.dataset.numeric_mask_token, device=self.device)
            ),
            numeric_broadcast,
        )

        numeric_broadcast = self.pos_encoding(numeric_broadcast)

        if categorical_embeddings is not None:
            tabular_data = torch.cat([numeric_broadcast, categorical_embeddings], dim=1)
        else:
            tabular_data = numeric_broadcast

        if categorical_embeddings is not None:
            mask_input = torch.cat([numeric_inputs, categorical_inputs], dim=1)
        else:
            mask_input = numeric_inputs

        if mask_data:
            mask = self.generate_mask(mask_input)
        else:
            mask = None

        if categorical_embeddings is not None:
            col_embeddings = torch.cat(
                [numeric_col_embeddings, categorical_col_embeddings], dim=1
            )
        else:
            col_embeddings = numeric_col_embeddings

        out = self.transformer_block(
            tabular_data, col_embeddings, tabular_data, mask=mask
        )
        return out

    def generate_mask(self, mask_input):
        seq_len = mask_input.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device), diagonal=1
        ).bool()
        pad_mask = mask_input.isnan().any(dim=-1).unsqueeze(1).repeat(1, seq_len, 1)
        mask = causal_mask | pad_mask
        return mask


class ReservoirEmbedding(nn.Module):
    def __init__(self, dataset, features, device, frozen_index=0):
        super().__init__()
        self.dataset = dataset
        self.features = features
        self.frozen_index = frozen_index
        self.device = device
        self.embedding = nn.Embedding(self.dataset.tokenizer.vocab_size, self.features)
        self.to(self.device)

    def forward(self, base_indices):
        base_indices = base_indices.to(self.device)
        embedding_weight = self.embedding.weight.clone()
        embedding_weight.data[self.frozen_index] = 0
        penultimate_embedding = embedding_weight.detach() + (
            embedding_weight - embedding_weight.detach()
        ) * (
            torch.arange(self.dataset.tokenizer.vocab_size, device=self.device)
            != self.frozen_index
        ).unsqueeze(
            1
        )

        token_reservoir_lookup = self.dataset.reservoir_encoded.to(self.device)
        reservoir_indices = token_reservoir_lookup[base_indices]

        ultimate_embedding = penultimate_embedding[reservoir_indices]
        ultimate_embedding = torch.sum(ultimate_embedding, dim=-2)

        return ultimate_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_pos_encoding):
        super().__init__()
        self.encoding = self._get_positional_encoding(max_len, d_pos_encoding)

    def _get_positional_encoding(self, max_len, d_pos_encoding):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_pos_encoding, 2) * -(math.log(10000.0) / d_pos_encoding)
        )
        encoding = torch.zeros(max_len, d_pos_encoding)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)

    def forward(self, x):
        ic(x.shape, self.encoding.shape)
        return x + self.encoding[:, : x.size(1)].to(x.device)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.norm1(q), self.norm1(k), self.norm1(v)
        attention_output, _ = self.attention(q, k, v, attn_mask=mask)
        out = q + attention_output
        out = self.norm2(out)
        ffn_output = self.ffn(out)
        out = out + ffn_output
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        print("What the hell is going on?")
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class SimplePred(nn.Module):
    def __init__(self, dataset: SimpleDS, d_model: int, n_heads: int):
        super().__init__()
        self.dataset = dataset
        self.d_model = d_model
        self.n_heads = n_heads
        self.time_series_transformer = TimeSeriesTransformer(
            self.dataset, self.d_model, self.n_heads
        )

    def forward(
        self, numeric_inputs, categorical_inputs, deterministic=False, mask_data=True
    ):
        out = self.time_series_transformer(
            numeric_inputs=numeric_inputs,
            categorical_inputs=categorical_inputs,
            deterministic=deterministic,
        )
        ic(out.shape)
        ic(f"Nan values in simplePred out 1: {torch.isnan(out).any()}")
        numeric_out = out.swapaxes(1, 2)
        numeric_out = numeric_out.reshape(
            numeric_out.shape[0], numeric_out.shape[1], -1
        )
