# %%
import math
import re
from dataclasses import dataclass, field
from numbers import Number
from typing import List, Union  # Any, Callable, Dict, List, Optional, Tuple,

import numpy as np
import polars as pl
import torch

# import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset  # DataLoader, dataset

# from tqdm import tqdm, trange

# %%


def scale_numeric(df):
    for col in df.columns:
        if df[col].dtype == pl.Float64 or df[col].dtype == pl.Int64:
            df = df.with_columns(
                ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
            )  # .select(pl.col(["dew_point_temp", "NewCOL"]))
    return df


def make_lower_remove_special_chars(df):
    df = df.with_columns(
        pl.col(pl.Utf8).str.to_lowercase().str.replace_all("[^a-zA-Z0-9]", " ")
    )
    return df


def get_unique_utf8_values(df):
    arr = np.array([])
    for col in df.select(pl.col(pl.Utf8)).columns:
        arr = np.append(arr, df[col].unique().to_numpy())

    return np.unique(arr)


def get_col_tokens(df):
    tokens = []
    for col_name in df.columns:
        sub_strs = re.split(r"[^a-zA-Z0-9]", col_name)
        tokens.extend(sub_strs)
    return np.unique(np.array(tokens))


@dataclass
class StringNumeric:
    value: Union[str, float]
    # all_tokens: np.array
    is_numeric: bool = field(default=None, repr=True)
    embedding_idx: int = field(default=None, repr=True)
    is_special: bool = field(default=False, repr=True)

    def __post_init__(self):
        if isinstance(self.value, str):
            self.is_numeric = False
        else:
            self.is_numeric = True
            self.embedding_idx = 0

    def gen_embed_idx(self, tokens: np.array, special_tokens: np.array):
        if not self.is_numeric:
            try:
                self.embedding_idx = np.where(tokens == self.value)[0][0] + 1
            except IndexError:
                self.embedding_idx = np.where(tokens == "<unk>")[0][0] + 1
            if self.value in special_tokens:
                self.is_special = True


class TabularDataset(Dataset):
    # def __init__(self, df: pl.DataFrame, vocab_dict: Dict, m_dim: int) -> Dataset:
    def __init__(
        self,
        df: pl.DataFrame,
        vocab,
        special_tokens: np.array,
        shuffle_cols=False,
        max_row_length=512,
    ) -> Dataset:
        self.df = df
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.vocab_len = vocab.shape[0]
        self.shuffle_cols = shuffle_cols
        self.max_row_length = max_row_length
        # self.vocab_dict = vocab_dict
        # self.embedding = nn.Embedding(len(self.string_vocab), m_dim)
        # Numeric Scale

        # self.col_vocab = self.df.columns

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        length = self.df.shape[0]
        return length

    def __getitem__(self, idx):
        """Returns a tuple of (input, target) at the given index."""
        row = self.df[idx]
        row = self.splitter(row)
        return row

    def splitter(self, row: pl.DataFrame) -> List[Union[str, float, None]]:
        vals = ["<row-start>"]
        cols = row.columns
        if self.shuffle_cols:
            np.random.shuffle(cols)

        for col in cols:
            value = row[col][0]
            col = col.split("_")
            vals.extend(col)
            vals.append(":")
            if isinstance(value, Number):
                vals.append(value)
            elif value is None:
                vals.append("missing")
                # Nones are only for numeric columns, others are "None"
            elif isinstance(value, str):
                vals.extend(value.split(" "))
            else:
                raise ValueError("Unknown type")
            vals.append(",")
        vals.append("<row-end>")

        val_len = len(vals)
        if val_len < self.max_row_length:
            diff = self.max_row_length - val_len
            vals.extend(["<pad>"] * diff)
        elif val_len > self.max_row_length:
            vals = vals[: self.max_row_length - 1]
            # add warning

            vals = np.append(vals, ["<row-end>"])
            print("Row too long, truncating")
            Warning("Row too long, truncating")
        vals = [StringNumeric(value=val) for val in vals]
        for val in vals:
            val.gen_embed_idx(self.vocab, self.special_tokens)

        return vals


class StringNumericEmbedding(nn.Module):
    def __init__(self, n_token: int, d_model: int, device: torch.device):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(n_token + 1, d_model).to(device)  # padding_idx=0
        self.numeric_embedding = nn.Linear(1, d_model).to(device)

    def forward(self, input: StringNumeric):
        embedding_tensor = torch.zeros(
            (len(input), self.embedding.embedding_dim), dtype=torch.float32
        ).to(self.device)
        for idx, val in enumerate(input):
            if val.is_numeric:
                val = torch.Tensor([val.value]).float().to(self.device)
                embedding_tensor[idx] = self.numeric_embedding(val)
            else:
                embed_idx = torch.Tensor([val.embedding_idx]).long().to(self.device)
                embedding_tensor[idx] = self.embedding(embed_idx)

        return embedding_tensor


def mask_row(row, tokens, special_tokens):
    row = row[:]
    prob = 0.15
    for idx, val in enumerate(row):
        if val.is_special:
            continue
        if np.random.rand() < prob:
            if val.is_numeric:
                val = StringNumeric(value="<numeric_mask>")
                val.gen_embed_idx(tokens, special_tokens)
                row[idx] = val
            else:
                val = StringNumeric(value="<mask>")
                val.gen_embed_idx(tokens, special_tokens)
                row[idx] = val
    return row


def batch_data(ds, idx: int, n_row=4):
    target = []
    if len(ds) > n_row + idx:
        end_idx = n_row + idx
    else:
        end_idx = len(ds) - 1
    for i in range(idx, end_idx):
        target.extend(ds[i])

    batch = mask_row(target, ds.vocab, ds.special_tokens)

    return batch, target


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_token: int,
        d_model: int,
        n_head: int,
        d_hid: int,
        n_layers: int,
        device: torch.device,
        dropout: float = 0.15,
    ):
        super().__init__()
        n_token = n_token + 1
        self.n_token = n_token
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = StringNumericEmbedding(n_token, d_model, device)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_token)
        self.numeric_ingester = nn.Linear(d_model, n_token * 2)
        self.numeric_hidden = nn.Linear(n_token * 2, n_token)
        self.numeric_flattener = nn.Linear(n_token, 1)

        # self.numeric_decoder = nn.Linear(d_model)

        self.init_weights()

    def init_weights(self) -> None:
        init_range = 0.1
        self.encoder.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, n_token]``
        """
        # src_shape = src.shape
        # print(f"raw src_shape: {len(src)}")
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = torch.unsqueeze(src, dim=1)
        # print(f"encoded src_shape: {src.shape}")

        # src_shape = src.shape
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # print(f"output_shape: {output.shape}")
        numeric_output = torch.relu(self.numeric_ingester(output))  # .flatten()
        numeric_output = torch.relu(self.numeric_hidden(numeric_output))
        numeric_output = torch.squeeze(numeric_output, dim=1)
        # numeric_output = torch.mean(numeric_output, [1])
        numeric_output = self.numeric_flattener(numeric_output)
        # numeric_output = nn.flatten(numeric_output)
        output = self.decoder(output)
        output = torch.squeeze(output, dim=1)
        numeric_output = numeric_output.view(output.shape[0])

        # print(f"output_shape decoded: {output.shape}")
        # output = output.view(-1, self.n_token+1)
        # output = output.view(-1, src_shape[0]).T
        # print(f"output_shape view: {output.shape}")

        return output, numeric_output


def hephaestus_loss(
    class_preds, numeric_preds, raw_data, tokens, special_tokens, device
):
    cross_entropy = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    raw_data_numeric_class = raw_data[:]

    for idx, val in enumerate(raw_data_numeric_class):
        if val.is_numeric:
            val = StringNumeric(value="<numeric>")
            val.gen_embed_idx(tokens, special_tokens)
            raw_data_numeric_class[idx] = val

    class_target = torch.tensor([i.embedding_idx for i in raw_data_numeric_class]).to(
        device
    )
    class_loss = cross_entropy(class_preds, class_target)

    actual_num_idx = torch.tensor(
        [idx for idx, j in enumerate(raw_data) if j.is_numeric]
    ).to(device)
    pred_nums = numeric_preds[actual_num_idx]
    # print(actual_num_idx.shape)
    actual_nums = torch.tensor([i.value for i in raw_data if i.is_numeric]).to(device)
    # print(actual_nums.shape)
    # print(pred_nums.shape)
    reg_loss = mse_loss(pred_nums, actual_nums)
    reg_loss_adjuster = 6  # class_loss/reg_loss

    return reg_loss * reg_loss_adjuster + class_loss, {  # , class_loss
        "reg_loss": reg_loss,
        "class_loss": class_loss,  # class_loss,
    }
