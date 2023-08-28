# %%
import re
from dataclasses import dataclass, field
from numbers import Number
from typing import List, Union  # Any, Callable, Dict, List, Optional, Tuple,

import numpy as np
import polars as pl
import torch
from torch import Tensor, nn

# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset  # DataLoader, dataset

from transformers import BertForPreTraining, BertTokenizer

# import torch.nn.functional as F

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
    # is_special: bool = field(default=False, repr=True)

    def __post_init__(self):
        if isinstance(self.value, str):
            self.is_numeric = False
        else:
            self.is_numeric = True


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

        # val_len = len(vals)
        # if val_len < self.max_row_length:
        #     diff = self.max_row_length - val_len
        #     vals.extend(["<pad>"] * diff)
        # elif val_len > self.max_row_length:
        #     vals = vals[: self.max_row_length - 1]
        #     # add warning

        #     vals = np.append(vals, ["<row-end>"])
        #     print("Row too long, truncating")
        #     Warning("Row too long, truncating")
        vals = [StringNumeric(value=val) for val in vals]
        # for val in vals:
        #     val.gen_embed_idx(self.vocab, self.special_tokens)

        return vals


class StringNumericEmbedding(nn.Module):
    def __init__(self, state_dict, device: torch.device, tokenizer):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.word_embeddings = nn.Embedding(*state_dict["weight"].shape).to(device)
        self.word_embeddings.load_state_dict(state_dict)  # .to(device)
        self.numeric_embedding = nn.Sequential(
            nn.Linear(1, 128),  # First hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),  # Second hidden layer
            nn.ReLU(),
            nn.Linear(64, state_dict["weight"].shape[1]),  # Output layer
        ).to(device)

        # self.numeric_embedding = nn.Linear(1, d_model).to(device)

    def forward(self, input: StringNumeric):
        start_token = self.tokenizer.encode(
            "[CLS]", add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        index_list = [start_token.item()]
        tensor_list = [
            self.word_embeddings(start_token).to(self.device).reshape(1, 1, -1)
        ]  # Start token
        for val in input:
            if val.is_numeric:
                index_list.extend(
                    self.tokenizer.encode("[NUMERIC]", add_special_tokens=False)
                )
                val = Tensor([val.value]).float().to(self.device)
                val = self.numeric_embedding(val)
                val = val.reshape(1, 1, -1)  # val.shape[0])
                tensor_list.append(val)

            else:
                tokens_ids = self.tokenizer.encode(
                    val.value, return_tensors="pt", add_special_tokens=False
                )
                index_list.extend(tokens_ids.tolist())
                tensor_list.append(self.word_embeddings(tokens_ids.to(self.device)))
        # end_token = self.tokenizer.encode(
        #     "[SEP]", add_special_tokens=False, return_tensors="pt"
        # ).to(self.device)
        # tensor_list.append(
        #     self.word_embeddings(end_token).to(self.device).reshape(1, 1, -1)
        # )
        # End token
        tensor_list = torch.cat(tensor_list, dim=-2)
        # print(index_list)
        # print(tensor_list.shape, len(index_list))
        return tensor_list


def mask_row(row, model, prob=0.2):
    # row = row[:]
    return_row = []
    is_numeric_mask = []
    for idx, val in enumerate(row):
        if np.random.rand() < prob:
            if val.is_numeric:
                val = StringNumeric(value="[NUMERIC]")
                # val.gen_embed_idx(tokens, special_tokens)
                return_row.append(val)
                is_numeric_mask.append(True)
            else:
                # calculate the number of tokens to mask and then mask ALL of them
                n_tokens = model.tokenizer.encode(val.value, add_special_tokens=False)
                n_tokens = len(n_tokens)
                vals = [StringNumeric(value="[MASK]") for _ in range(n_tokens)]
                # val.gen_embed_idx(tokens, special_tokens)
                return_row.extend(vals)
                is_numeric_mask.extend([False for _ in range(n_tokens)])
        else:
            # if not val.is_numeric:
            # n_tokens = model.tokenizer.encode(val.value, add_special_tokens=False)
            return_row.append(val)
            is_numeric_mask.append(val.is_numeric)

    return return_row, is_numeric_mask


def batch_data(
    ds,
    idx: int,
    model,
    n_row=4,
):
    target = []
    if len(ds) > n_row + idx:
        end_idx = n_row + idx
    else:
        end_idx = len(ds) - 1
    for i in range(idx, end_idx):
        target.extend(ds[i])

    batch, is_numeric_mask = mask_row(target, model)
    # Get a true_false mask for the numeric values
    # is_numeric_mask = [i.is_numeric for i in target]
    return batch, target, is_numeric_mask


class TransformerModel(nn.Module):
    def __init__(
        self,
        device: torch.device,
        bert_model_name="bert-base-uncased",
    ):
        super(TransformerModel, self).__init__()

        # BERT Tokenizer and Model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_lm = BertForPreTraining.from_pretrained("bert-base-uncased")
        # self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.tokenizer.add_tokens(
            ["[MISSING]", "[NUMERIC]", "<row-start>", "<row-end>", "<pad>"]
        )
        self.device = device
        # Add tokens to BERT model

        # self.bert = BertModel.from_pretrained(bert_model_name).to(device)
        self.bert_lm.resize_token_embeddings(len(self.tokenizer))
        self.bert_embedding_state_dict = (
            self.bert_lm.bert.embeddings.word_embeddings.state_dict()
        )
        self.embedding_dim = self.bert_lm.bert.config.hidden_size
        self.string_numeric_embd = StringNumericEmbedding(
            state_dict=self.bert_embedding_state_dict,
            device=device,
            tokenizer=self.tokenizer,
        )
        # self.decoder = nn.Linear(self.embedding_dim, len(self.tokenizer)).to(device)
        # Numeric Neural Net for numbers prediction after BERT
        self.numeric_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, input: StringNumeric, finetune=False):
        input = self.string_numeric_embd(input)
        bert_output = self.bert_lm.bert(inputs_embeds=input)
        last_hidden_state = bert_output.last_hidden_state
        pooled_output = bert_output.pooler_output
        
        bert_logits = self.bert_lm.cls(last_hidden_state, pooled_output)[0]
        numeric_prediction = self.numeric_predictor(last_hidden_state)
        if finetune:
            print("Fine Tune Enabled")
            return last_hidden_state, numeric_prediction
        else:
            return bert_logits, numeric_prediction 


def gen_class_target_tokens(model, input):
    tokenizer = model.tokenizer
    target_tokens = [
        tokenizer.encode(
            "[CLS]", add_special_tokens=False, return_tensors="pt"
        ).squeeze(0)
    ]
    for val in input:
        if val.is_numeric:
            encoded_token = tokenizer.encode(
                "[NUMERIC]", add_special_tokens=False, return_tensors="pt"
            )
        else:
            encoded_token = tokenizer.encode(
                val.value, add_special_tokens=False, return_tensors="pt"
            )
        target_tokens.append(encoded_token.squeeze(0))

    # target_tokens.append(
    #     tokenizer.encode(
    #         "[SEP]", add_special_tokens=False, return_tensors="pt"
    #     ).squeeze(0)
    # )
    target_tokens = torch.cat(target_tokens, dim=-1).to(model.device)
    # if target_tokens.ndim == 1:
    #     target_tokens = target_tokens.unsqueeze(0)
    return target_tokens
    # return target_tokens


def hephaestus_loss(class_preds, numeric_preds, target, is_numeric_mask, model):
    cross_entropy = nn.CrossEntropyLoss()
    rrmse_loss = nn.RMSELoss()
    device = model.device
    # raw_data_numeric_class = raw_data[:]

    class_target = gen_class_target_tokens(model, target)
    # print(f"Class Target Shape: {class_target.shape},
    # Class Preds: {class_preds.shape}")
    class_loss = cross_entropy(class_preds[0], class_target)
    actual_num_index = [idx for idx, val in enumerate(is_numeric_mask) if val]
    actual_num_index = torch.tensor(actual_num_index).to(device)
    # actual_num_idx = torch.tensor(
    #     [idx for idx, j in enumerate(raw_data) if j.is_numeric]
    # ).to(device)
    numeric_preds = numeric_preds.squeeze()
    numeric_preds = numeric_preds[actual_num_index]
    # print(actual_num_idx.shape)
    actual_nums = torch.tensor([i.value for i in target if i.is_numeric]).to(device)
    # print(
    #     f"Pred Num Shape: {numeric_preds.shape},
    # Actual Num Shape: {actual_nums.shape}"
    # )
    # print(actual_nums.shape)
    # print(pred_nums.shape)
    reg_loss = rmse_loss(numeric_preds, actual_nums)
    reg_loss_adjuster = 1  # / 10  # class_loss/reg_loss
    # Scale the regression loss to be on the same scale as the classification loss
    # reg_loss_adjuster = class_loss / reg_loss

    return reg_loss * reg_loss_adjuster + class_loss, {  # , class_loss
        "reg_loss": reg_loss,
        "class_loss": class_loss,  # class_loss,
    }


def evaluate_custom(model: nn.Module, ds, idx, tokens, device) -> None:
    model.eval()  # turn on train mode
    n_row = 1  # one because it's not time series
    with torch.no_grad():
        data, targets = batch_data(ds, idx, n_row=n_row)
        class_preds, numeric_preds = model(data)
        loss, loss_dict = hephaestus_loss(
            class_preds, numeric_preds, targets, tokens, device
        )
        return {
            "loss": loss.item(),
            "loss_dict": loss_dict,
            "data": data,
            "targets": targets,
            "class_preds": class_preds,
            "numeric_preds": numeric_preds,
        }


def evaluate_bert(model, ds, i):
    def replacer(s):
        return (
            s.replace(" ##", "")
            .replace(" ,", ",")
            .replace(" :", ":")
            .replace(" .", ".")
        )

    model.eval()

    masked_data, target_data, _ = batch_data(ds, i, model, n_row=1)
    with torch.no_grad():
        class_preds, numeric_preds = model(masked_data)
    class_preds = class_preds.squeeze()
    numeric_preds = numeric_preds.squeeze()
    # target_tensor = gen_class_target_tokens(model, target_data)
    actuals = replacer(" ".join([str(val.value) for val in target_data]))
    class_preds_max = torch.argmax(class_preds.squeeze(), dim=1)
    preds = []
    for idx, pred in enumerate(class_preds_max):
        decoded = model.tokenizer.decode([pred], skip_special_tokens=True)
        if decoded == "[numeric]":
            preds.append(str(numeric_preds[idx].item()))
        else:
            preds.append(decoded)
    masked = replacer(" ".join([str(val.value) for val in masked_data])).strip()
    preds = replacer(" ".join(preds)).strip()

    return {"masked": masked, "actuals": actuals, "predictions": preds}


def show_results(res, tokens):
    actuals = [str(i.value) for i in res["targets"]]
    actuals_ = " ".join(actuals)
    actual_str = actuals_.split("<row-end>")[0]
    masked_str = [str(i.value) for i in res["data"]]
    masked_str = " ".join(masked_str)
    masked_str = masked_str.split("<row-end>")[0]

    lsm = nn.Softmax(dim=0)
    softmax_cats = lsm(res["class_preds"])
    softmax_cats = torch.argmax(softmax_cats, dim=1)
    gen_tokens = []
    for idx, pred in enumerate(softmax_cats):
        token = tokens[pred - 1]
        if token == "<numeric>":
            gen_tokens.append(str(res["numeric_preds"][idx].item()))
        else:
            gen_tokens.append(token)
    preds = " ".join(gen_tokens)

    s = (
        f"Targets   : {actual_str}\n"
        + f"Masked    : {masked_str}\n"
        + f"Predicted : {preds.split('<row-end>')[0]}"
    )
    return s
