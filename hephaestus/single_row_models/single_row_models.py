import torch
import torch.nn as nn
import torch.nn.functional as F

from hephaestus.single_row_models.single_row_utils import (
    initialize_parameters,
)
from hephaestus.utils import NumericCategoricalData


# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # self.initialize_parameters()
        self.apply(initialize_parameters)

    def forward(self, q, k, v, mask=None, input_feed_forward=False):
        batch_size = q.size(0)

        if input_feed_forward:
            q = (
                self.q_linear(q)
                .view(batch_size, -1, self.n_heads, self.d_head)
                .transpose(1, 2)
            )
            k = (
                self.k_linear(k)
                .view(batch_size, -1, self.n_heads, self.d_head)
                .transpose(1, 2)
            )
            v = (
                self.v_linear(v)
                .view(batch_size, -1, self.n_heads, self.d_head)
                .transpose(1, 2)
            )

        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        out = self.out_linear(attn_output)
        return out

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        d_k = q.size(-1)
        scaled_attention_logits = matmul_qk / (d_k**0.5)

        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerEncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
            # nn.Linear(4 * d_model, d_model * 4),
            # nn.ReLU(),
            # nn.Linear(d_model * 4, d_model),
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        # self.initialize_parameters()
        self.apply(initialize_parameters)

    def forward(self, q, k, v, mask=None, input_feed_forward=False):
        attn_output = self.multi_head_attention(q, k, v, mask, input_feed_forward)
        out1 = self.layernorm1(q + attn_output)

        ff_output = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + ff_output)

        return out2


# %%
class TabularEncoder(nn.Module):
    def __init__(
        self,
        model_config,
        d_model=64,
        n_heads=4,
    ):
        super(TabularEncoder, self).__init__()
        self.d_model = d_model
        self.tokens = model_config.tokens
        self.token_dict = model_config.token_dict
        # self.decoder_dict = {v: k for k, v in self.token_dict.items()}
        # Masks
        self.cat_mask_token = torch.tensor(self.token_dict["[MASK]"])
        self.numeric_mask_token = torch.tensor(self.token_dict["[NUMERIC_MASK]"])

        self.n_tokens = len(self.tokens)  # TODO Make this
        # Embedding layers for categorical features
        self.embeddings = nn.Embedding(self.n_tokens, self.d_model)
        self.n_numeric_cols = len(model_config.numeric_col_tokens)
        self.n_cat_cols = len(model_config.categorical_col_tokens)
        self.col_tokens = (
            model_config.categorical_col_tokens + model_config.numeric_col_tokens
        )
        self.n_columns = self.n_numeric_cols + self.n_cat_cols
        # self.numeric_embeddings = NumericEmbedding(d_model=self.d_model)
        self.col_indices = torch.tensor(
            [self.tokens.index(col) for col in self.col_tokens], dtype=torch.long
        )
        self.numeric_indices = torch.tensor(
            [self.tokens.index(col) for col in model_config.numeric_col_tokens],
            dtype=torch.long,
        )
        self.transformer_encoder1 = TransformerEncoderLayer(d_model, n_heads=n_heads)
        self.transformer_encoder2 = TransformerEncoderLayer(d_model, n_heads=n_heads)

        self.mlm_decoder = nn.Sequential(nn.Linear(d_model, self.n_tokens))

        self.mnm_decoder = nn.Sequential(
            nn.Linear(
                self.n_columns * self.d_model, self.d_model * 4
            ),  # Try making more complex
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.n_numeric_cols),
        )

        self.flatten_layer = nn.Linear(len(self.col_tokens), 1)
        self.apply(initialize_parameters)

    def forward(self, num_inputs, cat_inputs):
        # Embed column indices
        repeated_col_indices = self.col_indices.unsqueeze(0).repeat(
            num_inputs.size(0), 1
        )
        col_embeddings = self.embeddings(repeated_col_indices)
        # Cast cat_inputs to int
        cat_inputs = cat_inputs.long()  # TODO Fix this in the dataset class
        cat_embeddings = self.embeddings(cat_inputs)

        expanded_num_inputs = num_inputs.unsqueeze(2).repeat(1, 1, self.d_model)
        with torch.no_grad():
            repeated_numeric_indices = self.numeric_indices.unsqueeze(0).repeat(
                num_inputs.size(0), 1
            )
            numeric_col_embeddings = self.embeddings(repeated_numeric_indices)

            inf_mask = (expanded_num_inputs == float("-inf")).all(dim=2)

        base_numeric = torch.zeros_like(expanded_num_inputs)

        num_embeddings = (
            numeric_col_embeddings[~inf_mask] * expanded_num_inputs[~inf_mask]
        )
        base_numeric[~inf_mask] = num_embeddings
        base_numeric[inf_mask] = self.embeddings(self.numeric_mask_token)

        query_embeddings = torch.cat([cat_embeddings, base_numeric], dim=1)
        out = self.transformer_encoder1(
            col_embeddings,
            # query_embeddings,
            query_embeddings,
            query_embeddings,
            # col_embeddings, query_embeddings, query_embeddings
        )
        out = self.transformer_encoder2(out, out, out)

        return out


class TabularEncoderRegressor(nn.Module):
    def __init__(
        self,
        model_config,
        d_model=64,
        n_heads=4,
    ):
        super(TabularEncoderRegressor, self).__init__()
        self.d_model = d_model
        self.tokens = model_config.tokens
        self.model_config = model_config

        self.tabular_encoder = TabularEncoder(model_config, d_model, n_heads)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 1),
        )
        self.flatten_layer = nn.Linear(self.model_config.n_columns, 1)
        self.apply(initialize_parameters)

    def forward(self, num_inputs, cat_inputs):
        out = self.tabular_encoder(num_inputs, cat_inputs)
        out = self.regressor(out)
        out = self.flatten_layer(out)
        return out


class MaskedTabularModeling(nn.Module):
    def __init__(
        self,
        model_config,
        d_model=64,
        n_heads=4,
    ):
        super(MaskedTabularModeling, self).__init__()
        self.d_model = d_model
        self.tokens = model_config.tokens

        self.tabular_encoder = TabularEncoder(model_config, d_model, n_heads)
        self.mlm_decoder = nn.Sequential(nn.Linear(d_model, self.n_tokens))
        self.mnm_decoder = nn.Sequential(
            nn.Linear(
                self.n_columns * self.d_model, self.d_model * 4
            ),  # Try making more complex
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.n_numeric_cols),
        )
        self.flatten_layer = nn.Linear(len(self.col_tokens), 1)
        self.apply(initialize_parameters)

    def forward(self, num_inputs, cat_inputs):
        out = self.tabular_encoder(num_inputs, cat_inputs)
        cat_out = self.mlm_decoder(out)
        numeric_out = out.view(out.size(0), -1)
        # print(f"numeric_out shape: {numeric_out.shape}")
        numeric_out = self.mnm_decoder(numeric_out)
        return NumericCategoricalData(numeric_out=numeric_out, categorical_out=cat_out)
