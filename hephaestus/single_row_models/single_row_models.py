import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from hephaestus.single_row_models.single_row_utils import (
    initialize_parameters,
)
from hephaestus.utils import NumericCategoricalData

from .model_data_classes import SingleRowConfig


# %%
class NumericEmbedding(nn.Module):
    """Linear embedding for numeric features with normalization."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(1, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size] numeric values
        Returns:
            [batch_size, d_model] embedded features
        """
        x = x.unsqueeze(-1)  # [batch_size, 1]
        embedded = self.projection(x)  # [batch_size, d_model]
        embedded = self.layer_norm(embedded)
        return self.dropout(embedded)


class SimpleNumericEmbedding(nn.Module):
    """Simplified linear embedding without normalization or dropout."""
    
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(1, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size] numeric values
        Returns:
            [batch_size, d_model] embedded features
        """
        x = x.unsqueeze(-1)  # [batch_size, 1]
        return self.projection(x)  # [batch_size, d_model]


class ComplexNumericEmbedding(nn.Module):
    """More complex numeric embedding with multiple layers and higher capacity."""
    
    def __init__(self, d_model, hidden_multiplier=4, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * hidden_multiplier
        
        self.embedding_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size] numeric values
        Returns:
            [batch_size, d_model] embedded features
        """
        x = x.unsqueeze(-1)  # [batch_size, 1]
        embedded = self.embedding_net(x)  # [batch_size, d_model]
        return self.layer_norm(embedded)


class MultiHeadAttention(nn.Module):  # Try to use nn.MultiheadAttention
    def __init__(self, d_model, n_heads):
        super().__init__()
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
    def __init__(self, d_model, n_heads, dropout=0.2):
        super(TransformerEncoderLayer, self).__init__()

        # self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.multi_head_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * d_model, d_model * 4),
            nn.Linear(4 * d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 4, d_model),
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        # self.initialize_parameters()
        self.apply(initialize_parameters)

    def forward(self, q, k, v, mask=None, input_feed_forward=False):
        attn_output, weights = self.multi_head_attention(q, k, v, mask)
        out1 = self.layernorm1(q + attn_output)

        ff_output = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + ff_output)

        return out2


# %%
class TabularEncoder(nn.Module):
    def __init__(
        self,
        model_config: SingleRowConfig,
        d_model=64,
        n_heads=4,
        use_linear_numeric_embedding=True,
        numeric_embedding_type="original",  # "original", "simple", "standard", "complex"
    ):
        super(TabularEncoder, self).__init__()
        self.d_model = d_model
        self.tokens = model_config.tokens
        self.use_linear_numeric_embedding = use_linear_numeric_embedding
        self.numeric_embedding_type = numeric_embedding_type

        self.token_dict = model_config.token_dict
        # self.decoder_dict = {v: k for k, v in self.token_dict.items()}
        # Masks
        # self.cat_mask_token = torch.tensor(self.token_dict["[MASK]"])
        self.register_buffer("cat_mask_token", torch.tensor(self.token_dict["[MASK]"]))
        # self.numeric_mask_token = torch.tensor(self.token_dict["[NUMERIC_MASK]"])
        self.register_buffer(
            "numeric_mask_token", torch.tensor(self.token_dict["[NUMERIC_MASK]"])
        )

        self.n_tokens = len(self.tokens)  # TODO Make this
        # Embedding layers for categorical features
        self.embeddings = nn.Embedding(self.n_tokens, self.d_model)
        self.n_numeric_cols = model_config.n_numeric_cols
        self.n_cat_cols = model_config.n_cat_cols
        self.col_tokens = (
            model_config.categorical_col_tokens + model_config.numeric_col_tokens
        )
        self.n_columns = self.n_numeric_cols + self.n_cat_cols
        # self.numeric_embeddings = NumericEmbedding(d_model=self.d_model)
        self.register_buffer(
            "col_indices",
            torch.tensor(
                [self.tokens.index(col) for col in self.col_tokens], dtype=torch.long
            ),
        )

        self.register_buffer(
            "numeric_indices",
            torch.tensor(
                [self.tokens.index(col) for col in model_config.numeric_col_tokens],
                dtype=torch.long,
            ),
        )
        
        # Add numeric embeddings for linear embedding approach
        if self.use_linear_numeric_embedding:
            if self.numeric_embedding_type == "simple":
                self.numeric_embeddings = nn.ModuleList([
                    SimpleNumericEmbedding(d_model) for _ in range(self.n_numeric_cols)
                ])
            elif self.numeric_embedding_type == "complex":
                self.numeric_embeddings = nn.ModuleList([
                    ComplexNumericEmbedding(d_model) for _ in range(self.n_numeric_cols)
                ])
            else:  # "standard" or default
                self.numeric_embeddings = nn.ModuleList([
                    NumericEmbedding(d_model) for _ in range(self.n_numeric_cols)
                ])
        self.transformer_encoder1 = TransformerEncoderLayer(d_model, n_heads=n_heads)
        self.transformer_encoder2 = TransformerEncoderLayer(d_model, n_heads=n_heads)
        self.transformer_encoder3 = TransformerEncoderLayer(d_model, n_heads=n_heads)
        self.transformer_encoder4 = TransformerEncoderLayer(d_model, n_heads=n_heads)
        
        # self.flatten_layer = nn.Linear(len(self.col_tokens), 1)
        self.apply(initialize_parameters)

    def forward(self, num_inputs, cat_inputs):
        # expand dims if only 1

        if num_inputs.dim() == 1 and cat_inputs.dim() == 1:
            num_inputs = num_inputs.unsqueeze(0)
            cat_inputs = cat_inputs.unsqueeze(0)
        elif num_inputs.dim() == 2 and cat_inputs.dim() == 2:
            pass
        else:
            raise ValueError(
                f"Incorrect input dimensions {num_inputs.dim()=}, {cat_inputs.dim()=}"
            )
        # Embed column indices
        repeated_col_indices = self.col_indices.unsqueeze(0).repeat(
            num_inputs.size(0), 1
        )
        col_embeddings = self.embeddings(repeated_col_indices)
        # Cast cat_inputs to int
        cat_inputs = cat_inputs.long()  # TODO Fix this in the dataset class
        cat_embeddings = self.embeddings(cat_inputs)

        if self.use_linear_numeric_embedding:
            # New linear embedding approach
            numeric_embeddings_list = []
            for i, embedding_layer in enumerate(self.numeric_embeddings):
                feature_values = num_inputs[:, i]  # [batch_size]
                
                # Handle mask tokens
                inf_mask = (feature_values == float("-inf"))
                clean_values = feature_values.clone()
                clean_values[inf_mask] = 0.0  # Set to 0 for embedding
                
                embedded_feature = embedding_layer(clean_values)  # [batch_size, d_model]
                
                # Apply mask token for masked positions
                if inf_mask.any():
                    mask_embedding = self.embeddings(self.numeric_mask_token)
                    embedded_feature[inf_mask] = mask_embedding
                
                numeric_embeddings_list.append(embedded_feature)
            
            base_numeric = torch.stack(numeric_embeddings_list, dim=1)  # [batch_size, n_numeric, d_model]
        
        else:
            # Original multiplication approach (for backward compatibility)
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
        out1 = self.transformer_encoder1(
            col_embeddings,
            # query_embeddings,
            query_embeddings,
            query_embeddings,
            # col_embeddings, query_embeddings, query_embeddings
        )
        
        # No skipping connection
        out2 = self.transformer_encoder2(out1, out1, out1)
        out2 = out2 + out1
        
        out3 = self.transformer_encoder3(out2, out2, out2)
        out3 = out3 + out2
        out4 = self.transformer_encoder4(out3, out3, out3)
        out4 = out4 + out3
        return out4


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Query vector for attention
        self.query = nn.Parameter(torch.randn(d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.scaling_factor = d_model**-0.5

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Project sequence to keys
        keys = self.key_proj(x)  # [batch_size, seq_len, d_model]

        # Calculate attention scores
        query = self.query.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        scores = (
            torch.matmul(query, keys.transpose(-2, -1)) * self.scaling_factor
        )  # [batch_size, 1, seq_len]

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, seq_len]

        # Apply attention weights to sequence
        context = torch.matmul(attn_weights, x)  # [batch_size, 1, d_model]

        return context.squeeze(1)  # [batch_size, d_model]


class TabularEncoderRegressor(nn.Module):
    def __init__(
        self,
        model_config: SingleRowConfig,
        d_model=64,
        n_heads=4,
        use_linear_numeric_embedding=True,
        numeric_embedding_type="standard",
    ):
        super(TabularEncoderRegressor, self).__init__()
        self.d_model = d_model
        self.tokens = model_config.tokens
        self.model_config = model_config
        self.Dropout_rate = 0.2
        self.dropout = nn.Dropout(self.Dropout_rate)
        self.tabular_encoder = TabularEncoder(
            model_config, d_model, n_heads, use_linear_numeric_embedding, numeric_embedding_type
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),  # Try GELU instead of ReLU
            nn.Dropout(self.Dropout_rate),
            nn.Linear(self.d_model * 2, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.Dropout_rate),
            nn.Linear(self.d_model * 2, 1),
        )

        self.pooling = AttentionPooling(d_model)
        # self.pooling = nn.MaxPool3d((1, 1, self.d_model))
        # self.flatten_layer = nn.Linear(
        #     self.model_config.n_columns_no_target, 1
        # )
        # self.apply(initialize_parameters)

    def forward(self, num_inputs, cat_inputs):
        out = self.tabular_encoder(num_inputs, cat_inputs)
        # out = self.pooling(out)  # Could use max pooling too
        # out = torch.max(out, dim=1)[0]  # [batch_size, d_model]
        out = self.pooling(out)
        out = self.dropout(out)
        # skip = out
        out = self.regressor(out)
        # out = out + skip
        return out
    


class MaskedTabularEncoder(nn.Module):
    def __init__(
        self,
        model_config: SingleRowConfig,
        d_model=64,
        n_heads=4,
        use_linear_numeric_embedding=True,
        numeric_embedding_type="standard",
    ):
        super().__init__()
        self.d_model = d_model
        self.tokens = model_config.tokens
        self.n_tokens = len(self.tokens)
        self.model_config = model_config
        self.tabular_encoder = TabularEncoder(
            model_config, d_model, n_heads, use_linear_numeric_embedding, numeric_embedding_type
        )
        # Initialize decoder layers with dynamic sizing
        self.mlm_decoder = None
        self.mnm_decoder = None
        self._decoders_initialized = False

        # self.apply(initialize_parameters)

    def _initialize_decoders(self, encoder_output_shape):
        """Initialize decoder layers based on actual encoder output dimensions."""
        if self._decoders_initialized:
            return
            
        batch_size, seq_len, feature_dim = encoder_output_shape
        flattened_dim = seq_len * feature_dim
        
        # MLM decoder for categorical prediction
        self.mlm_decoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                flattened_dim,
                self.model_config.n_cat_cols * self.model_config.n_tokens,
            ),
        )
        
        # MNM decoder for numeric prediction
        self.mnm_decoder = nn.Sequential(
            nn.Linear(flattened_dim, self.d_model * 4),
            nn.GELU(),
            nn.Linear(self.d_model * 4, self.model_config.n_numeric_cols),
        )
        
        # Move to same device as encoder
        try:
            device = next(self.tabular_encoder.parameters()).device
            self.mlm_decoder = self.mlm_decoder.to(device)
            self.mnm_decoder = self.mnm_decoder.to(device)
        except StopIteration:
            # No parameters in encoder, use CPU
            pass
            
        self._decoders_initialized = True

    def forward(self, num_inputs, cat_inputs):
        out = self.tabular_encoder(num_inputs, cat_inputs)
        
        # Initialize decoders based on actual output shape
        self._initialize_decoders(out.shape)

        # Ensure categorical output is logits
        cat_out = self.mlm_decoder(out)
        cat_out = cat_out.view(
            out.size(0), self.model_config.n_cat_cols, self.model_config.n_tokens
        )
        # cat_out = cat_out.permute(0, 2, 1)

        # No need to flatten here; keep shape as [batch_size, seq_len, num_classes]
        numeric_out = out.view(out.size(0), -1)  # Flatten numeric output
        numeric_out = self.mnm_decoder(numeric_out)
        return NumericCategoricalData(numeric=numeric_out, categorical=cat_out)
    
