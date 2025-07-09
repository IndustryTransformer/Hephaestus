"""
Enhanced categorical variable processing for MoE-enabled tabular models.

This module provides specialized components for handling categorical variables
in the context of Mixture of Experts models, including:
- Enhanced categorical embeddings
- Categorical-numeric interaction modeling
- Categorical feature engineering
- Categorical attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from torch import Tensor


class EnhancedCategoricalEmbedding(nn.Module):
    """
    Enhanced categorical embedding that supports multiple embedding strategies.
    
    This class provides several improvements over standard embeddings:
    - Frequency-based embedding scaling
    - Rare category handling
    - Hierarchical embeddings for high-cardinality categories
    - Contextual embeddings that consider other categorical features
    
    Args:
        vocab_size: Size of the categorical vocabulary
        embedding_dim: Dimension of the embedding vectors
        dropout: Dropout rate
        use_frequency_scaling: Whether to scale embeddings by frequency
        use_hierarchical: Whether to use hierarchical embeddings for high cardinality
        rare_threshold: Threshold for considering categories as rare
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout: float = 0.1,
        use_frequency_scaling: bool = True,
        use_hierarchical: bool = False,
        rare_threshold: int = 5,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_frequency_scaling = use_frequency_scaling
        self.use_hierarchical = use_hierarchical
        self.rare_threshold = rare_threshold
        
        # Main embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Frequency scaling parameters
        if use_frequency_scaling:
            self.register_buffer("category_counts", torch.ones(vocab_size))
            self.register_buffer("total_count", torch.tensor(1.0))
            
        # Hierarchical embeddings for high cardinality
        if use_hierarchical:
            self.cluster_embedding = nn.Embedding(vocab_size // 10 + 1, embedding_dim // 2)
            self.detail_embedding = nn.Embedding(vocab_size, embedding_dim // 2)
            
        # Rare category handling
        self.rare_category_embedding = nn.Parameter(torch.randn(embedding_dim))
        
        # Contextual attention for categorical interactions
        self.contextual_attention = nn.MultiheadAttention(
            embedding_dim, 
            num_heads=max(1, embedding_dim // 64),
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(
        self, 
        category_ids: Tensor, 
        context_categories: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass with enhanced categorical processing.
        
        Args:
            category_ids: Categorical indices [batch_size, seq_len] or [batch_size]
            context_categories: Other categorical features for context [batch_size, context_len]
            
        Returns:
            Enhanced categorical embeddings [batch_size, seq_len, embedding_dim]
        """
        # Handle single dimension input
        if category_ids.dim() == 1:
            category_ids = category_ids.unsqueeze(1)
            
        batch_size, seq_len = category_ids.shape
        
        # Get base embeddings
        if self.use_hierarchical:
            # Hierarchical embeddings
            cluster_ids = category_ids // 10  # Simple clustering
            cluster_emb = self.cluster_embedding(cluster_ids)
            detail_emb = self.detail_embedding(category_ids)
            base_embeddings = torch.cat([cluster_emb, detail_emb], dim=-1)
        else:
            # Standard embeddings
            base_embeddings = self.embedding(category_ids)
            
        # Apply frequency scaling
        if self.use_frequency_scaling and self.training:
            self._update_frequency_counts(category_ids)
            frequencies = self.category_counts[category_ids] / self.total_count
            freq_weights = torch.log(1 + frequencies).unsqueeze(-1)
            base_embeddings = base_embeddings * freq_weights
            
        # Handle rare categories
        if self.rare_threshold > 0:
            rare_mask = self.category_counts[category_ids] < self.rare_threshold
            if rare_mask.any():
                base_embeddings[rare_mask] = self.rare_category_embedding
                
        # Apply contextual attention if context is provided
        if context_categories is not None:
            context_emb = self.embedding(context_categories)
            attended_emb, _ = self.contextual_attention(
                base_embeddings, context_emb, context_emb
            )
            base_embeddings = base_embeddings + attended_emb
            
        # Apply normalization and dropout
        embeddings = self.layer_norm(base_embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def _update_frequency_counts(self, category_ids: Tensor):
        """Update frequency counts for categories."""
        # Count occurrences
        unique_ids, counts = torch.unique(category_ids, return_counts=True)
        
        # Update counts with momentum
        momentum = 0.99
        for category_id, count in zip(unique_ids, counts):
            if category_id < self.vocab_size:
                self.category_counts[category_id] = (
                    momentum * self.category_counts[category_id] + 
                    (1 - momentum) * count.float()
                )
                
        # Update total count
        self.total_count = momentum * self.total_count + (1 - momentum) * category_ids.numel()


class CategoricalInteractionLayer(nn.Module):
    """
    Specialized layer for modeling categorical-numeric interactions.
    
    This layer captures complex interactions between categorical and numeric features
    using various interaction mechanisms:
    - Categorical gating of numeric features
    - Numeric modulation of categorical embeddings
    - Cross-modal attention mechanisms
    
    Args:
        categorical_dim: Dimension of categorical embeddings
        numeric_dim: Dimension of numeric features
        output_dim: Output dimension
        interaction_type: Type of interaction ('gating', 'modulation', 'attention')
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        categorical_dim: int,
        numeric_dim: int,
        output_dim: int,
        interaction_type: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.categorical_dim = categorical_dim
        self.numeric_dim = numeric_dim
        self.output_dim = output_dim
        self.interaction_type = interaction_type
        
        if interaction_type == "gating":
            # Categorical features gate numeric features
            self.categorical_gate = nn.Sequential(
                nn.Linear(categorical_dim, numeric_dim),
                nn.Sigmoid(),
            )
            self.output_proj = nn.Linear(numeric_dim, output_dim)
            
        elif interaction_type == "modulation":
            # Numeric features modulate categorical embeddings
            self.numeric_modulator = nn.Sequential(
                nn.Linear(numeric_dim, categorical_dim),
                nn.Tanh(),
            )
            self.output_proj = nn.Linear(categorical_dim, output_dim)
            
        elif interaction_type == "attention":
            # Cross-modal attention between categorical and numeric
            self.cross_attention = nn.MultiheadAttention(
                max(categorical_dim, numeric_dim),
                num_heads=max(1, max(categorical_dim, numeric_dim) // 64),
                dropout=dropout,
                batch_first=True
            )
            
            # Projection layers
            self.cat_proj = nn.Linear(categorical_dim, max(categorical_dim, numeric_dim))
            self.num_proj = nn.Linear(numeric_dim, max(categorical_dim, numeric_dim))
            self.output_proj = nn.Linear(max(categorical_dim, numeric_dim), output_dim)
            
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")
            
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self, 
        categorical_features: Tensor, 
        numeric_features: Tensor
    ) -> Tensor:
        """
        Forward pass for categorical-numeric interaction.
        
        Args:
            categorical_features: Categorical embeddings [batch_size, seq_len, categorical_dim]
            numeric_features: Numeric features [batch_size, seq_len, numeric_dim]
            
        Returns:
            Interaction output [batch_size, seq_len, output_dim]
        """
        if self.interaction_type == "gating":
            # Categorical gating
            gates = self.categorical_gate(categorical_features)
            gated_numeric = numeric_features * gates
            output = self.output_proj(gated_numeric)
            
        elif self.interaction_type == "modulation":
            # Numeric modulation
            modulation = self.numeric_modulator(numeric_features)
            modulated_categorical = categorical_features * modulation
            output = self.output_proj(modulated_categorical)
            
        elif self.interaction_type == "attention":
            # Cross-modal attention
            cat_proj = self.cat_proj(categorical_features)
            num_proj = self.num_proj(numeric_features)
            
            # Attention from categorical to numeric
            attended_output, _ = self.cross_attention(cat_proj, num_proj, num_proj)
            output = self.output_proj(attended_output)
            
        # Apply normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class CategoricalFeatureEngineer(nn.Module):
    """
    Neural feature engineering for categorical variables.
    
    This module automatically discovers and creates categorical features:
    - Categorical combinations and interactions
    - Frequency-based features
    - Categorical clustering
    - Temporal patterns in categorical data
    
    Args:
        vocab_size: Size of categorical vocabulary
        embedding_dim: Dimension of embeddings
        num_engineered_features: Number of features to engineer
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_engineered_features: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_engineered_features = num_engineered_features
        
        # Base embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Feature engineering networks
        self.combination_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_engineered_features),
        )
        
        self.frequency_network = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim),  # +1 for frequency
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_engineered_features),
        )
        
        # Clustering for categorical similarity
        self.cluster_centers = nn.Parameter(torch.randn(10, embedding_dim))
        
        # Frequency tracking
        self.register_buffer("category_frequencies", torch.ones(vocab_size))
        self.register_buffer("total_samples", torch.tensor(1.0))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, category_ids: Tensor) -> Tensor:
        """
        Engineer categorical features.
        
        Args:
            category_ids: Categorical indices [batch_size, seq_len]
            
        Returns:
            Engineered features [batch_size, seq_len, num_engineered_features]
        """
        batch_size, seq_len = category_ids.shape
        
        # Get base embeddings
        embeddings = self.embedding(category_ids)
        
        # Update frequencies
        if self.training:
            self._update_frequencies(category_ids)
            
        # 1. Combination features (pairwise interactions)
        combination_features = []
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                combined = torch.cat([embeddings[:, i], embeddings[:, j]], dim=-1)
                combo_feat = self.combination_network(combined)
                combination_features.append(combo_feat)
                
        # 2. Frequency-based features
        frequencies = self.category_frequencies[category_ids] / self.total_samples
        freq_input = torch.cat([embeddings, frequencies.unsqueeze(-1)], dim=-1)
        freq_features = self.frequency_network(freq_input)
        
        # 3. Clustering-based features
        cluster_distances = torch.cdist(embeddings, self.cluster_centers)
        cluster_features = F.softmax(-cluster_distances, dim=-1)
        
        # Combine all engineered features
        all_features = [freq_features]
        if combination_features:
            all_features.extend(combination_features)
        all_features.append(cluster_features)
        
        # Aggregate features
        if len(all_features) > 1:
            engineered_features = torch.stack(all_features, dim=2).mean(dim=2)
        else:
            engineered_features = all_features[0]
            
        return self.dropout(engineered_features)
    
    def _update_frequencies(self, category_ids: Tensor):
        """Update category frequency statistics."""
        unique_ids, counts = torch.unique(category_ids, return_counts=True)
        
        momentum = 0.99
        for category_id, count in zip(unique_ids, counts):
            if category_id < self.vocab_size:
                self.category_frequencies[category_id] = (
                    momentum * self.category_frequencies[category_id] + 
                    (1 - momentum) * count.float()
                )
                
        self.total_samples = momentum * self.total_samples + (1 - momentum) * category_ids.numel()


class CategoricalAwareTabularEncoder(nn.Module):
    """
    Tabular encoder with enhanced categorical variable processing.
    
    This encoder extends the standard tabular encoder with:
    - Enhanced categorical embeddings
    - Categorical-numeric interactions
    - Categorical feature engineering
    - Categorical-aware attention mechanisms
    
    Args:
        model_config: Model configuration
        d_model: Model dimension
        n_heads: Number of attention heads
        use_enhanced_categorical: Whether to use enhanced categorical processing
        use_categorical_interactions: Whether to model categorical-numeric interactions
        use_categorical_engineering: Whether to use categorical feature engineering
    """
    
    def __init__(
        self,
        model_config,
        d_model: int = 64,
        n_heads: int = 4,
        use_enhanced_categorical: bool = True,
        use_categorical_interactions: bool = True,
        use_categorical_engineering: bool = True,
    ):
        super().__init__()
        
        self.model_config = model_config
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_enhanced_categorical = use_enhanced_categorical
        self.use_categorical_interactions = use_categorical_interactions
        self.use_categorical_engineering = use_categorical_engineering
        
        # Enhanced categorical embeddings
        if use_enhanced_categorical and model_config.n_cat_cols > 0:
            self.enhanced_cat_embeddings = nn.ModuleList([
                EnhancedCategoricalEmbedding(
                    vocab_size=model_config.n_tokens,
                    embedding_dim=d_model,
                    use_frequency_scaling=True,
                    use_hierarchical=model_config.n_tokens > 1000,
                ) for _ in range(model_config.n_cat_cols)
            ])
            
        # Categorical-numeric interactions
        if use_categorical_interactions and model_config.n_cat_cols > 0 and model_config.n_numeric_cols > 0:
            self.categorical_interactions = nn.ModuleList([
                CategoricalInteractionLayer(
                    categorical_dim=d_model,
                    numeric_dim=d_model,
                    output_dim=d_model,
                    interaction_type="attention",
                ) for _ in range(min(model_config.n_cat_cols, model_config.n_numeric_cols))
            ])
            
        # Categorical feature engineering
        if use_categorical_engineering and model_config.n_cat_cols > 0:
            self.categorical_engineer = CategoricalFeatureEngineer(
                vocab_size=model_config.n_tokens,
                embedding_dim=d_model,
                num_engineered_features=d_model // 2,
            )
            
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self, 
        categorical_features: Tensor, 
        numeric_features: Tensor
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Forward pass with enhanced categorical processing.
        
        Args:
            categorical_features: Categorical indices [batch_size, n_cat_cols]
            numeric_features: Numeric features [batch_size, n_numeric_cols]
            
        Returns:
            enhanced_categorical: Enhanced categorical embeddings [batch_size, n_cat_cols, d_model]
            enhanced_numeric: Enhanced numeric features [batch_size, n_numeric_cols, d_model]
            metrics: Dictionary of categorical processing metrics
        """
        batch_size = categorical_features.shape[0] if categorical_features.numel() > 0 else numeric_features.shape[0]
        metrics = {}
        
        # Process categorical features
        if self.use_enhanced_categorical and categorical_features.numel() > 0:
            enhanced_categorical_list = []
            for i, embedding_layer in enumerate(self.enhanced_cat_embeddings):
                if i < categorical_features.shape[1]:
                    cat_emb = embedding_layer(categorical_features[:, i])
                    enhanced_categorical_list.append(cat_emb)
                    
            if enhanced_categorical_list:
                enhanced_categorical = torch.stack(enhanced_categorical_list, dim=1)
            else:
                enhanced_categorical = torch.zeros(batch_size, 0, self.d_model)
        else:
            enhanced_categorical = torch.zeros(batch_size, 0, self.d_model)
            
        # Process numeric features (placeholder - would integrate with existing numeric processing)
        if numeric_features.numel() > 0:
            enhanced_numeric = numeric_features.unsqueeze(-1).expand(-1, -1, self.d_model)
        else:
            enhanced_numeric = torch.zeros(batch_size, 0, self.d_model)
            
        # Categorical-numeric interactions
        if (self.use_categorical_interactions and 
            enhanced_categorical.shape[1] > 0 and 
            enhanced_numeric.shape[1] > 0):
            
            interaction_outputs = []
            for i, interaction_layer in enumerate(self.categorical_interactions):
                if i < min(enhanced_categorical.shape[1], enhanced_numeric.shape[1]):
                    cat_feat = enhanced_categorical[:, i:i+1]
                    num_feat = enhanced_numeric[:, i:i+1]
                    interaction_out = interaction_layer(cat_feat, num_feat)
                    interaction_outputs.append(interaction_out)
                    
            if interaction_outputs:
                interaction_features = torch.cat(interaction_outputs, dim=1)
                metrics["interaction_features"] = interaction_features.mean().item()
                
        # Categorical feature engineering
        if self.use_categorical_engineering and categorical_features.numel() > 0:
            engineered_features = self.categorical_engineer(categorical_features)
            metrics["engineered_features"] = engineered_features.mean().item()
            
        # Apply output projection
        if enhanced_categorical.shape[1] > 0:
            enhanced_categorical = self.output_proj(enhanced_categorical)
        if enhanced_numeric.shape[1] > 0:
            enhanced_numeric = self.output_proj(enhanced_numeric)
            
        return enhanced_categorical, enhanced_numeric, metrics