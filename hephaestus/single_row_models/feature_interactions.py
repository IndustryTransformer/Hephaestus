import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


class LowRankCross(nn.Module):
    """Low-rank bilinear cross layer for numeric feature interactions."""
    
    def __init__(self, in_dim: int, rank: int = 32, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.W1 = nn.Linear(in_dim, rank, bias=False)
        self.W2 = nn.Linear(in_dim, rank, bias=False)
        self.W_out = nn.Linear(rank, in_dim, bias=False)  # Separate output projection
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(in_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, d_model)
        Returns:
            (batch_size, d_model) with cross interactions
        """
        u, v = self.W1(x), self.W2(x)  # (batch_size, rank)
        z = u * v  # Element-wise product (batch_size, rank)
        z = self.W_out(z)  # Back-project to original dimension (batch_size, d_model)
        z = self.dropout(z)
        return self.bn(x + z)  # Residual connection


class CategoryInteractionHash(nn.Module):
    """Efficient categorical feature interaction using hashing."""
    
    def __init__(self, n_cat_features: int, hash_size: int = 1000, d_model: int = 64):
        super().__init__()
        self.n_cat_features = n_cat_features
        self.hash_size = hash_size
        self.d_model = d_model
        self.interaction_embeddings = nn.Embedding(hash_size, d_model)
        
    def _hash_pair(self, cat_i: torch.Tensor, cat_j: torch.Tensor) -> torch.Tensor:
        """Hash categorical pairs using a simple hash function."""
        # Simple hash: combine categorical values and mod by hash_size
        hash_val = (cat_i * 17 + cat_j * 31) % self.hash_size
        return hash_val
        
    def forward(self, cat_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cat_features: (batch_size, n_cat_features)
        Returns:
            (batch_size, n_interactions, d_model) interaction embeddings
        """
        if self.n_cat_features < 2:
            return torch.zeros(cat_features.size(0), 0, self.d_model, device=cat_features.device)
            
        interactions = []
        for i in range(self.n_cat_features):
            for j in range(i + 1, self.n_cat_features):
                hash_idx = self._hash_pair(cat_features[:, i], cat_features[:, j])
                interaction_embed = self.interaction_embeddings(hash_idx)
                interactions.append(interaction_embed)
                
        if interactions:
            return torch.stack(interactions, dim=1)
        else:
            return torch.zeros(cat_features.size(0), 0, self.d_model, device=cat_features.device)


class CrossNetV2(nn.Module):
    """Deep Cross Network v2 for learning feature crosses."""
    
    def __init__(self, in_dim: int, num_layers: int = 2, mix_type: str = 'matrix'):
        super().__init__()
        self.num_layers = num_layers
        self.mix_type = mix_type
        
        if mix_type == 'matrix':
            self.kernels = nn.ModuleList([
                nn.Linear(in_dim, in_dim) for _ in range(num_layers)
            ])
        elif mix_type == 'expert':
            # Low-rank expert mixing
            self.U = nn.ModuleList([
                nn.Linear(in_dim, in_dim // 4) for _ in range(num_layers)
            ])
            self.V = nn.ModuleList([
                nn.Linear(in_dim // 4, in_dim) for _ in range(num_layers)
            ])
        
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(in_dim)) for _ in range(num_layers)
        ])
        
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: (batch_size, in_dim) input features
        Returns:
            (batch_size, in_dim) with cross interactions
        """
        x_l = x0
        
        for i in range(self.num_layers):
            if self.mix_type == 'matrix':
                xl_w = self.kernels[i](x_l)
            elif self.mix_type == 'expert':
                xl_w = self.V[i](self.U[i](x_l))
            
            x_l = x0 * xl_w + self.bias[i] + x_l
            
        return x_l


class FeatureImportanceFilter:
    """Filter features based on importance scores."""
    
    def __init__(self, method: str = 'mutual_info', top_k: int = 50):
        self.method = method
        self.top_k = top_k
        self.selected_features = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """
        Fit the feature selector.
        
        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) target values
            feature_names: List of feature names
        """
        if self.method == 'mutual_info':
            scores = mutual_info_regression(X, y, random_state=42)
        elif self.method == 'rf_importance':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            scores = rf.feature_importances_
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # Get top-k features
        top_indices = np.argsort(scores)[-self.top_k:]
        self.selected_features = [feature_names[i] for i in top_indices]
        
    def get_important_pairs(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str], top_pairs: int = 20) -> List[Tuple[int, int]]:
        """Get top feature pairs based on interaction importance."""
        if self.method == 'rf_importance':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Simple heuristic: multiply individual importances
            importances = rf.feature_importances_
            pair_scores = []
            
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    score = importances[i] * importances[j]
                    pair_scores.append((score, i, j))
                    
            # Sort by score and return top pairs
            pair_scores.sort(reverse=True)
            return [(i, j) for _, i, j in pair_scores[:top_pairs]]
        
        return []


class InteractionAwareEncoder(nn.Module):
    """Enhanced encoder with feature interaction capabilities."""
    
    def __init__(self, base_encoder, d_model: int, n_numeric_cols: int, 
                 n_cat_cols: int, interaction_config: dict = None):
        super().__init__()
        self.base_encoder = base_encoder
        self.d_model = d_model
        self.n_numeric_cols = n_numeric_cols
        self.n_cat_cols = n_cat_cols
        
        # Default interaction configuration
        if interaction_config is None:
            interaction_config = {
                'use_lowrank_cross': True,
                'use_cat_interactions': True,
                'use_crossnet': True,
                'lowrank_rank': 32,
                'crossnet_layers': 2,
                'hash_size': 1000
            }
        
        self.config = interaction_config
        
        # Initialize interaction layers
        if self.config['use_lowrank_cross']:
            self.lowrank_cross = LowRankCross(
                d_model, rank=self.config['lowrank_rank']
            )
        
        if self.config['use_cat_interactions'] and n_cat_cols > 1:
            self.cat_interactions = CategoryInteractionHash(
                n_cat_cols, hash_size=self.config['hash_size'], d_model=d_model
            )
        
        if self.config['use_crossnet']:
            self.crossnet = CrossNetV2(
                d_model, num_layers=self.config['crossnet_layers']
            )
        
        # Projection layer to combine interactions
        interaction_dim = d_model
        if self.config['use_cat_interactions'] and n_cat_cols > 1:
            n_cat_pairs = n_cat_cols * (n_cat_cols - 1) // 2
            interaction_dim += n_cat_pairs * d_model
            
        # Always create interaction projection, even if just identity
        self.interaction_proj = nn.Linear(interaction_dim, d_model)
        
        # If no categorical interactions, create identity projection
        if interaction_dim == d_model:
            self.interaction_proj = nn.Identity()
        
    def forward(self, num_inputs: torch.Tensor, cat_inputs: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with feature interactions.
        
        Args:
            num_inputs: (batch_size, n_numeric_cols)
            cat_inputs: (batch_size, n_cat_cols)
        Returns:
            (batch_size, seq_len, d_model) enhanced features
        """
        # Get base encoder output
        base_output = self.base_encoder(num_inputs, cat_inputs)
        
        # Apply pooling to get sequence representation
        pooled_output = base_output.mean(dim=1)  # (batch_size, d_model)
        
        # Apply interaction layers
        enhanced_features = pooled_output
        
        if self.config['use_lowrank_cross']:
            enhanced_features = self.lowrank_cross(enhanced_features)
            
        if self.config['use_crossnet']:
            enhanced_features = self.crossnet(enhanced_features)
        
        # Add categorical interactions if applicable
        if (self.config['use_cat_interactions'] and 
            cat_inputs is not None and 
            self.n_cat_cols > 1 and
            hasattr(self, 'cat_interactions')):
            cat_interactions = self.cat_interactions(cat_inputs)  # (batch_size, n_pairs, d_model)
            if cat_interactions.size(1) > 0:  # Only if there are actual interactions
                cat_interactions_flat = cat_interactions.view(cat_inputs.size(0), -1)
                enhanced_features = torch.cat([enhanced_features, cat_interactions_flat], dim=1)
        
        # Always apply projection (identity if no categorical interactions)
        enhanced_features = self.interaction_proj(enhanced_features)
        
        # Expand back to sequence format for compatibility
        enhanced_features = enhanced_features.unsqueeze(1).expand(-1, base_output.size(1), -1)
        
        return enhanced_features + base_output  # Residual connection