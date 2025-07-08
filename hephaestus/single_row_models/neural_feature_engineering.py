"""
Neural Feature Engineering for Automatic Feature Interaction Discovery.

This module implements sophisticated automatic feature engineering that learns
meaningful feature interactions, polynomial combinations, ratios, and attention-based
relationships without manual feature engineering.

Key components:
- FeatureImportanceGate: Learns feature relevance per sample
- InteractionDiscovery: Discovers meaningful pairwise interactions
- PolynomialFeatureDiscovery: Learns polynomial combinations
- RatioFeatureDiscovery: Discovers ratios and differences
- CrossFeatureAttention: Attention-based feature interactions
- NeuralFeatureEngineer: Integrated feature engineering system
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeatureImportanceGate(nn.Module):
    """
    Learns which features are important for each sample dynamically.
    
    This gate assigns importance weights to features based on the input,
    allowing the model to focus on relevant features per sample rather
    than treating all features equally.
    
    Args:
        num_features: Number of input features
        hidden_dim: Hidden dimension for the gating network
        dropout: Dropout rate for regularization
        temperature: Temperature for importance weights (lower = more selective)
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.temperature = temperature
        
        # Importance network
        self.importance_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_features),
        )
        
        # Global feature statistics for normalization
        self.register_buffer("feature_means", torch.zeros(num_features))
        self.register_buffer("feature_stds", torch.ones(num_features))
        self.register_buffer("update_count", torch.tensor(0.0))
        
    def _update_feature_stats(self, x: Tensor):
        """Update running statistics of features."""
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_std = x.std(dim=0) + 1e-8
            
            # Moving average update
            momentum = 0.99
            self.feature_means = momentum * self.feature_means + (1 - momentum) * batch_mean
            self.feature_stds = momentum * self.feature_stds + (1 - momentum) * batch_std
            self.update_count += 1
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute importance-weighted features.
        
        Args:
            x: Input features [batch_size, num_features]
            
        Returns:
            weighted_features: Importance-weighted features [batch_size, num_features]
            importance_weights: Feature importance weights [batch_size, num_features]
        """
        self._update_feature_stats(x)
        
        # Normalize input for stable importance computation
        normalized_x = (x - self.feature_means) / self.feature_stds
        
        # Compute importance logits
        importance_logits = self.importance_net(normalized_x)
        
        # Apply temperature and sigmoid for importance weights
        importance_weights = torch.sigmoid(importance_logits / self.temperature)
        
        # Apply importance weighting
        weighted_features = x * importance_weights
        
        return weighted_features, importance_weights


class InteractionDiscovery(nn.Module):
    """
    Automatically discovers meaningful pairwise feature interactions.
    
    Uses learnable interaction weights to determine which feature pairs
    should interact, then processes the most important interactions through
    specialized MLPs.
    
    Args:
        num_features: Number of input features
        max_interactions: Maximum number of interactions to consider
        hidden_dim: Hidden dimension for interaction MLPs
        dropout: Dropout rate
        interaction_threshold: Minimum threshold for interaction selection
    """
    
    def __init__(
        self,
        num_features: int,
        max_interactions: int = 20,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        interaction_threshold: float = 0.1,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.max_interactions = max_interactions
        self.interaction_threshold = interaction_threshold
        
        # Learnable interaction importance matrix (upper triangular)
        self.interaction_weights = nn.Parameter(
            torch.randn(num_features, num_features) * 0.1
        )
        
        # Mask for upper triangular (avoid duplicate pairs)
        self.register_buffer(
            "triu_mask", 
            torch.triu(torch.ones(num_features, num_features), diagonal=1)
        )
        
        # Interaction processing MLPs
        self.interaction_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(max_interactions)
        ])
        
        # Context-aware interaction selector
        self.context_selector = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_interactions),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Discover and process feature interactions.
        
        Args:
            x: Input features [batch_size, num_features]
            
        Returns:
            interaction_features: Processed interactions [batch_size, max_interactions]
            metrics: Dictionary with interaction statistics
        """
        batch_size = x.shape[0]
        
        # Compute interaction importance scores (upper triangular only)
        interaction_scores = torch.sigmoid(self.interaction_weights) * self.triu_mask
        
        # Get context-dependent interaction selection
        context_weights = self.context_selector(x)  # [batch_size, max_interactions]
        
        # Find top interactions globally
        flat_scores = interaction_scores[self.triu_mask.bool()]
        top_indices = torch.topk(flat_scores, min(self.max_interactions, len(flat_scores))).indices
        
        # Convert flat indices back to 2D coordinates
        triu_positions = torch.nonzero(self.triu_mask, as_tuple=False)
        selected_pairs = triu_positions[top_indices]
        
        # Process interactions
        interaction_outputs = []
        interaction_importances = []
        
        for idx, (i, j) in enumerate(selected_pairs):
            if idx >= self.max_interactions:
                break
                
            # Get feature pair
            feature_i = x[:, i].unsqueeze(1)  # [batch_size, 1]
            feature_j = x[:, j].unsqueeze(1)  # [batch_size, 1]
            pair_input = torch.cat([feature_i, feature_j], dim=1)  # [batch_size, 2]
            
            # Process through interaction MLP
            interaction_output = self.interaction_mlps[idx](pair_input)  # [batch_size, 1]
            
            # Apply context weighting
            context_weight = context_weights[:, idx].unsqueeze(1)  # [batch_size, 1]
            weighted_output = interaction_output * context_weight
            
            interaction_outputs.append(weighted_output)
            interaction_importances.append(interaction_scores[i, j].item())
        
        # Pad with zeros if we have fewer interactions than max
        while len(interaction_outputs) < self.max_interactions:
            interaction_outputs.append(torch.zeros(batch_size, 1, device=x.device))
            interaction_importances.append(0.0)
        
        # Concatenate all interactions
        interaction_features = torch.cat(interaction_outputs, dim=1)  # [batch_size, max_interactions]
        
        # Metrics for monitoring
        metrics = {
            "interaction_importances": torch.tensor(interaction_importances),
            "context_weights_mean": context_weights.mean(dim=0),
            "active_interactions": sum(1 for imp in interaction_importances if imp > self.interaction_threshold),
            "selected_pairs": selected_pairs.cpu() if len(selected_pairs) > 0 else torch.empty(0, 2)
        }
        
        return interaction_features, metrics


class PolynomialFeatureDiscovery(nn.Module):
    """
    Learns polynomial combinations of features automatically.
    
    Instead of manually creating all polynomial combinations, this module
    learns which features should be combined polynomially and at what degrees.
    
    Args:
        num_features: Number of input features
        max_degree: Maximum polynomial degree to consider
        hidden_dim: Hidden dimension for selection networks
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_features: int,
        max_degree: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.max_degree = max_degree
        
        # Feature selection for polynomial computation
        self.feature_selector = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()
        )
        
        # Degree-specific weights
        self.degree_weights = nn.ParameterList([
            nn.Parameter(torch.randn(num_features) * 0.1)
            for _ in range(1, max_degree + 1)
        ])
        
        # Polynomial combination network
        self.combination_net = nn.Sequential(
            nn.Linear(max_degree, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_degree),
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Generate polynomial features.
        
        Args:
            x: Input features [batch_size, num_features]
            
        Returns:
            polynomial_features: Generated polynomial features [batch_size, max_degree]
            metrics: Dictionary with polynomial statistics
        """
        batch_size = x.shape[0]
        
        # Learn which features are important for polynomials
        selection_weights = self.feature_selector(x)  # [batch_size, num_features]
        selected_features = x * selection_weights
        
        # Generate polynomial features for each degree
        polynomial_features = []
        feature_contributions = []
        
        for degree_idx, degree in enumerate(range(1, self.max_degree + 1)):
            # Compute polynomial of this degree
            poly_features = torch.pow(selected_features + 1e-8, degree)  # Avoid zero^0
            
            # Apply learnable weights for this degree
            degree_weights = self.degree_weights[degree_idx]
            weighted_poly = poly_features * degree_weights
            
            # Aggregate across features
            poly_aggregate = weighted_poly.sum(dim=1, keepdim=True)  # [batch_size, 1]
            polynomial_features.append(poly_aggregate)
            
            # Track contributions
            feature_contributions.append(torch.abs(degree_weights).mean().item())
        
        # Stack polynomial features
        poly_tensor = torch.cat(polynomial_features, dim=1)  # [batch_size, max_degree]
        
        # Apply combination network for final processing
        final_polynomials = self.combination_net(poly_tensor)
        
        # Metrics
        metrics = {
            "selection_weights_mean": selection_weights.mean(dim=0),
            "feature_contributions": torch.tensor(feature_contributions),
            "polynomial_magnitudes": torch.abs(final_polynomials).mean(dim=0),
        }
        
        return final_polynomials, metrics


class RatioFeatureDiscovery(nn.Module):
    """
    Automatically discovers meaningful ratios and differences between features.
    
    This module learns which feature pairs should form ratios or differences,
    which is particularly useful for engineering features like efficiency ratios,
    pressure differentials, etc.
    
    Args:
        num_features: Number of input features
        max_ratios: Maximum number of ratios to generate
        hidden_dim: Hidden dimension for selection networks
        dropout: Dropout rate
        eps: Small value to prevent division by zero
    """
    
    def __init__(
        self,
        num_features: int,
        max_ratios: int = 15,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.max_ratios = max_ratios
        self.eps = eps
        
        # Learnable selection of which features should form ratios
        self.ratio_selector = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features * num_features),
            nn.Sigmoid()
        )
        
        # Operation selector (ratio vs difference vs sum vs product)
        self.operation_selector = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 operations
            nn.Softmax(dim=-1)
        )
        
        # Feature transformation before operations
        self.feature_transform = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
        )
        
    def _safe_ratio(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        """Compute ratio with protection against division by zero."""
        return numerator / (torch.abs(denominator) + self.eps)
    
    def _safe_log_ratio(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        """Compute log ratio safely."""
        return torch.log(torch.abs(numerator) + self.eps) - torch.log(torch.abs(denominator) + self.eps)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Generate ratio and difference features.
        
        Args:
            x: Input features [batch_size, num_features]
            
        Returns:
            ratio_features: Generated ratio/difference features [batch_size, max_ratios]
            metrics: Dictionary with ratio statistics
        """
        batch_size, num_features = x.shape
        
        # Transform features before operations
        transformed_features = self.feature_transform(x)
        
        # Get selection weights for feature pairs
        selection_logits = self.ratio_selector(x)  # [batch_size, num_features^2]
        selection_weights = selection_logits.view(batch_size, num_features, num_features)
        
        # Get operation preferences
        op_weights = self.operation_selector(x)  # [batch_size, 4]
        
        # Generate ratio features
        ratio_features = []
        pair_importances = []
        
        # Get top feature pairs based on selection weights
        flat_selection = selection_weights.view(batch_size, -1)
        top_k = min(self.max_ratios, flat_selection.shape[1])
        top_values, top_indices = torch.topk(flat_selection, top_k, dim=1)
        
        for k in range(top_k):
            # Convert flat index to 2D coordinates
            batch_indices = torch.arange(batch_size, device=x.device)
            flat_idx = top_indices[:, k]
            i_indices = flat_idx // num_features
            j_indices = flat_idx % num_features
            
            # Get feature pairs
            feature_i = transformed_features[batch_indices, i_indices].unsqueeze(1)  # [batch_size, 1]
            feature_j = transformed_features[batch_indices, j_indices].unsqueeze(1)  # [batch_size, 1]
            
            # Compute different operations
            ratio = self._safe_ratio(feature_i, feature_j)
            log_ratio = self._safe_log_ratio(feature_i, feature_j)
            difference = feature_i - feature_j
            product = feature_i * feature_j
            
            # Combine operations based on learned weights
            operations = torch.cat([ratio, log_ratio, difference, product], dim=1)  # [batch_size, 4]
            combined_feature = (operations * op_weights).sum(dim=1, keepdim=True)  # [batch_size, 1]
            
            ratio_features.append(combined_feature)
            pair_importances.append(top_values[:, k].mean().item())
        
        # Pad if necessary
        while len(ratio_features) < self.max_ratios:
            ratio_features.append(torch.zeros(batch_size, 1, device=x.device))
            pair_importances.append(0.0)
        
        # Concatenate all ratio features
        ratio_tensor = torch.cat(ratio_features, dim=1)  # [batch_size, max_ratios]
        
        # Metrics
        metrics = {
            "pair_importances": torch.tensor(pair_importances),
            "operation_preferences": op_weights.mean(dim=0),
            "ratio_magnitudes": torch.abs(ratio_tensor).mean(dim=0),
            "selection_entropy": torch.mean(-torch.sum(selection_weights * torch.log(selection_weights + 1e-8), dim=[1, 2]))
        }
        
        return ratio_tensor, metrics


class CrossFeatureAttention(nn.Module):
    """
    Learns attention-based feature interactions.
    
    Uses multi-head attention to discover which features should attend to
    which other features, creating rich interaction representations.
    
    Args:
        num_features: Number of input features
        d_model: Model dimension for attention
        num_heads: Number of attention heads
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for positional encoding
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Feature embedding
        self.feature_embedding = nn.Linear(1, d_model)
        
        # Positional encoding for feature positions
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        self.position_embedding = nn.Embedding(num_features, d_model)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature-specific attention weights
        self.feature_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def _create_positional_encoding(self, max_seq_len: int, d_model: int) -> Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute attention-based feature interactions.
        
        Args:
            x: Input features [batch_size, num_features]
            
        Returns:
            attention_features: Attention-processed features [batch_size, num_features]
            metrics: Dictionary with attention statistics
        """
        batch_size, num_features = x.shape
        
        # Embed each feature individually
        feature_embeds = self.feature_embedding(x.unsqueeze(-1))  # [batch_size, num_features, d_model]
        
        # Add positional information
        positions = torch.arange(num_features, device=x.device)
        pos_embeds = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine embeddings
        combined_embeds = feature_embeds + pos_embeds
        combined_embeds = self.layer_norm(combined_embeds)
        combined_embeds = self.dropout(combined_embeds)
        
        # Apply cross-attention
        attended_features, attention_weights = self.cross_attention(
            combined_embeds, combined_embeds, combined_embeds
        )
        
        # Apply feature-specific attention weighting
        feature_weights = self.feature_attention(attended_features)  # [batch_size, num_features, 1]
        weighted_features = attended_features * feature_weights
        
        # Project to final features
        attention_features = self.output_projection(weighted_features).squeeze(-1)  # [batch_size, num_features]
        
        # Compute attention statistics
        attention_entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + 1e-8), dim=-1
        ).mean()
        
        metrics = {
            "attention_weights": attention_weights.mean(dim=0),  # Average across batch
            "attention_entropy": attention_entropy,
            "feature_weights": feature_weights.squeeze(-1).mean(dim=0),
            "feature_magnitudes": torch.abs(attention_features).mean(dim=0),
        }
        
        return attention_features, metrics


class NeuralFeatureEngineer(nn.Module):
    """
    Complete automatic feature engineering system that integrates all discovery components.
    
    This is the main class that combines feature importance gating, interaction discovery,
    polynomial features, ratio features, and attention-based interactions into a single
    cohesive feature engineering system.
    
    Args:
        num_features: Number of input features
        d_model: Model dimension (should match the rest of the architecture)
        max_interactions: Maximum number of pairwise interactions to discover
        max_ratios: Maximum number of ratio features to generate
        polynomial_degree: Maximum polynomial degree
        attention_heads: Number of attention heads for cross-feature attention
        dropout: Dropout rate
        fusion_strategy: How to combine engineered features ('concat', 'attention', 'learned')
        enable_feature_importance: Whether to use feature importance gating
        enable_interactions: Whether to discover pairwise interactions
        enable_polynomials: Whether to generate polynomial features
        enable_ratios: Whether to generate ratio features
        enable_attention: Whether to use cross-feature attention
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        max_interactions: int = 15,
        max_ratios: int = 10,
        polynomial_degree: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        fusion_strategy: str = "learned",
        enable_feature_importance: bool = True,
        enable_interactions: bool = True,
        enable_polynomials: bool = True,
        enable_ratios: bool = True,
        enable_attention: bool = True,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.fusion_strategy = fusion_strategy
        
        # Feature engineering components
        self.enable_feature_importance = enable_feature_importance
        self.enable_interactions = enable_interactions
        self.enable_polynomials = enable_polynomials
        self.enable_ratios = enable_ratios
        self.enable_attention = enable_attention
        
        # Initialize components
        if enable_feature_importance:
            self.importance_gate = FeatureImportanceGate(
                num_features=num_features,
                hidden_dim=min(64, d_model // 2),
                dropout=dropout
            )
        
        if enable_interactions:
            self.interaction_discovery = InteractionDiscovery(
                num_features=num_features,
                max_interactions=max_interactions,
                hidden_dim=min(64, d_model // 2),
                dropout=dropout
            )
        
        if enable_polynomials:
            self.polynomial_discovery = PolynomialFeatureDiscovery(
                num_features=num_features,
                max_degree=polynomial_degree,
                hidden_dim=min(64, d_model // 2),
                dropout=dropout
            )
        
        if enable_ratios:
            self.ratio_discovery = RatioFeatureDiscovery(
                num_features=num_features,
                max_ratios=max_ratios,
                hidden_dim=min(64, d_model // 2),
                dropout=dropout
            )
        
        if enable_attention:
            self.cross_attention = CrossFeatureAttention(
                num_features=num_features,
                d_model=min(d_model, 128),  # Cap attention dimension
                num_heads=attention_heads,
                dropout=dropout
            )
        
        # Calculate total engineered features
        total_features = num_features  # Original features
        if enable_interactions:
            total_features += max_interactions
        if enable_polynomials:
            total_features += polynomial_degree
        if enable_ratios:
            total_features += max_ratios
        if enable_attention:
            total_features += num_features
        
        # Feature fusion layer
        if fusion_strategy == "learned":
            self.feature_fusion = nn.Sequential(
                nn.Linear(total_features, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        elif fusion_strategy == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                d_model, num_heads=attention_heads, dropout=dropout, batch_first=True
            )
            self.feature_projection = nn.Linear(total_features, d_model)
        else:  # concat
            self.feature_projection = nn.Linear(total_features, d_model)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Adaptive gating for feature types
        self.feature_type_gates = nn.Sequential(
            nn.Linear(d_model, 5),  # 5 feature types
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, any]]:
        """
        Perform comprehensive feature engineering.
        
        Args:
            x: Input features [batch_size, num_features]
            
        Returns:
            engineered_features: Engineered feature representation [batch_size, d_model]
            metrics: Comprehensive metrics from all components
        """
        batch_size = x.shape[0]
        engineered_parts = []
        all_metrics = {}
        
        # 1. Feature importance gating
        if self.enable_feature_importance:
            important_features, importance_metrics = self.importance_gate(x)
            engineered_parts.append(important_features)
            all_metrics["importance"] = importance_metrics
            working_features = important_features
        else:
            working_features = x
            engineered_parts.append(x)
        
        # 2. Pairwise interaction discovery
        if self.enable_interactions:
            interaction_features, interaction_metrics = self.interaction_discovery(working_features)
            engineered_parts.append(interaction_features)
            all_metrics["interactions"] = interaction_metrics
        
        # 3. Polynomial feature generation
        if self.enable_polynomials:
            polynomial_features, polynomial_metrics = self.polynomial_discovery(working_features)
            engineered_parts.append(polynomial_features)
            all_metrics["polynomials"] = polynomial_metrics
        
        # 4. Ratio and difference features
        if self.enable_ratios:
            ratio_features, ratio_metrics = self.ratio_discovery(working_features)
            engineered_parts.append(ratio_features)
            all_metrics["ratios"] = ratio_metrics
        
        # 5. Cross-feature attention
        if self.enable_attention:
            attention_features, attention_metrics = self.cross_attention(working_features)
            engineered_parts.append(attention_features)
            all_metrics["attention"] = attention_metrics
        
        # 6. Combine all engineered features
        if len(engineered_parts) > 1:
            combined_features = torch.cat(engineered_parts, dim=1)
        else:
            combined_features = engineered_parts[0]
        
        # 7. Feature fusion
        if self.fusion_strategy == "learned":
            fused_features = self.feature_fusion(combined_features)
        elif self.fusion_strategy == "attention":
            # Project to consistent dimension first
            projected = self.feature_projection(combined_features).unsqueeze(1)  # [batch, 1, d_model]
            attended, _ = self.fusion_attention(projected, projected, projected)
            fused_features = attended.squeeze(1)  # [batch, d_model]
        else:  # concat
            fused_features = self.feature_projection(combined_features)
        
        # 8. Final processing
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        
        # 9. Adaptive feature type gating
        type_weights = self.feature_type_gates(fused_features)
        all_metrics["feature_type_weights"] = type_weights.mean(dim=0)
        
        # 10. Compile comprehensive metrics
        summary_metrics = {
            "total_engineered_features": combined_features.shape[1],
            "feature_magnitude": torch.norm(fused_features, dim=1).mean(),
            "component_metrics": all_metrics,
            "enabled_components": {
                "importance": self.enable_feature_importance,
                "interactions": self.enable_interactions,
                "polynomials": self.enable_polynomials,
                "ratios": self.enable_ratios,
                "attention": self.enable_attention,
            }
        }
        
        return fused_features, summary_metrics
    
    def get_feature_importance_summary(self) -> Dict[str, Tensor]:
        """Get a summary of discovered feature importances and interactions."""
        summary = {}
        
        if self.enable_feature_importance and hasattr(self.importance_gate, 'feature_means'):
            summary["feature_statistics"] = {
                "means": self.importance_gate.feature_means,
                "stds": self.importance_gate.feature_stds,
            }
        
        if self.enable_interactions:
            interaction_weights = torch.sigmoid(self.interaction_discovery.interaction_weights)
            interaction_weights = interaction_weights * self.interaction_discovery.triu_mask
            summary["top_interactions"] = torch.topk(
                interaction_weights[interaction_weights > 0], 
                min(10, (interaction_weights > 0).sum().item())
            )
        
        if self.enable_polynomials:
            degree_contributions = []
            for degree_weight in self.polynomial_discovery.degree_weights:
                degree_contributions.append(torch.abs(degree_weight).mean())
            summary["polynomial_contributions"] = torch.stack(degree_contributions)
        
        return summary
    
    def explain_predictions(self, x: Tensor, top_k: int = 5) -> Dict[str, any]:
        """
        Provide explanations for predictions by showing which engineered features matter most.
        
        Args:
            x: Input features [batch_size, num_features]
            top_k: Number of top features to explain
            
        Returns:
            explanations: Dictionary with feature explanations
        """
        with torch.no_grad():
            engineered_features, metrics = self.forward(x)
            
            explanations = {
                "input_features": x,
                "engineered_features": engineered_features,
                "feature_magnitudes": torch.abs(engineered_features).mean(dim=0),
                "metrics": metrics,
            }
            
            # Add component-specific explanations
            if self.enable_feature_importance:
                explanations["feature_importance"] = metrics["component_metrics"]["importance"][1]
            
            if self.enable_interactions:
                interaction_metrics = metrics["component_metrics"]["interactions"]
                explanations["top_interactions"] = {
                    "pairs": interaction_metrics["selected_pairs"],
                    "importances": interaction_metrics["interaction_importances"],
                }
            
            return explanations