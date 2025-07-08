"""
Mixture of Experts (MoE) components for enhanced feature interactions in tabular data.

This module provides the core MoE infrastructure including:
- MoELayer: Main MoE layer with routing and expert coordination
- ExpertRouter: Learned routing mechanism with load balancing
- GatingNetwork: Neural network for expert selection
- Expert classes: Specialized networks for different feature types
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GatingNetwork(nn.Module):
    """
    Neural network for routing inputs to experts based on learned patterns.
    
    Args:
        input_dim: Dimension of input features
        num_experts: Number of experts to route to
        hidden_dim: Hidden dimension for gating network
        dropout: Dropout rate for regularization
        feature_type_aware: Whether to use feature type information in routing
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        feature_type_aware: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.feature_type_aware = feature_type_aware
        
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 32)
            
        # Main gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts),
        )
        
        # Feature type embedding if enabled
        if feature_type_aware:
            self.feature_type_embedding = nn.Embedding(3, hidden_dim // 4)  # numeric, categorical, interaction
            self.type_projection = nn.Linear(hidden_dim // 4, num_experts)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(
        self, 
        x: Tensor, 
        feature_types: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute gating weights for expert selection.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            feature_types: Feature type indicators [batch_size, seq_len]
                          0=numeric, 1=categorical, 2=interaction
                          
        Returns:
            Gating weights [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute main gating weights
        gate_logits = self.gate_network(x)  # [batch_size, seq_len, num_experts]
        
        # Add feature type bias if enabled
        if self.feature_type_aware and feature_types is not None:
            type_embed = self.feature_type_embedding(feature_types)  # [batch_size, seq_len, hidden_dim//4]
            type_bias = self.type_projection(type_embed)  # [batch_size, seq_len, num_experts]
            gate_logits = gate_logits + type_bias
            
        # Apply softmax to get probabilities
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        return gate_weights


class ExpertRouter(nn.Module):
    """
    Handles routing logic and load balancing for MoE.
    
    Args:
        num_experts: Number of experts
        capacity_factor: Controls expert capacity (higher = more capacity)
        top_k: Number of experts to route to per token
        use_auxiliary_loss: Whether to use auxiliary loss for load balancing
    """
    
    def __init__(
        self,
        num_experts: int,
        capacity_factor: float = 1.0,
        top_k: int = 2,
        use_auxiliary_loss: bool = True,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.top_k = top_k
        self.use_auxiliary_loss = use_auxiliary_loss
        
        # For auxiliary loss computation
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self.register_buffer("gate_sums", torch.zeros(num_experts))
        
    def forward(
        self, 
        gate_weights: Tensor,
        training: bool = True
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Route inputs to experts based on gate weights.
        
        Args:
            gate_weights: Gating weights [batch_size, seq_len, num_experts]
            training: Whether in training mode
            
        Returns:
            expert_weights: Weights for selected experts [batch_size, seq_len, top_k]
            expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            auxiliary_loss: Load balancing loss (if enabled)
        """
        batch_size, seq_len, num_experts = gate_weights.shape
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            gate_weights, self.top_k, dim=-1
        )
        
        # Renormalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary loss for load balancing
        auxiliary_loss = None
        if self.use_auxiliary_loss and training:
            auxiliary_loss = self._compute_auxiliary_loss(gate_weights)
            
        # Update statistics for monitoring
        if training:
            self._update_statistics(gate_weights, expert_indices)
            
        return expert_weights, expert_indices, auxiliary_loss
    
    def _compute_auxiliary_loss(self, gate_weights: Tensor) -> Tensor:
        """Compute auxiliary loss for load balancing."""
        # Compute fraction of tokens routed to each expert
        gate_mean = gate_weights.mean(dim=[0, 1])  # [num_experts]
        
        # Compute entropy-based loss to encourage uniform distribution
        entropy = -torch.sum(gate_mean * torch.log(gate_mean + 1e-8))
        max_entropy = math.log(self.num_experts)
        
        # Auxiliary loss encourages uniform expert usage
        auxiliary_loss = 1.0 - (entropy / max_entropy)
        
        return auxiliary_loss
    
    def _update_statistics(self, gate_weights: Tensor, expert_indices: Tensor):
        """Update expert usage statistics."""
        # Update expert counts
        expert_mask = F.one_hot(expert_indices, self.num_experts).float()
        expert_counts = expert_mask.sum(dim=[0, 1, 2])  # [num_experts]
        
        # Moving average update
        momentum = 0.9
        self.expert_counts = momentum * self.expert_counts + (1 - momentum) * expert_counts
        
        # Update gate sums
        gate_sums = gate_weights.sum(dim=[0, 1])  # [num_experts]
        self.gate_sums = momentum * self.gate_sums + (1 - momentum) * gate_sums
        
    def get_expert_utilization(self) -> Dict[str, Tensor]:
        """Get expert utilization statistics."""
        return {
            "expert_counts": self.expert_counts,
            "gate_sums": self.gate_sums,
            "utilization_variance": self.expert_counts.var(),
            "load_balance": self.gate_sums.std() / (self.gate_sums.mean() + 1e-8)
        }


class MoELayer(nn.Module):
    """
    Main Mixture of Experts layer with routing and expert coordination.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for experts
        num_experts: Number of experts
        expert_types: List of expert types ['numeric', 'categorical', 'interaction']
        top_k: Number of experts to use per token
        dropout: Dropout rate
        activation: Activation function
        use_auxiliary_loss: Whether to use auxiliary loss
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        expert_types: Optional[List[str]] = None,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        use_auxiliary_loss: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_auxiliary_loss = use_auxiliary_loss
        
        # Default expert types
        if expert_types is None:
            expert_types = ['numeric', 'categorical', 'interaction'] * (num_experts // 3 + 1)
            expert_types = expert_types[:num_experts]
        
        self.expert_types = expert_types
        
        # Gating network
        self.gate = GatingNetwork(
            input_dim=input_dim,
            num_experts=num_experts,
            dropout=dropout,
            feature_type_aware=True,
        )
        
        # Router
        self.router = ExpertRouter(
            num_experts=num_experts,
            top_k=top_k,
            use_auxiliary_loss=use_auxiliary_loss,
        )
        
        # Experts (will be populated by subclasses)
        self.experts = nn.ModuleList()
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
    def add_expert(self, expert: nn.Module):
        """Add an expert to the MoE layer."""
        self.experts.append(expert)
        
    def forward(
        self, 
        x: Tensor, 
        feature_types: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            feature_types: Feature type indicators [batch_size, seq_len]
            
        Returns:
            output: Processed tensor [batch_size, seq_len, input_dim]
            metrics: Dictionary of metrics and losses
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Compute gating weights
        gate_weights = self.gate(x, feature_types)  # [batch_size, seq_len, num_experts]
        
        # Route to experts
        expert_weights, expert_indices, aux_loss = self.router(
            gate_weights, training=self.training
        )
        
        # Process through experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)  # [batch_size, seq_len, hidden_dim]
            expert_outputs.append(expert_output)
            
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-2)  # [batch_size, seq_len, num_experts, hidden_dim]
        
        # Combine expert outputs using routing weights
        output = torch.zeros(
            batch_size, seq_len, self.hidden_dim, 
            device=x.device, dtype=x.dtype
        )
        
        for k in range(self.top_k):
            expert_idx = expert_indices[:, :, k]  # [batch_size, seq_len]
            expert_weight = expert_weights[:, :, k:k+1]  # [batch_size, seq_len, 1]
            
            # Gather expert outputs
            expert_output = torch.gather(
                expert_outputs, 
                dim=2, 
                index=expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
            ).squeeze(2)  # [batch_size, seq_len, hidden_dim]
            
            # Weighted sum
            output = output + expert_weight * expert_output
            
        # Apply activation and output projection
        output = self.activation(output)
        output = self.output_proj(output)
        
        # Collect metrics
        metrics = {
            "auxiliary_loss": aux_loss,
            "gate_weights": gate_weights,
            "expert_weights": expert_weights,
            "expert_indices": expert_indices,
        }
        
        # Add expert utilization stats
        metrics.update(self.router.get_expert_utilization())
        
        return output, metrics


class NumericExpert(nn.Module):
    """
    Expert specialized for processing numeric features.
    
    Optimized for continuous data patterns, statistical operations,
    and numerical relationships.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for processing
        dropout: Dropout rate
        activation: Activation function
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Specialized architecture for numeric processing
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            self._get_activation(activation),
            nn.Dropout(dropout),
            
            # Statistical feature extraction
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            self._get_activation(activation),
            nn.Dropout(dropout),
            
            # Final transformation
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Residual connection for numeric features
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Process numeric features with specialized numeric operations.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Processed tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for batch normalization
        x_reshaped = x.reshape(-1, self.input_dim)
        
        # Apply numeric processing
        processed = self.feature_processor(x_reshaped)
        
        # Residual connection
        residual = self.residual_proj(x_reshaped)
        output = processed + residual
        
        # Reshape back and apply layer norm
        output = output.reshape(batch_size, seq_len, self.hidden_dim)
        output = self.layer_norm(output)
        
        return output


class CategoricalExpert(nn.Module):
    """
    Expert specialized for processing categorical features.
    
    Optimized for discrete patterns, relational structures,
    and categorical interactions.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for processing
        dropout: Dropout rate
        activation: Activation function
        use_attention: Whether to use self-attention for categorical relationships
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Specialized architecture for categorical processing
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            
            # Categorical relationship modeling
            nn.Linear(hidden_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            
            # Final transformation
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Self-attention for categorical relationships
        if use_attention:
            self.self_attention = nn.MultiheadAttention(
                hidden_dim, 
                num_heads=max(1, hidden_dim // 64),
                dropout=dropout,
                batch_first=True
            )
            
        # Residual connection
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Process categorical features with specialized categorical operations.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Processed tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply categorical processing
        processed = self.feature_processor(x)
        
        # Apply self-attention for categorical relationships
        if self.use_attention:
            attended, _ = self.self_attention(processed, processed, processed)
            processed = processed + attended
            
        # Residual connection
        residual = self.residual_proj(x)
        output = processed + residual
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class InteractionExpert(nn.Module):
    """
    Expert specialized for modeling feature interactions.
    
    Focuses on cross-feature relationships, especially between
    numeric and categorical features.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for processing
        dropout: Dropout rate
        activation: Activation function
        interaction_type: Type of interaction modeling ('bilinear', 'multiplicative', 'additive')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        interaction_type: str = "bilinear",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.interaction_type = interaction_type
        
        # Feature transformation layers
        self.feature_proj_1 = nn.Linear(input_dim, hidden_dim)
        self.feature_proj_2 = nn.Linear(input_dim, hidden_dim)
        
        # Interaction modeling
        if interaction_type == "bilinear":
            self.interaction_layer = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        elif interaction_type == "multiplicative":
            self.interaction_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif interaction_type == "additive":
            self.interaction_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")
            
        # Cross-attention for feature interactions
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=max(1, hidden_dim // 64),
            dropout=dropout,
            batch_first=True
        )
        
        # Final processing
        self.final_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Process features with specialized interaction modeling.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Processed tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to two different feature spaces
        feat_1 = self.feature_proj_1(x)  # [batch_size, seq_len, hidden_dim]
        feat_2 = self.feature_proj_2(x)  # [batch_size, seq_len, hidden_dim]
        
        # Model interactions
        if self.interaction_type == "bilinear":
            # Bilinear interaction
            interaction = self.interaction_layer(feat_1, feat_2)
        elif self.interaction_type == "multiplicative":
            # Element-wise multiplication then MLP
            interaction = self.interaction_layer(torch.cat([feat_1 * feat_2, feat_1 + feat_2], dim=-1))
        elif self.interaction_type == "additive":
            # Additive interaction
            interaction = self.interaction_layer(feat_1 + feat_2)
            
        # Cross-attention for enhanced interactions
        cross_attended, _ = self.cross_attention(feat_1, feat_2, feat_2)
        interaction = interaction + cross_attended
        
        # Final processing
        output = self.final_processor(interaction)
        
        # Residual connection
        residual = self.residual_proj(x)
        output = output + residual
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class TabularMoELayer(MoELayer):
    """
    Specialized MoE layer for tabular data with categorical variable support.
    
    This layer automatically creates the appropriate expert types and
    handles feature type routing.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for experts
        num_experts: Number of experts (should be divisible by 3 for balanced expert types)
        top_k: Number of experts to use per token
        dropout: Dropout rate
        activation: Activation function
        use_auxiliary_loss: Whether to use auxiliary loss
        expert_balance: Balance of expert types [numeric_ratio, categorical_ratio, interaction_ratio]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        use_auxiliary_loss: bool = True,
        expert_balance: Optional[List[float]] = None,
    ):
        # Calculate expert distribution
        if expert_balance is None:
            expert_balance = [0.4, 0.3, 0.3]  # 40% numeric, 30% categorical, 30% interaction
            
        num_numeric = max(1, int(num_experts * expert_balance[0]))
        num_categorical = max(1, int(num_experts * expert_balance[1]))
        num_interaction = max(1, num_experts - num_numeric - num_categorical)
        
        # Create expert type list
        expert_types = (['numeric'] * num_numeric + 
                       ['categorical'] * num_categorical + 
                       ['interaction'] * num_interaction)
        
        # Initialize base MoE layer
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            expert_types=expert_types,
            top_k=top_k,
            dropout=dropout,
            activation=activation,
            use_auxiliary_loss=use_auxiliary_loss,
        )
        
        # Create specialized experts
        for expert_type in expert_types:
            if expert_type == 'numeric':
                expert = NumericExpert(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                )
            elif expert_type == 'categorical':
                expert = CategoricalExpert(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                )
            elif expert_type == 'interaction':
                expert = InteractionExpert(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                )
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")
                
            self.add_expert(expert)


class FeatureTypeAwareRouter(nn.Module):
    """
    Advanced routing system that considers feature types for expert selection.
    
    This router implements a multi-level routing strategy:
    1. Primary routing based on feature type (numeric/categorical/interaction)
    2. Secondary routing based on learned feature patterns
    3. Load balancing to ensure expert utilization
    
    Args:
        input_dim: Input feature dimension
        num_experts: Number of experts
        expert_types: List of expert types
        feature_type_weight: Weight for feature type bias in routing
        temperature: Temperature for softmax (lower = more decisive routing)
        load_balance_weight: Weight for load balancing loss
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        expert_types: List[str],
        feature_type_weight: float = 1.0,
        temperature: float = 1.0,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_types = expert_types
        self.feature_type_weight = feature_type_weight
        self.temperature = temperature
        self.load_balance_weight = load_balance_weight
        
        # Create expert type mapping
        self.type_to_experts = {
            'numeric': [],
            'categorical': [],
            'interaction': []
        }
        
        for i, expert_type in enumerate(expert_types):
            if expert_type in self.type_to_experts:
                self.type_to_experts[expert_type].append(i)
        
        # Primary routing network (feature type aware)
        self.primary_router = GatingNetwork(
            input_dim=input_dim,
            num_experts=num_experts,
            feature_type_aware=True,
        )
        
        # Secondary routing network (pattern aware)
        self.secondary_router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_experts),
        )
        
        # Feature type classifier
        self.feature_type_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 3),  # 3 feature types
        )
        
        # Load balancing tracking
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("total_tokens", torch.tensor(0.0))
        
    def forward(
        self, 
        x: Tensor, 
        feature_types: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Perform multi-level routing with feature type awareness.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            feature_types: Known feature types [batch_size, seq_len]
                          0=numeric, 1=categorical, 2=interaction
                          
        Returns:
            routing_weights: Expert routing weights [batch_size, seq_len, num_experts]
            predicted_types: Predicted feature types [batch_size, seq_len, 3]
            metrics: Dictionary of routing metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Predict feature types if not provided
        if feature_types is None:
            type_logits = self.feature_type_classifier(x)
            predicted_types = F.softmax(type_logits, dim=-1)
            feature_types = torch.argmax(predicted_types, dim=-1)
        else:
            predicted_types = F.one_hot(feature_types, 3).float()
            
        # Primary routing with feature type awareness
        primary_weights = self.primary_router(x, feature_types)
        
        # Secondary routing for pattern-based selection
        secondary_logits = self.secondary_router(x)
        secondary_weights = F.softmax(secondary_logits / self.temperature, dim=-1)
        
        # Combine routing weights
        routing_weights = (
            self.feature_type_weight * primary_weights + 
            (1 - self.feature_type_weight) * secondary_weights
        )
        
        # Apply feature type biases
        routing_weights = self._apply_feature_type_bias(
            routing_weights, feature_types
        )
        
        # Renormalize
        routing_weights = F.softmax(routing_weights / self.temperature, dim=-1)
        
        # Update usage statistics
        if self.training:
            self._update_usage_stats(routing_weights)
            
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(routing_weights)
        
        # Prepare metrics
        metrics = {
            "load_balance_loss": load_balance_loss,
            "expert_usage": self.expert_usage.clone(),
            "routing_entropy": self._compute_routing_entropy(routing_weights),
            "feature_type_accuracy": self._compute_feature_type_accuracy(
                predicted_types, feature_types
            ),
        }
        
        return routing_weights, predicted_types, metrics
    
    def _apply_feature_type_bias(
        self, 
        routing_weights: Tensor, 
        feature_types: Tensor
    ) -> Tensor:
        """Apply feature type biases to routing weights."""
        batch_size, seq_len, num_experts = routing_weights.shape
        
        # Create bias tensor
        bias = torch.zeros_like(routing_weights)
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                feature_type = feature_types[batch_idx, seq_idx].item()
                
                if feature_type == 0:  # Numeric
                    expert_indices = self.type_to_experts.get('numeric', [])
                elif feature_type == 1:  # Categorical
                    expert_indices = self.type_to_experts.get('categorical', [])
                elif feature_type == 2:  # Interaction
                    expert_indices = self.type_to_experts.get('interaction', [])
                else:
                    expert_indices = []
                
                # Apply positive bias to matching experts
                for expert_idx in expert_indices:
                    if expert_idx < num_experts:
                        bias[batch_idx, seq_idx, expert_idx] = 0.5
                        
        return routing_weights + bias
    
    def _update_usage_stats(self, routing_weights: Tensor):
        """Update expert usage statistics."""
        # Update total tokens
        self.total_tokens += routing_weights.numel() // self.num_experts
        
        # Update expert usage (moving average)
        current_usage = routing_weights.sum(dim=[0, 1])
        momentum = 0.99
        self.expert_usage = momentum * self.expert_usage + (1 - momentum) * current_usage
        
    def _compute_load_balance_loss(self, routing_weights: Tensor) -> Tensor:
        """Compute load balancing loss to encourage uniform expert usage."""
        # Compute fraction of tokens routed to each expert
        expert_fractions = routing_weights.mean(dim=[0, 1])
        
        # Compute coefficient of variation (std/mean) as load imbalance measure
        target_fraction = 1.0 / self.num_experts
        imbalance = torch.var(expert_fractions) / (target_fraction ** 2)
        
        return self.load_balance_weight * imbalance
    
    def _compute_routing_entropy(self, routing_weights: Tensor) -> Tensor:
        """Compute routing entropy to measure decision confidence."""
        # Compute entropy across experts for each token
        log_weights = torch.log(routing_weights + 1e-8)
        entropy = -torch.sum(routing_weights * log_weights, dim=-1)
        
        return entropy.mean()
    
    def _compute_feature_type_accuracy(
        self, 
        predicted_types: Tensor, 
        true_types: Tensor
    ) -> Tensor:
        """Compute accuracy of feature type prediction."""
        pred_labels = torch.argmax(predicted_types, dim=-1)
        accuracy = (pred_labels == true_types).float().mean()
        
        return accuracy
    
    def get_routing_stats(self) -> Dict[str, Tensor]:
        """Get detailed routing statistics."""
        return {
            "expert_usage": self.expert_usage,
            "total_tokens": self.total_tokens,
            "usage_variance": self.expert_usage.var(),
            "usage_entropy": -(self.expert_usage * torch.log(self.expert_usage + 1e-8)).sum(),
        }


class AdaptiveTabularMoELayer(TabularMoELayer):
    """
    Adaptive MoE layer that dynamically adjusts expert routing based on data characteristics.
    
    This layer extends TabularMoELayer with:
    - Adaptive routing based on feature statistics
    - Dynamic expert balancing
    - Categorical variable awareness
    - Performance-based expert selection
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for experts
        num_experts: Number of experts
        top_k: Number of experts to use per token
        dropout: Dropout rate
        activation: Activation function
        expert_balance: Expert type balance ratios
        adaptive_routing: Whether to use adaptive routing
        categorical_aware: Whether to use categorical-aware processing
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        expert_balance: Optional[List[float]] = None,
        adaptive_routing: bool = True,
        categorical_aware: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            activation=activation,
            use_auxiliary_loss=True,
            expert_balance=expert_balance,
        )
        
        self.adaptive_routing = adaptive_routing
        self.categorical_aware = categorical_aware
        
        # Replace the basic router with feature-type-aware router
        if adaptive_routing:
            self.adaptive_router = FeatureTypeAwareRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                expert_types=self.expert_types,
            )
            
        # Categorical feature detector
        if categorical_aware:
            self.categorical_detector = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid(),
            )
            
    def forward(
        self, 
        x: Tensor, 
        feature_types: Optional[Tensor] = None,
        categorical_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with adaptive routing and categorical awareness.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            feature_types: Feature type indicators [batch_size, seq_len]
            categorical_mask: Mask for categorical features [batch_size, seq_len]
            
        Returns:
            output: Processed tensor [batch_size, seq_len, input_dim]
            metrics: Dictionary of metrics and losses
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Detect categorical features if not provided
        if categorical_mask is None and self.categorical_aware:
            categorical_scores = self.categorical_detector(x)
            categorical_mask = (categorical_scores > 0.5).squeeze(-1)
            
        # Use adaptive routing if enabled
        if self.adaptive_routing:
            routing_weights, predicted_types, routing_metrics = self.adaptive_router(
                x, feature_types
            )
            
            # Route to top-k experts
            expert_weights, expert_indices = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
            
            # Compute auxiliary loss
            aux_loss = routing_metrics["load_balance_loss"]
            
        else:
            # Use standard routing
            gate_weights = self.gate(x, feature_types)
            expert_weights, expert_indices, aux_loss = self.router(
                gate_weights, training=self.training
            )
            routing_metrics = {}
            
        # Process through experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            expert_outputs.append(expert_output)
            
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-2)
        
        # Combine expert outputs
        output = torch.zeros(
            batch_size, seq_len, self.hidden_dim,
            device=x.device, dtype=x.dtype
        )
        
        for k in range(self.top_k):
            expert_idx = expert_indices[:, :, k]
            expert_weight = expert_weights[:, :, k:k+1]
            
            # Gather expert outputs
            expert_output = torch.gather(
                expert_outputs,
                dim=2,
                index=expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
            ).squeeze(2)
            
            # Weighted sum
            output = output + expert_weight * expert_output
            
        # Apply activation and output projection
        output = self.activation(output)
        output = self.output_proj(output)
        
        # Collect metrics
        metrics = {
            "auxiliary_loss": aux_loss,
            "expert_weights": expert_weights,
            "expert_indices": expert_indices,
        }
        
        # Add routing metrics
        metrics.update(routing_metrics)
        
        # Add categorical detection metrics
        if self.categorical_aware and categorical_mask is not None:
            metrics["categorical_ratio"] = categorical_mask.float().mean()
            
        return output, metrics