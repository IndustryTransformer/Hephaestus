import torch
import torch.nn as nn
import pytorch_lightning as L
import numpy as np
from typing import Dict, Any, Optional

from .single_row_models import TabularEncoder, AttentionPooling
from .feature_interactions import InteractionAwareEncoder, FeatureImportanceFilter
from .model_data_classes import SingleRowConfig


class EnhancedTabularRegressor(L.LightningModule):
    """Enhanced tabular regressor with automated feature interactions."""
    
    def __init__(
        self,
        model_config: SingleRowConfig,
        d_model: int = 64,
        n_heads: int = 4,
        lr: float = 1e-4,
        interaction_config: Optional[Dict[str, Any]] = None,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.8,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_config = model_config
        self.d_model = d_model
        self.n_heads = n_heads
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        
        # Default interaction configuration
        if interaction_config is None:
            interaction_config = {
                'use_lowrank_cross': True,
                'use_cat_interactions': True,
                'use_crossnet': True,
                'lowrank_rank': min(32, d_model // 2),
                'crossnet_layers': 2,
                'hash_size': 1000,
                'dropout': 0.1
            }
        
        self.interaction_config = interaction_config
        
        # Base encoder
        self.base_encoder = TabularEncoder(
            model_config, d_model=d_model, n_heads=n_heads
        )
        
        # Enhanced encoder with interactions
        self.enhanced_encoder = InteractionAwareEncoder(
            base_encoder=self.base_encoder,
            d_model=d_model,
            n_numeric_cols=model_config.n_numeric_cols,
            n_cat_cols=model_config.n_cat_cols,
            interaction_config=interaction_config
        )
        
        # Pooling and regression head
        self.pooling = AttentionPooling(d_model)
        self.dropout = nn.Dropout(interaction_config.get('dropout', 0.1))
        
        # More sophisticated regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(interaction_config.get('dropout', 0.1)),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(interaction_config.get('dropout', 0.1) * 0.5),
            nn.Linear(d_model, 1),
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
    def forward(self, batch):
        """Forward pass through the enhanced model."""
        num_inputs = batch.inputs.numeric
        cat_inputs = batch.inputs.categorical
        
        # Get enhanced features with interactions
        enhanced_features = self.enhanced_encoder(num_inputs, cat_inputs)
        
        # Pool and predict
        pooled = self.pooling(enhanced_features)
        pooled = self.dropout(pooled)
        
        # Regression
        output = self.regressor(pooled)
        return output
        
    def training_step(self, batch, batch_idx):
        """Training step."""
        predictions = self(batch)
        loss = self.loss_fn(predictions.squeeze(), batch.target.squeeze())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        predictions = self(batch)
        loss = self.loss_fn(predictions.squeeze(), batch.target.squeeze())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def predict_step(self, batch, batch_idx=None):
        """Prediction step."""
        return self(batch)
        
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class AdaptiveFeatureEngineering:
    """Adaptive feature engineering using importance-based filtering."""
    
    def __init__(self, model_config: SingleRowConfig):
        self.model_config = model_config
        self.feature_filter = FeatureImportanceFilter()
        self.important_pairs = []
        
    def analyze_features(self, train_df, target_col: str, top_pairs: int = 20):
        """Analyze features and identify important interactions."""
        # Prepare data
        X_numeric = train_df.select_dtypes(include=['number']).values
        y = train_df[target_col].values
        
        numeric_feature_names = list(train_df.select_dtypes(include=['number']).columns)
        if target_col in numeric_feature_names:
            numeric_feature_names.remove(target_col)
            # Remove target column from X
            target_idx = list(train_df.select_dtypes(include=['number']).columns).index(target_col)
            X_numeric = np.delete(X_numeric, target_idx, axis=1)
        
        # Get important feature pairs
        self.important_pairs = self.feature_filter.get_important_pairs(
            X_numeric, y, numeric_feature_names, top_pairs=top_pairs
        )
        
        print(f"Identified {len(self.important_pairs)} important feature pairs")
        
        return self.important_pairs
        
    def get_interaction_config(self, aggressive: bool = False) -> Dict[str, Any]:
        """Get interaction configuration based on feature analysis."""
        config = {
            'use_lowrank_cross': True,
            'use_cat_interactions': self.model_config.n_cat_cols > 1,
            'use_crossnet': True,
            'lowrank_rank': 32 if aggressive else 16,
            'crossnet_layers': 3 if aggressive else 2,
            'hash_size': 2000 if aggressive else 1000,
            'dropout': 0.1 if aggressive else 0.2
        }
        
        return config


def create_enhanced_model(
    model_config: SingleRowConfig,
    train_df=None,
    target_col: str = 'target',
    d_model: int = 128,
    n_heads: int = 4,
    lr: float = 1e-4,
    aggressive_interactions: bool = False
) -> EnhancedTabularRegressor:
    """
    Create an enhanced tabular regressor with automated feature engineering.
    
    Args:
        model_config: Configuration for the model
        train_df: Training dataframe for feature analysis (optional)
        target_col: Name of target column
        d_model: Model dimension
        n_heads: Number of attention heads
        lr: Learning rate
        aggressive_interactions: Whether to use more aggressive interaction settings
        
    Returns:
        EnhancedTabularRegressor instance
    """
    # Analyze features if training data is provided
    interaction_config = None
    if train_df is not None:
        adaptive_fe = AdaptiveFeatureEngineering(model_config)
        try:
            adaptive_fe.analyze_features(train_df, target_col)
            interaction_config = adaptive_fe.get_interaction_config(aggressive_interactions)
            print(f"Using adaptive interaction config: {interaction_config}")
        except Exception as e:
            print(f"Feature analysis failed: {e}. Using default config.")
    
    # Create model
    model = EnhancedTabularRegressor(
        model_config=model_config,
        d_model=d_model,
        n_heads=n_heads,
        lr=lr,
        interaction_config=interaction_config,
    )
    
    return model