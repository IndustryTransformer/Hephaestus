# Two-Stage Training for Anomaly Detection

This document explains the two-stage training approach implemented in `anomaly_encoder_decoder.py`.

## Overview

The two-stage training approach consists of:

1. **Pre-training Stage**: Learn general representations using masked modeling
2. **Fine-tuning Stage**: Specialize for anomaly classification with frozen encoder

## How It Works

### Stage 1: Pre-training with Masked Modeling

- Randomly masks 20% of numeric and categorical features
- Learns to reconstruct the masked values
- Uses MSE loss for numeric features and Cross-Entropy for categorical features
- Creates robust feature representations

### Stage 2: Fine-tuning for Classification

- Loads the pre-trained encoder and freezes its parameters
- Only trains the classification head
- Uses a lower learning rate (1/10 of pre-training rate)
- Focuses on learning the specific anomaly patterns

## Usage

### Training

In `anomaly_encoder_decoder.py`, set:

```python
USE_TWO_STAGE = True  # Enable two-stage training
```

Then run the notebook/script. The training will:

1. Pre-train the model for `MAX_EPOCHS_PRETRAIN` epochs
2. Save the best pre-trained model
3. Load and freeze the pre-trained encoder
4. Fine-tune for `MAX_EPOCHS_FINETUNE` epochs
5. Save the final model with both encoder and classifier weights

### Model Checkpoints

The training saves multiple checkpoints:

- **Pre-training**: `checkpoints/pretrain_<model_name>_<epoch>_<val_loss>.ckpt`
- **Fine-tuning**: `checkpoints/finetune_<model_name>_<epoch>_<val_loss>.ckpt`
- **Final model**: `checkpoints/two_stage_model_<model_name>_acc_<accuracy>.pt`

### Loading a Trained Model

Use the provided `load_pretrained_model.py` script:

```python
from load_pretrained_model import load_two_stage_model, predict_anomalies

# Load the model
model, config, events_names = load_two_stage_model(
    "checkpoints/two_stage_model_two_stage_encoder_decoder_acc_0.850.pt"
)

# Use for predictions
predictions, targets = predict_anomalies(model, test_dataloader)
```

## Benefits

1. **Better Generalization**: Pre-training helps the model learn general patterns
2. **Reduced Overfitting**: Frozen encoder prevents overfitting during fine-tuning
3. **Faster Convergence**: Fine-tuning typically needs fewer epochs
4. **Transfer Learning**: Pre-trained encoders can be reused for other tasks

## Hyperparameters

Key hyperparameters to tune:

- `MASK_PROBABILITY`: Percentage of features to mask (default: 0.2)
- `MAX_EPOCHS_PRETRAIN`: Pre-training epochs (default: 10)
- `MAX_EPOCHS_FINETUNE`: Fine-tuning epochs (default: 20)
- `LEARNING_RATE`: Base learning rate (fine-tuning uses 1/10 of this)
- `D_MODEL`: Model dimension (default: 32)
- `N_HEADS`: Number of attention heads (default: 4)

## Comparison with Single-Stage

To compare with single-stage training, set:

```python
USE_TWO_STAGE = False  # Use original single-stage approach
```

This will train the model end-to-end without pre-training.