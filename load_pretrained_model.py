"""
Example script showing how to load and use a pre-trained two-stage model.
"""

import torch
from hephaestus.timeseries_models.encoder_decoder import (
    TabularEncoderDecoder,
)
from hephaestus.timeseries_models.models import TimeSeriesTransformer


def load_two_stage_model(
    checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load a two-stage trained model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file
        device: Device to load the model on

    Returns:
        model: The loaded TabularEncoderDecoder model
        config: The TimeSeriesConfig used for training
        events_names: Dictionary mapping event indices to names
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract components
    config = checkpoint["config"]
    d_model = checkpoint["d_model"]
    n_heads = checkpoint["n_heads"]
    events_names = checkpoint["events_names"]

    # Create encoder
    encoder = TimeSeriesTransformer(
        config=config,
        d_model=d_model,
        n_heads=n_heads,
    )

    # Load encoder weights
    encoder.load_state_dict(checkpoint["pretrained_encoder_state_dict"])

    # Create full model with pre-trained encoder
    model = TabularEncoderDecoder(
        config,
        d_model=d_model,
        n_heads=n_heads,
        classification_values=list(events_names.values()),
        pretrained_encoder=encoder,
    )

    # Load full model weights
    model.load_state_dict(checkpoint["full_model_state_dict"])

    # Move to device
    model = model.to(device)
    model.eval()

    return model, config, events_names


def predict_anomalies(
    model, data_loader, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Use the model to predict anomalies on new data.

    Args:
        model: The loaded TabularEncoderDecoder model
        data_loader: DataLoader with the data to predict on
        device: Device to run predictions on

    Returns:
        predictions: List of predicted class indices
        targets: List of true class indices (if available)
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, batch_targets = batch

            # Move to device
            inputs.numeric = inputs.numeric.to(device)
            inputs.categorical = inputs.categorical.to(device)

            # Forward pass
            outputs = model(
                input_numeric=inputs.numeric,
                input_categorical=inputs.categorical,
                deterministic=True,
            )

            # Get predictions
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(batch_preds.reshape(-1))

            if batch_targets is not None:
                targets.extend(batch_targets.categorical.reshape(-1).cpu().numpy())

    return predictions, targets


# Example usage
if __name__ == "__main__":
    # Path to your saved model
    model_path = "checkpoints/two_stage_model_two_stage_encoder_decoder_acc_0.850.pt"

    # Load the model
    try:
        model, config, events_names = load_two_stage_model(model_path)
        print("Model loaded successfully!")
        print(f"Events: {events_names}")

        # To use the model:
        # 1. Prepare your data using the same preprocessing as training
        # 2. Create a DataLoader with EncoderDecoderDataset
        # 3. Call predict_anomalies(model, data_loader)

    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        print("Please train the model first using anomaly_encoder_decoder.py")
