import pandas as pd
import torch

from hephaestus.models.models import TimeSeriesConfig, TimeSeriesDecoder, TimeSeriesDS
from hephaestus.training.training import add_input_offsets


def debug_model_output_shapes():
    """Debug function to check shapes of model inputs and outputs"""
    # Load a small sample of your data
    df = pd.read_parquet("data/planets.parquet").head(100)

    # Create configuration and dataset
    config = TimeSeriesConfig.generate(df=df)
    ds = TimeSeriesDS(df, config)

    # Create model with the same configuration as in your script
    model = TimeSeriesDecoder(config, d_model=512, n_heads=8)

    # Set to evaluation mode
    model.eval()

    # Create a small batch
    batch_size = 4
    numeric_list = []
    categorical_list = []

    for i in range(batch_size):
        item = ds[i]
        numeric_list.append(item[0])
        categorical_list.append(item[1])

    numeric_tensor = torch.tensor(numeric_list, dtype=torch.float32)
    categorical_tensor = torch.tensor(categorical_list, dtype=torch.float32)

    print(f"Input numeric tensor shape: {numeric_tensor.shape}")
    print(f"Input categorical tensor shape: {categorical_tensor.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(
            numeric_inputs=numeric_tensor, categorical_inputs=categorical_tensor
        )

    print(f"Output numeric tensor shape: {outputs['numeric'].shape}")
    print(f"Output categorical tensor shape: {outputs['categorical'].shape}")

    # Test the loss function
    try:
        # Add debug prints to understand the shape mismatch
        print("\nTesting add_input_offsets function:")
        numeric_inputs = numeric_tensor
        numeric_outputs = outputs["numeric"]
        print(f"Before function - Input shape: {numeric_inputs.shape}")
        print(f"Before function - Output shape: {numeric_outputs.shape}")

        # Try the function that's causing the error
        modified_inputs, masked_outputs, nan_mask = add_input_offsets(
            numeric_inputs, numeric_outputs, inputs_offset=1
        )

        print("Function completed successfully")
        print(f"Modified inputs shape: {modified_inputs.shape}")
        print(f"Masked outputs shape: {masked_outputs.shape}")
        print(f"NaN mask shape: {nan_mask.shape}")

    except RuntimeError as e:
        print(f"Error in add_input_offsets: {e}")

        # If error occurs, try to understand the shape details
        if "must match" in str(e):
            print("\nDetailed shape information:")
            # Get more detailed shape information
            print(
                f"Numeric inputs: {numeric_inputs.shape} (features: {numeric_inputs.shape[2]})"
            )
            print(
                f"Numeric outputs: {numeric_outputs.shape} (features: {numeric_outputs.shape[2]})"
            )

            # Check if model config matches the input/output dimensions
            print("\nModel configuration:")
            print(f"Number of numeric features: {len(config.numeric_columns)}")
            print(f"Number of categorical features: {len(config.categorical_columns)}")


if __name__ == "__main__":
    debug_model_output_shapes()
