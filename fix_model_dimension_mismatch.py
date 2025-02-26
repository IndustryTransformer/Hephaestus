import pandas as pd
import torch

from hephaestus.models.models import TimeSeriesConfig, TimeSeriesDecoder, TimeSeriesDS


def diagnose_and_fix_dimension_issue():
    """Function to diagnose and propose fixes for dimension mismatches."""
    print("Loading data and model...")
    # Load a small sample of your data
    df = pd.read_parquet("data/planets.parquet").head(1000)

    # Create configuration and dataset
    config = TimeSeriesConfig.generate(df=df)
    ds = TimeSeriesDS(df, config)

    print(f"Dataset length: {len(ds)}")

    # Create model with the same configuration as in your script
    model = TimeSeriesDecoder(config, d_model=512, n_heads=8)
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

    # Check for dimension mismatches
    numeric_shape_match = numeric_tensor.shape == outputs["numeric"].shape
    categorical_shape_match = (
        categorical_tensor.shape[0:2] == outputs["categorical"].shape[0:2]
    )

    print(f"Numeric shapes match: {numeric_shape_match}")
    print(f"Categorical shapes match: {categorical_shape_match}")

    # If shapes don't match, inspect the model configuration
    if not numeric_shape_match or not categorical_shape_match:
        print("\nInspecting model configuration...")
        print(f"Model input_size for numeric data: {model.numeric_input_size}")
        print(f"Model input_size for categorical data: {model.categorical_input_size}")
        print(f"Number of numeric columns: {len(config.numeric_columns)}")
        print(f"Number of categorical columns: {len(config.categorical_columns)}")

        # Check if output sizes match the original input sizes
        print("\nOutput layer configuration:")
        print(
            f"Numeric embedding dimension: {model.numeric_in_embedding.embedding_dim}"
        )
        print(
            f"Categorical embedding dimension: {model.categorical_in_embedding.embedding_dim}"
        )

        # Debug the decoder outputs
        print("\nDecoder output details:")
        print(f"Hidden dimension used in model: {model.d_model}")
        print(f"Input sequence length: {numeric_tensor.shape[1]}")
        print(f"Output sequence length: {outputs['numeric'].shape[1]}")

        # Try to identify potential fixes
        if numeric_tensor.shape[1] != outputs["numeric"].shape[1]:
            print("\nPotential issue found: Sequence length mismatch")
            print(
                "This could be because the model architecture is changing the sequence length."
            )
            print("Fix options:")
            print(
                "1. Modify the add_input_offsets function to handle sequence length mismatches"
            )
            print(
                "2. Check the model architecture to ensure it preserves sequence length"
            )
            print(
                "3. Add additional sequence length parameters to the model configuration"
            )

        if numeric_tensor.shape[2] != outputs["numeric"].shape[2]:
            print("\nPotential issue found: Feature dimension mismatch")
            print(
                "The model is producing a different number of features than the input."
            )
            print("Fix options:")
            print(
                "1. Check if the model's output linear layers match the input dimensions"
            )
            print(
                "2. Modify the add_input_offsets function to handle feature dimension mismatches"
            )

    return numeric_tensor, outputs


if __name__ == "__main__":
    diagnose_and_fix_dimension_issue()
