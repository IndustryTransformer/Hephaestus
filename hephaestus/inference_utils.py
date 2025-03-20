from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch


@dataclass
class ModelInputs:
    """Container for model inputs in the correct format."""

    numeric: torch.Tensor
    categorical: torch.Tensor = None


def prepare_model_inputs(
    model_inputs: Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor], object
    ],
) -> ModelInputs:
    """
    Prepares inputs for the TabularDecoder model in the correct format.

    Args:
        model_inputs: The inputs to prepare, in various possible formats:
            - torch.Tensor: Assumed to be numeric data
            - Tuple[torch.Tensor, torch.Tensor]: (numeric, categorical)
            - Dict: With 'numeric' and optionally 'categorical' keys
            - object: Any object with numeric and categorical attributes

    Returns:
        ModelInputs object with properly formatted tensors
    """
    numeric = None
    categorical = None

    # Handle different input types
    if isinstance(model_inputs, torch.Tensor):
        numeric = model_inputs
    elif isinstance(model_inputs, tuple) and len(model_inputs) >= 1:
        numeric = model_inputs[0]
        if len(model_inputs) > 1:
            categorical = model_inputs[1]
    elif isinstance(model_inputs, dict):
        numeric = model_inputs.get("numeric")
        categorical = model_inputs.get("categorical")
    else:
        # Try to access attributes
        try:
            numeric = getattr(model_inputs, "numeric", None)
            categorical = getattr(model_inputs, "categorical", None)
        except Exception as e:
            print(f"Error: {e}")
            raise ValueError(f"Unsupported input format: {type(model_inputs)}")

    # Ensure numeric is not None
    if numeric is None:
        raise ValueError("Numeric data is required but was not provided")

    # Ensure proper shapes
    if numeric.dim() == 2:
        # If shape is [batch_size, seq_len], add a column dimension
        numeric = numeric.unsqueeze(1)

    if categorical is not None and categorical.dim() == 2:
        # If shape is [batch_size, seq_len], add a column dimension
        categorical = categorical.unsqueeze(1)

    # Ensure batch dimension
    if numeric.dim() == 2:  # [cols, seq_len]
        numeric = numeric.unsqueeze(0)

    if categorical is not None and categorical.dim() == 2:
        categorical = categorical.unsqueeze(0)

    return ModelInputs(numeric=numeric, categorical=categorical)
