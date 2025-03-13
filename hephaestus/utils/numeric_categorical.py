from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NumericCategoricalData:
    """
    Data class to store time series batch data.

    Attributes:
        numeric: Numeric inputs for the model.
        categorical: Categorical inputs for the model, may be None if no categorical data exists.
    """

    numeric: torch.Tensor
    categorical: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        """
        Move tensors to the specified device.

        Args:
            device: The device to move tensors to (e.g., 'cuda', 'cpu', or torch.device)

        Returns:
            NumericCategoricalData: Self with tensors moved to the specified device
        """
        self.numeric = self.numeric.to(device)
        if self.categorical is not None:
            self.categorical = self.categorical.to(device)
        return self
