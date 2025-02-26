import torch


def ensure_tensor_dtype(tensor, target_dtype=torch.float32):
    """
    Ensure a tensor has the target dtype.

    Args:
        tensor: The input tensor
        target_dtype: The target data type (default: torch.float32)

    Returns:
        The tensor with the target dtype
    """
    if tensor.dtype != target_dtype:
        return tensor.to(target_dtype)
    return tensor


def ensure_batch_dtypes(batch, target_dtype=torch.float32):
    """
    Ensure all tensors in a batch have the target dtype.

    Args:
        batch: Dictionary of tensors or list/tuple of tensors
        target_dtype: The target data type (default: torch.float32)

    Returns:
        The batch with all tensors converted to the target dtype
    """
    if isinstance(batch, dict):
        return {
            k: ensure_tensor_dtype(v, target_dtype) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
    elif isinstance(batch, (list, tuple)):
        return type(batch)(
            ensure_tensor_dtype(v, target_dtype) if torch.is_tensor(v) else v
            for v in batch
        )
    elif torch.is_tensor(batch):
        return ensure_tensor_dtype(batch, target_dtype)
    return batch
