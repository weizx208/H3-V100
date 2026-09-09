"""Cached, process-stable CUDA device capability checks."""
from functools import lru_cache
import torch

def _cuda_index(device) -> int | None:
    value = torch.device(device)
    if value.type != 'cuda':
        return None
    if value.index is not None:
        return int(value.index)
    try:
        return int(torch.cuda.current_device())
    except (RuntimeError, AttributeError):
        return None

@lru_cache(maxsize=None)
def _capability(index: int) -> tuple[int, int] | None:
    try:
        major, minor = torch.cuda.get_device_capability(index)
        return (int(major), int(minor))
    except (RuntimeError, AttributeError):
        return None

def is_sm70_device(device) -> bool:
    """Return whether *device* is an NVIDIA Volta compute capability 7.0 GPU."""
    index = _cuda_index(device)
    return index is not None and _capability(index) == (7, 0)
