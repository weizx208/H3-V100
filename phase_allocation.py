"""Acquire real allocator storage before admitting the following native stage."""
import torch
from .runtime_memory import request_cuda_headroom

def allocate_with_recovery(allocate, device, options, *, required_bytes, reason, on_oom=None):
    """Use cached blocks directly; retry once after an actual allocation OOM.

    No inactive-byte total is treated as proof of capacity. The returned live
    tensors themselves prove allocation, including alignment/fragmentation.
    The failed callback frame must be gone before reclaim touches any pool.
    """
    try:
        return allocate()
    except torch.cuda.OutOfMemoryError:
        pass
    if on_oom is not None:
        on_oom()
    request_cuda_headroom(device, options, reason=reason, required_free_bytes=int(required_bytes), minimum_reclaimable_mib=0, allow_vbar_release=True)
    return allocate()
