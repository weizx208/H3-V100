"""Explicit ownership for secondary QKV acquired before capacity admission."""
import torch


class SecondaryQKV:
    """CUDA tensors are created, used and released only by the fixed worker."""
    def __init__(self, device, heads, tokens, head_dim):
        self.device = torch.device('cuda', device)
        self.shape = (1, heads, tokens, head_dim)
        self.nbytes = 3 * heads * tokens * head_dim * 2
        self.qkv = None

    def allocate(self):
        with torch.cuda.device(self.device), torch.inference_mode():
            try:
                self.qkv = tuple(torch.empty(self.shape, dtype=torch.float16,
                                            device=self.device) for _ in range(3))
            except torch.cuda.OutOfMemoryError:
                self.qkv = None
                return False
        return True

    def release(self):
        with torch.cuda.device(self.device):
            self.qkv = None
            torch.cuda.synchronize(self.device)


class AcquiredQKV(tuple):
    """Primary tensor tuple plus an opaque secondary owner, never raw CUDA1 values."""
    def __new__(cls, primary, secondary):
        result = super().__new__(cls, primary)
        result.secondary = secondary
        return result


def acquire_secondary(state, device, heads, tokens, head_dim):
    owner = SecondaryQKV(device, heads, tokens, head_dim)
    if state._graph_executor.submit(owner.allocate).result():
        return owner
    state._graph_executor.submit(owner.release).result()
    return None


def release_secondary(state, owner):
    if owner is not None:
        state._graph_executor.submit(owner.release).result()
