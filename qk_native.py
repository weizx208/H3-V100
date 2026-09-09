"""Packaged SM70 FP32 Q/K RMSNorm + partial split-half RoPE kernel."""
from pathlib import Path
import torch
from .device_support import is_sm70_device
_loaded = False
_load_attempted = False
_load_error = None
_device_available = {}

def _ops_available():
    return hasattr(torch.ops.h3_v100_qk_cuda, 'rms_rope_split_half')

def load_extension():
    global _loaded, _load_attempted, _load_error
    if _loaded or _ops_available():
        _loaded = True
        return True
    if _load_attempted:
        return False
    _load_attempted = True
    binaries = list(Path(__file__).resolve().parent.glob('h3_v100_qk_cuda*.pyd'))
    if len(binaries) != 1:
        _load_error = RuntimeError(f'expected exactly one packaged h3_v100_qk_cuda binary, found {len(binaries)}')
        return False
    try:
        torch.ops.load_library(str(binaries[0]))
        _loaded = _ops_available()
        if not _loaded:
            _load_error = RuntimeError('packaged Q/K operator was not registered')
    except Exception as exc:
        _load_error = exc
        _loaded = False
    return _loaded

def available_for(device):
    device = torch.device(device)
    if device.type != 'cuda':
        return False
    key = (device.type, device.index)
    cached = _device_available.get(key)
    if cached is not None:
        return cached
    if not is_sm70_device(device):
        _device_available[key] = False
        return False
    available = bool(load_extension())
    _device_available[key] = available
    return available

def supports(q, k, rope, q_weight, k_weight, rot_dim):
    return bool(q.is_cuda and k.is_cuda and rope.is_cuda and (q.dtype == k.dtype == rope.dtype == torch.float32) and (q.ndim == 4) and (q.shape == k.shape) and (q.shape[0] == 1) and (q.shape[-1] == 128) and q.is_contiguous() and k.is_contiguous() and q_weight.is_cuda and k_weight.is_cuda and (q_weight.dtype == k_weight.dtype == torch.float32) and (q_weight.numel() == k_weight.numel() == 128) and q_weight.is_contiguous() and k_weight.is_contiguous() and (rope.ndim == 6) and (tuple(rope.shape[:3]) == (1, q.shape[1], 1)) and (tuple(rope.shape[-2:]) == (2, 2)) and (rope.shape[-3] * 2 == int(rot_dim)) and rope.is_contiguous() and (q.device == k.device == rope.device == q_weight.device == k_weight.device) and (0 < int(rot_dim) <= 128) and (int(rot_dim) % 2 == 0) and available_for(q.device))

def rms_rope_split_half(q, k, rope, q_weight, k_weight, epsilon, rot_dim, *, output_fp16=False):
    if not load_extension():
        raise RuntimeError('packaged H3 V100 Q/K operator is unavailable') from _load_error
    return torch.ops.h3_v100_qk_cuda.rms_rope_split_half(q, k, rope, q_weight, k_weight, float(epsilon), int(rot_dim), bool(output_fp16))
