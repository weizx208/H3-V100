"""Packaged SM70 kernels for the validated scaled-FP16 H3 MLP boundary."""
from pathlib import Path
import torch
from .device_support import is_sm70_device
_loaded = False
_load_attempted = False
_load_error = None
_device_available = {}

def _ops_available():
    namespace = torch.ops.h3_v100_mlp_cuda
    return hasattr(namespace, 'scaled_swiglu_out') and hasattr(namespace, 'scale_store_fp16_to_fp32')

def load_extension():
    global _loaded, _load_attempted, _load_error
    if _loaded or _ops_available():
        _loaded = True
        return True
    if _load_attempted:
        return False
    _load_attempted = True
    binaries = list(Path(__file__).resolve().parent.glob('h3_v100_mlp_cuda*.pyd'))
    if len(binaries) != 1:
        _load_error = RuntimeError(f'expected exactly one packaged h3_v100_mlp_cuda binary, found {len(binaries)}')
        return False
    try:
        torch.ops.load_library(str(binaries[0]))
        _loaded = _ops_available()
        if not _loaded:
            _load_error = RuntimeError('packaged MLP operators were not registered')
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

def supports_scaled_swiglu(up, out=None):
    if not (up.is_cuda and up.dtype == torch.float16 and (up.ndim == 2) and up.is_contiguous() and (up.shape[1] % 2 == 0) and available_for(up.device)):
        return False
    return out is None or (out.is_cuda and out.dtype == torch.float16 and (out.ndim == 2) and out.is_contiguous() and (out.device == up.device) and (out.shape[0] == up.shape[0]) and (out.shape[1] * 2 == up.shape[1]))

def scaled_swiglu_out(up, out, branch_scale, fc2_scale):
    return torch.ops.h3_v100_mlp_cuda.scaled_swiglu_out(up, out, float(branch_scale), float(fc2_scale))

def supports_scale_store(input_value, output, output_element_offset=0):
    return input_value.is_cuda and input_value.dtype == torch.float16 and input_value.is_contiguous() and output.is_cuda and (output.dtype == torch.float32) and output.is_contiguous() and (input_value.device == output.device) and (int(output_element_offset) >= 0) and (int(output_element_offset) + input_value.numel() <= output.numel()) and available_for(input_value.device)

def scale_store_fp16_to_fp32(input_value, output, output_element_offset, scale):
    return torch.ops.h3_v100_mlp_cuda.scale_store_fp16_to_fp32(input_value, output, int(output_element_offset), float(scale))
