"""Clean up only synchronous Comfy casts owned by a failed H3 call."""
from contextlib import contextmanager
import torch
from .lora_failure_guard import strict_lora_merges

def dense_cast_weight(weight, dtype):
    """Complete a prepared cast inside its caller-owned Comfy context."""
    from comfy.quant_ops import QuantizedTensor
    return weight.dequantize().to(dtype=dtype) if isinstance(weight, QuantizedTensor) else weight

@contextmanager
def owned_casts(modules, device, options, *, stage='mlp'):
    owned = [m for m in modules if hasattr(m, '_v') and (not hasattr(m, '_prefetch'))]
    try:
        with strict_lora_merges():
            yield
    except Exception:
        abandoned = [(m, getattr(m, '_prefetch', None)) for m in owned]
        abandoned = [(m, p) for m, p in abandoned if isinstance(p, dict) and 'signature' in p and ('resident' in p)]
        if abandoned:
            torch.cuda.synchronize(device)
            from comfy.ops import uncast_bias_weight
            from .runtime_memory import _runtime_state
            for module, prefetch in abandoned:
                if not prefetch['resident']:
                    module._v_signature = None
                for key in ('weight', 'bias'):
                    patch = getattr(module, key + '_lowvram_function', None)
                    if patch is not None:
                        patch.clear_prepared()
                if prefetch['signature'] is not None:
                    uncast_bias_weight(module, None, None, (None, device, None))
                delattr(module, '_prefetch')
            state = _runtime_state(options)
            key = f'abandoned_{stage}_casts_cleaned'
            state[key] = state.get(key, 0) + len(abandoned)
        raise
