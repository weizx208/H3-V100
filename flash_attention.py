"""Workflow-scoped exact Flash/Sol dispatcher for the stable V100 profile."""
import torch
from . import backend
from . import sol_attention
from .exact_flash_range import make_exact_flash_range_stream
from .runtime_memory import POLICY_KEY, request_cuda_headroom
ENABLED_KEY = 'v100_flash_attention_enabled'
MIN_TOKENS_KEY = 'v100_flash_attention_min_tokens'
STREAM_OUTPUT_KEY = 'v100_h3_sol_stream_output'
SELECTED_BACKEND_KEY = 'v100_h3_selected_attention_backend'
SOL_EXACT_RANGE_KEY = 'v100_h3_sol_exact_range_allowed'
SOL_EXACT_RANGE_REQUIRED_KEY = 'v100_h3_sol_exact_range_required'
EXACT_STREAM_MIN_TOKENS = 38000
EXACT_STREAM_CHUNK_TOKENS = 2048
EXACT_STREAM_SAFETY_MIB = 512

def _exact_stream_decision(q, transformer_options):
    """Use bounded exact output at the long tier or under live pressure."""
    if not transformer_options.get(STREAM_OUTPUT_KEY, False):
        return (False, 'consumer-unavailable', None, None)
    selected = transformer_options.get(SELECTED_BACKEND_KEY)
    if selected != sol_attention.MODE_FLASH and (not bool(transformer_options.get(SOL_EXACT_RANGE_KEY, False))):
        return (False, 'not-selected-flash', None, None)
    tokens = int(q.shape[2])
    threshold = max(EXACT_STREAM_MIN_TOKENS, int(transformer_options.get('v100_h3_qkv_chunk_threshold', 38000)))
    estimated_peak = int(q.numel()) * 9
    if transformer_options.get(SOL_EXACT_RANGE_KEY, False) and transformer_options.get(SOL_EXACT_RANGE_REQUIRED_KEY, False):
        return (True, 'sol-calibration-contract', estimated_peak / 2 ** 20, None)
    if tokens >= threshold:
        return (True, 'projection-chunking-tier', estimated_peak / 2 ** 20, None)
    driver_free = None
    if q.is_cuda:
        driver_free = int(torch.cuda.mem_get_info(q.device)[0])
    if driver_free is not None and driver_free < estimated_peak + EXACT_STREAM_SAFETY_MIB * 2 ** 20:
        required = estimated_peak + EXACT_STREAM_SAFETY_MIB * 2 ** 20
        if POLICY_KEY in transformer_options:
            reclaim = request_cuda_headroom(q.device, transformer_options, reason='flash-full-output-admission', required_free_bytes=required, minimum_reclaimable_mib=256)
            driver_free = int(reclaim['after']['free_bytes'])
        enabled = driver_free < required
        reason = 'runtime-memory-budget' if enabled else 'full-output-after-cache-reclaim'
    else:
        enabled, reason = (False, 'full-output-profitable')
    return (enabled, reason, estimated_peak / 2 ** 20, None if driver_free is None else driver_free / 2 ** 20)

def v100_attention_override(original, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    transformer_options = kwargs.get('transformer_options') or {}
    enabled = bool(transformer_options.get(ENABLED_KEY, True))
    min_tokens = int(transformer_options.get(MIN_TOKENS_KEY, 1024))
    flash_reasons = list(backend.support_reasons(q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs))
    if q.shape[-2] < min_tokens or k.shape[-2] < min_tokens:
        flash_reasons.append('below-min-tokens')
    if not enabled:
        flash_reasons.append('disabled')
    flash_eligible = not flash_reasons
    if flash_eligible:
        stream, stream_reason, estimated_peak_mib, driver_free_mib = _exact_stream_decision(q, transformer_options)
        if stream:
            return make_exact_flash_range_stream(q, k, v, scale=kwargs.get('scale'), chunk_tokens=EXACT_STREAM_CHUNK_TOKENS, audio_ranges=transformer_options.get('minimax_h3_fp32_audio_ranges', ()), audio_overwrite_active=bool(transformer_options.get('v100_h3_fp32_audio_overwrite_active', False)))
        return backend.comfy_attention(q, k, v, heads, scale=kwargs.get('scale'))
    return original(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
v100_attention_override._v100_flash_attention_override = True

class V100FlashAttention:

    def patch(self, model, enabled=True, min_tokens=1024, attention_mode=sol_attention.MODE_FLASH, sol_tau=1.0, sol_min_tokens=16384, sol_block_size=64, allow_sol=False, sol_start_percent=0.2, sol_end_percent=0.8):
        if not enabled:
            return (model,)
        if not torch.cuda.is_available() or not any((torch.cuda.get_device_capability(index) == (7, 0) for index in range(torch.cuda.device_count()))):
            raise RuntimeError('V100 FlashAttention requires an SM70 CUDA device.')
        backend.load_extension()
        patched = model.clone()
        transformer_options = patched.model_options.setdefault('transformer_options', {})
        existing = transformer_options.get('optimized_attention_override')
        if existing is not None and (not getattr(existing, '_v100_flash_attention_override', False)):
            raise RuntimeError('Another optimized_attention_override is already active on this MODEL.')
        transformer_options['optimized_attention_override'] = v100_attention_override
        transformer_options[ENABLED_KEY] = True
        transformer_options[MIN_TOKENS_KEY] = int(min_tokens)
        transformer_options[sol_attention.MODE_KEY] = str(attention_mode)
        transformer_options['v100_sol_attention_h3_enabled'] = bool(allow_sol)
        return (patched,)
