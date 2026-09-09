"""Workflow-scoped MiniMax H3 mixed-precision patch for NVIDIA V100.

SPDX-License-Identifier: GPL-3.0-only

The attention precision split is based on the community-tested Plan 2 profile
from Icbears/minimax-h3-v100-patch.  Unlike the original file patcher, this
module applies the replacement to a cloned ComfyUI MODEL via object patches.
"""
import logging
import math
import types
from contextlib import contextmanager
import torch
from .dual_runtime import _fatal_device_failure
from .device_support import is_sm70_device
from . import qk_native
import comfy.model_management as model_management
import comfy.quant_ops
from .weight_profile import qkv_native_reserve_bytes
from comfy.ldm.modules.attention import attention_pytorch, optimized_attention
from .sol_attention import BLOCK_COUNT_KEY, BLOCK_INDEX_KEY
from .runtime_memory import request_cuda_headroom
from .cast_failure_cleanup import owned_casts
LOGGER = logging.getLogger('H3V100MixedPrecision')
PATCH_MARKER = '_h3_v100_mixed_precision_plan2'
OPTION_KEY = 'h3_v100_mixed_precision_v100_only'
AUDIO_RANGES_OPTION_KEY = 'minimax_h3_fp32_audio_ranges'
AUDIO_OVERWRITE_ACTIVE_OPTION_KEY = 'v100_h3_fp32_audio_overwrite_active'
QKV_CHUNKING_OPTION_KEY = 'v100_h3_qkv_chunking'
QKV_CHUNK_TOKENS_OPTION_KEY = 'v100_h3_qkv_chunk_tokens'
QKV_CHUNK_THRESHOLD_OPTION_KEY = 'v100_h3_qkv_chunk_threshold'
QKV_CACHE_TRIM_THRESHOLD_OPTION_KEY = 'v100_h3_qkv_cache_trim_threshold_mb'
EXPERIMENTAL_FP16_OPTION_KEY = 'v100_h3_experimental_fp16_linear'
PROJECTION_WEIGHT_REUSE_OPTION_KEY = 'v100_h3_projection_weight_reuse'
SOL_STREAM_OUTPUT_OPTION_KEY = 'v100_h3_sol_stream_output'
BLOCK_PATCH_MARKER = '_h3_v100_audio_ranges'
BLOCK_ORIGINAL_FORWARD_ATTR = '_h3_v100_audio_ranges_original_forward'
REFINER_OPTION_KEY = 'v100_h3_token_refiner_attention'
REFINER_BLOCK_PATCH_MARKER = '_h3_v100_refiner_fp32_residual'
CONDITION_PATCH_MARKER = '_h3_v100_condition_fp32_input'
_missing_audio_ranges_reported = set()
_projection_reuse_fallback_reported = set()
_PROJECTION_WEIGHT_REUSE_DRIVER_FLOOR_MIB = 1024
_AUDIO_DRIVER_RESERVE_MIB = 512
_POST_ATTENTION_ACTIVATION_RESERVE_MIB = 512
_AUDIO_PLAN_ORDER = (('key-major', 512), ('key-major', 256), ('key-major', 128), ('query-major', 128), ('query-major', 64))

class _ProjectionWeightReuseUnsupported(TypeError):
    pass

@contextmanager
def _audio_overwrite_contract(transformer_options):
    """Publish and exactly restore the synchronous audio-overwrite promise."""
    missing = object()
    previous = transformer_options.get(AUDIO_OVERWRITE_ACTIVE_OPTION_KEY, missing)
    transformer_options[AUDIO_OVERWRITE_ACTIVE_OPTION_KEY] = True
    try:
        yield
    finally:
        if previous is missing:
            transformer_options.pop(AUDIO_OVERWRITE_ACTIVE_OPTION_KEY, None)
        else:
            transformer_options[AUDIO_OVERWRITE_ACTIVE_OPTION_KEY] = previous

def _tensor_nbytes(value):
    if value is None:
        return 0
    return int(value.numel()) * int(value.element_size())

def _reserve_post_attention_fp32_activation(result, transformer_options, sequence_length):
    """Hand unpinned weight residency to the next full FP32 norm if needed.

    Ultra-long range streaming avoids the old full attention allocation.  That
    is a net memory reduction, but it also means PyTorch no longer creates
    enough allocation pressure for ComfyUI to evict stale Dynamic VBAR pages.
    The next H3 operation is a full-size FP32 RMSNorm, outside the attention
    function and therefore invisible to the route preflight. PyTorch 2.8's
    FP32 RMSNorm needs two full activation-sized allocations at peak (measured
    independently on V100), not just its retained output. Reserve both plus
    the driver margin; inactive cache totals do not prove contiguous capacity.
    """
    from .bounded_norm import CURRENT_OPTIONS
    if CURRENT_OPTIONS.get() is transformer_options:
        return None
    if not isinstance(transformer_options, dict) or result.device.type != 'cuda' or int(sequence_length) <= int(transformer_options.get(QKV_CHUNK_THRESHOLD_OPTION_KEY, 38000)):
        return None
    _, total_bytes = torch.cuda.mem_get_info(result.device)
    driver_reserve_bytes = max(_POST_ATTENTION_ACTIVATION_RESERVE_MIB * 1024 ** 2, int(total_bytes * 0.04))
    return request_cuda_headroom(result.device, transformer_options, reason='post-attention-fp32-norm', required_free_bytes=2 * _tensor_nbytes(result) + driver_reserve_bytes, minimum_reclaimable_mib=128, allow_vbar_release=True)

def _is_projection_weight_resource_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any((marker in message for marker in ('out of memory', 'vbar_fault', 'vbar fault', 'result 2', 'cuda error: memory allocation', 'cudamalloc')))

@contextmanager
def _projection_weight_context(linear, device, dtype, options=None):
    """Pin one compute-precision projection weight for one invocation."""
    from comfy.ops import CastBiasWeightContext
    from .cast_failure_cleanup import dense_cast_weight
    if not hasattr(linear, '_forward'):
        raise _ProjectionWeightReuseUnsupported('H3 projection weight reuse requires ComfyUI Linear._forward.')
    if getattr(linear, 'pre_quant_scale', None) is not None:
        raise _ProjectionWeightReuseUnsupported('H3 projection weight reuse does not accept pre_quant_scale.')
    with owned_casts((linear,), device, options if options is not None else {}, stage='projection'), CastBiasWeightContext(linear, input=None, dtype=dtype, device=device, bias_dtype=dtype, offloadable=True, compute_dtype=dtype, want_requant=False) as weights:
        weight, bias = weights
        weight = dense_cast_weight(weight, dtype)
        yield (weight, bias)

def _prepared_projection(linear, input_part, prepared_weight):
    from comfy.ops import run_every_op
    weight, bias = prepared_weight
    run_every_op()
    return linear._forward(input_part, weight, bias)

def _ordinary_projection(linear, value, options):
    with owned_casts((linear,), value.device, options, stage='projection'):
        return linear(value)

def _projection_reuse_allowed(transformer_options, device, chunks, *, snapshot=None):
    if chunks <= 1 or device.type != 'cuda' or (not transformer_options.get(PROJECTION_WEIGHT_REUSE_OPTION_KEY, False)) or (not transformer_options.get(EXPERIMENTAL_FP16_OPTION_KEY, False)):
        return (False, 0.0, 'ineligible')
    if snapshot is None:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    else:
        free_bytes = int(snapshot['free_bytes'])
    free_mib = free_bytes / 1024 ** 2
    if free_mib < _PROJECTION_WEIGHT_REUSE_DRIVER_FLOOR_MIB:
        reclaim = request_cuda_headroom(device, transformer_options, reason='projection-weight-reuse', required_free_bytes=_PROJECTION_WEIGHT_REUSE_DRIVER_FLOOR_MIB * 1024 ** 2, snapshot=snapshot, minimum_reclaimable_mib=128)
        free_mib = reclaim['after']['free_bytes'] / 1024 ** 2
        if free_mib < _PROJECTION_WEIGHT_REUSE_DRIVER_FLOOR_MIB:
            return (False, free_mib, 'driver_transfer_floor')
    return (True, free_mib, 'eligible')

def _report_projection_reuse_fallback(projection, device, chunks, reason_type, reason_message):
    key = (projection, device.index, reason_type, reason_message[:160])
    if key in _projection_reuse_fallback_reported:
        return
    _projection_reuse_fallback_reported.add(key)
    LOGGER.warning('H3 V100 long projection weight reuse fallback: projection=%s, chunks=%d, reason=%s. Restoring the validated per-chunk path.', projection, chunks, f'{reason_type}: {reason_message}')

def _recover_projection_reuse_fallback(device):
    if device.type != 'cuda':
        return
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
_ATTENTION_FP16_SCALE = 16.0
_EXTREME_QKV_REFERENCE_TOKENS = 110000
_EXTREME_QKV_REFERENCE_VRAM = 16 * 1024 ** 3

def _extreme_qkv_policy(tokens, device, *, driver_snapshot=None):
    if device.type != 'cuda':
        return (False, 1024, 0)
    if driver_snapshot is None:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    else:
        free_bytes, total_bytes = driver_snapshot
    scaled_threshold = max(48000, int(_EXTREME_QKV_REFERENCE_TOKENS * total_bytes / _EXTREME_QKV_REFERENCE_VRAM))
    pressure_threshold = max(48000, int(scaled_threshold * 0.85))
    extreme = tokens >= scaled_threshold or (free_bytes < total_bytes // 16 and tokens >= pressure_threshold)
    return (extreme, 512 if extreme else 1024, scaled_threshold)

def _unwrap_our_block_forward(value):
    current = value
    seen = set()
    while current is not None:
        function = getattr(current, '__func__', current)
        if not getattr(function, BLOCK_PATCH_MARKER, False):
            return current
        identity = id(function)
        if identity in seen:
            raise RuntimeError('H3 mixed precision detected a cyclic wrapper chain.')
        seen.add(identity)
        current = getattr(function, BLOCK_ORIGINAL_FORWARD_ATTR, None)
    raise RuntimeError('H3 mixed precision could not recover its original forward.')

def _slice_rope(rope_freqs, start, stop):
    return rope_freqs[:, start:stop]

def _normalize_rope_pair(self, q, k, rope_freqs, device, *, output_fp16=False):
    """Run H3 Q/K RMSNorm+RoPE in FP32 for one token window."""
    qw = model_management.cast_to(self.q_norm.weight, dtype=q.dtype, device=device)
    kw = model_management.cast_to(self.k_norm.weight, dtype=k.dtype, device=device)
    rope = rope_freqs.to(q.dtype) if rope_freqs.dtype != q.dtype else rope_freqs
    rot = rope.shape[-3] * 2
    if qk_native.supports(q, k, rope, qw, kw, rot):
        return qk_native.rms_rope_split_half(q, k, rope, qw, kw, self.q_norm.eps, rot, output_fp16=output_fp16)
    return comfy.quant_ops.ck.rms_rope_split_half(q, k, rope, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)

def _chunked_qk_rope(self, q, k, rope_freqs, transformer_options):
    """Normalize/RoPE Q/K in bounded FP32 windows and overwrite FP16 inputs."""
    tokens = int(q.shape[0])
    chunk_tokens = max(1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024)))
    q_out = q.view(tokens, self.heads, self.head_dim)
    k_out = k.view(tokens, self.heads, self.head_dim)
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        qc = q_out[start:stop].unsqueeze(0).float()
        kc = k_out[start:stop].unsqueeze(0).float()
        qc, kc = _normalize_rope_pair(self, qc, kc, _slice_rope(rope_freqs, start, stop), q.device, output_fp16=True)
        q_out[start:stop].copy_(qc[0])
        k_out[start:stop].copy_(kc[0])
    return (q_out, k_out)

def _estimate_audio_workspace_mib(*, audio_rows, heads, head_dim, key_chunk, query_chunk):
    """Conservative peak for the exact query-major FP32 audio path.

    The estimate covers retained audio outputs, FP32 Q/K/V tiles, the score and
    probability tiles, the PV result and allocator/GEMM slack.  It deliberately
    overestimates the measured SM70 peak at small Query chunks so admission
    never trades long-sequence viability for speed.
    """
    element_bytes = 4
    retained = int(audio_rows) * int(heads) * int(head_dim) * element_bytes
    kv_tiles = 2 * int(key_chunk) * int(heads) * int(head_dim) * element_bytes
    score_tiles = 2 * int(query_chunk) * int(heads) * int(key_chunk) * element_bytes
    query_state = 3 * int(query_chunk) * int(heads) * int(head_dim) * element_bytes
    raw = retained + kv_tiles + score_tiles + query_state
    return raw * 1.4 / 1024 ** 2

def _estimate_audio_key_major_workspace_mib(*, audio_rows, heads, head_dim, key_chunk, query_chunk):
    """Conservative peak for retaining all audio states across each K/V tile."""
    element_bytes = 4
    retained_states = 2 * int(audio_rows) * int(heads) * int(head_dim) * element_bytes
    running_scalars = 2 * int(audio_rows) * int(heads) * element_bytes
    kv_tiles = 2 * int(key_chunk) * int(heads) * int(head_dim) * element_bytes
    score_tiles = 2 * int(query_chunk) * int(heads) * int(key_chunk) * element_bytes
    pv_tile = int(query_chunk) * int(heads) * int(head_dim) * element_bytes
    raw = retained_states + running_scalars + kv_tiles + score_tiles + pv_tile
    return raw * 1.6 / 1024 ** 2

def _choose_audio_plan_from_budget(usable_mib, *, audio_rows, heads, head_dim, key_chunk):
    for loop_order, query_chunk in _AUDIO_PLAN_ORDER:
        estimator = _estimate_audio_key_major_workspace_mib if loop_order == 'key-major' else _estimate_audio_workspace_mib
        required = estimator(audio_rows=audio_rows, heads=heads, head_dim=head_dim, key_chunk=key_chunk, query_chunk=query_chunk)
        if required <= float(usable_mib):
            return (loop_order, query_chunk, required)
    loop_order, query_chunk = _AUDIO_PLAN_ORDER[-1]
    return (loop_order, query_chunk, _estimate_audio_workspace_mib(audio_rows=audio_rows, heads=heads, head_dim=head_dim, key_chunk=key_chunk, query_chunk=query_chunk))

def _select_audio_plan(q, audio_ranges, heads, head_dim, key_chunk):
    audio_rows = sum((max(0, int(stop) - int(start)) for start, stop in audio_ranges))
    if q.device.type != 'cuda':
        required = _estimate_audio_workspace_mib(audio_rows=audio_rows, heads=heads, head_dim=head_dim, key_chunk=key_chunk, query_chunk=64)
        return ('query-major', 64, required, 0.0, 'non-cuda')
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(q.device)
        allocated_bytes = int(torch.cuda.memory_allocated(q.device))
        reserved_bytes = int(torch.cuda.memory_reserved(q.device))
        reclaimable_bytes = max(0, reserved_bytes - allocated_bytes)
        reserve_bytes = max(_AUDIO_DRIVER_RESERVE_MIB * 1024 ** 2, int(total_bytes * 0.03))
        usable_bytes = reclaimable_bytes + max(0, int(free_bytes) - reserve_bytes)
        usable_mib = usable_bytes / 1024 ** 2
        loop_order, query_chunk, required_mib = _choose_audio_plan_from_budget(usable_mib, audio_rows=audio_rows, heads=heads, head_dim=head_dim, key_chunk=key_chunk)
        return (loop_order, query_chunk, required_mib, usable_mib, 'runtime-memory-budget')
    except Exception:
        required = _estimate_audio_workspace_mib(audio_rows=audio_rows, heads=heads, head_dim=head_dim, key_chunk=key_chunk, query_chunk=64)
        return ('query-major', 64, required, 0.0, 'memory-query-failed')

def _streaming_audio_attention_chunked(q, k, v, audio_ranges, heads, head_dim, *, key_chunk, query_chunk):
    outputs = []
    scale = head_dim ** (-0.5)
    for range_start, range_stop in audio_ranges:
        for qs in range(range_start, range_stop, query_chunk):
            qe = min(range_stop, qs + query_chunk)
            qf = q[:, :, qs:qe].float()
            shape = qf.shape[:-1] + (1,)
            running_max = torch.full(shape, -torch.inf, dtype=torch.float32, device=q.device)
            running_sum = torch.zeros(shape, dtype=torch.float32, device=q.device)
            running_out = torch.zeros(qf.shape, dtype=torch.float32, device=q.device)
            for ks in range(0, k.shape[2], key_chunk):
                ke = min(k.shape[2], ks + key_chunk)
                scores = torch.matmul(qf, k[:, :, ks:ke].float().transpose(-2, -1)).mul_(scale)
                block_max = scores.amax(dim=-1, keepdim=True)
                new_max = torch.maximum(running_max, block_max)
                correction = torch.exp(running_max - new_max)
                probs = torch.exp(scores - new_max)
                del scores, block_max
                running_sum.mul_(correction).add_(probs.sum(dim=-1, keepdim=True))
                running_out.mul_(correction).add_(torch.matmul(probs, v[:, :, ks:ke].float()))
                running_max = new_max
                del correction, probs
            outputs.append((qs, qe, (running_out / running_sum).transpose(1, 2).reshape(1, qe - qs, heads * head_dim)))
    return outputs

def _streaming_audio_attention_key_major(q, k, v, audio_ranges, heads, head_dim, *, key_chunk, query_chunk):
    """Reuse each FP32 K/V tile across every exact audio Query tile."""
    states = []
    scale = head_dim ** (-0.5)
    for range_start, range_stop in audio_ranges:
        for qs in range(range_start, range_stop, query_chunk):
            qe = min(range_stop, qs + query_chunk)
            qf = q[:, :, qs:qe].float()
            shape = qf.shape[:-1] + (1,)
            states.append([qs, qe, qf, torch.full(shape, -torch.inf, dtype=torch.float32, device=q.device), torch.zeros(shape, dtype=torch.float32, device=q.device), torch.zeros(qf.shape, dtype=torch.float32, device=q.device)])
    for ks in range(0, k.shape[2], key_chunk):
        ke = min(k.shape[2], ks + key_chunk)
        kf_t = k[:, :, ks:ke].float().transpose(-2, -1)
        vf = v[:, :, ks:ke].float()
        for state in states:
            qf, running_max, running_sum, running_out = state[2:]
            scores = torch.matmul(qf, kf_t).mul_(scale)
            block_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)
            correction = torch.exp(running_max - new_max)
            probs = torch.exp(scores - new_max)
            del scores, block_max
            running_sum.mul_(correction).add_(probs.sum(dim=-1, keepdim=True))
            running_out.mul_(correction).add_(torch.matmul(probs, vf))
            state[3] = new_max
            del correction, probs
        del kf_t, vf
    return [(qs, qe, (running_out / running_sum).transpose(1, 2).reshape(1, qe - qs, heads * head_dim)) for qs, qe, _qf, _running_max, running_sum, running_out in states]

def _streaming_audio_attention(q, k, v, audio_ranges, heads, head_dim, key_chunk=1024, transformer_options=None):
    """Exact online-softmax audio attention with adaptive FP32 Query tiles."""
    loop_order, query_chunk, required_mib, usable_mib, reason = _select_audio_plan(q, audio_ranges, heads, head_dim, key_chunk)
    audio_rows = sum((int(stop) - int(start) for start, stop in audio_ranges))
    while True:
        try:
            if loop_order == 'key-major':
                return _streaming_audio_attention_key_major(q, k, v, audio_ranges, heads, head_dim, key_chunk=key_chunk, query_chunk=query_chunk)
            return _streaming_audio_attention_chunked(q, k, v, audio_ranges, heads, head_dim, key_chunk=key_chunk, query_chunk=query_chunk)
        except Exception as error:
            if not _is_projection_weight_resource_error(error):
                raise
            try:
                plan_index = _AUDIO_PLAN_ORDER.index((loop_order, query_chunk))
            except ValueError:
                plan_index = len(_AUDIO_PLAN_ORDER) - 1
            if plan_index + 1 >= len(_AUDIO_PLAN_ORDER):
                raise
            next_order, next_chunk = _AUDIO_PLAN_ORDER[plan_index + 1]
            LOGGER.warning('H3 V100 exact audio attention reduced plan after memory pressure: tokens=%d, plan=%s/%d->%s/%d, error=%s.', int(q.shape[2]), loop_order, query_chunk, next_order, next_chunk, type(error).__name__)
            loop_order, query_chunk = (next_order, next_chunk)
        torch.cuda.empty_cache()

def _is_v100_device(device):
    """Return True only for Volta compute capability 7.0 CUDA devices."""
    return is_sm70_device(device)

def _trim_qkv_cache_if_needed(device, transformer_options, *, required_free_mib=None, reason='qkv-attention-boundary', return_snapshot=False):
    if device.type != 'cuda':
        return (None, None) if return_snapshot else None
    threshold = float(transformer_options.get(QKV_CACHE_TRIM_THRESHOLD_OPTION_KEY, 2048))
    target_mib = max(threshold, 0.0 if required_free_mib is None else float(required_free_mib))
    reclaim = request_cuda_headroom(device, transformer_options, reason=str(reason), required_free_bytes=int(target_mib * 1024 ** 2), minimum_reclaimable_mib=512)
    current_snapshot = reclaim.get('after') or reclaim.get('before')
    if not reclaim['performed']:
        return (None, current_snapshot) if return_snapshot else None
    before = reclaim['before']
    after = reclaim['after']
    trimmed = (before['free_bytes'] / 1024 ** 2, after['free_bytes'] / 1024 ** 2, before['reserved_bytes'] / 1024 ** 2, after['reserved_bytes'] / 1024 ** 2)
    return (trimmed, current_snapshot) if return_snapshot else trimmed

def _trim_and_log(device, transformer_options, reason):
    trimmed = _trim_qkv_cache_if_needed(device, transformer_options, reason=reason)
    return trimmed

def _qkv_required_bytes(tokens, input_width, output_width, chunk_tokens, element_size, extreme, transformer_options=None):
    """Three persistent outputs plus tile conversions and the native reserve."""
    outputs = 3 * int(tokens) * int(output_width) * int(element_size)
    tile = min(int(tokens), int(chunk_tokens))
    scratch = tile * (int(input_width) * 6 + 3 * int(output_width) * int(element_size))
    reserve = qkv_native_reserve_bytes(transformer_options, extreme=bool(extreme))
    return outputs + scratch + reserve

def _use_local_qkv_input(self, x, transformer_options, extreme_policy, *, driver_snapshot=None):
    if extreme_policy[0]:
        return True
    if not transformer_options.get(QKV_CHUNKING_OPTION_KEY, False):
        return False
    if int(x.shape[0]) <= int(transformer_options.get(QKV_CHUNK_THRESHOLD_OPTION_KEY, 38000)):
        return False
    chunk = max(1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024)))
    required = _qkv_required_bytes(x.shape[0], x.shape[1], self.heads * self.head_dim, chunk, 2, False, transformer_options)
    whole_peak = max(x.numel() * 6, required + x.numel() * 2)
    if driver_snapshot is None:
        free, _ = torch.cuda.mem_get_info(x.device)
    else:
        free, _ = driver_snapshot
    return int(free) < whole_peak

def _qkv_input_policy(self, x, transformer_options, *, use_fp16):
    """Share one adjacent driver sample across QKV pressure decisions."""
    driver_snapshot = torch.cuda.mem_get_info(x.device) if x.device.type == 'cuda' else None
    extreme_policy = _extreme_qkv_policy(int(x.shape[0]), x.device, driver_snapshot=driver_snapshot)
    local_qkv = bool(use_fp16) and _use_local_qkv_input(self, x, transformer_options, extreme_policy, driver_snapshot=driver_snapshot)
    return (extreme_policy, local_qkv)

def _qkv_projection(self, proj_x, transformer_options, local_fp16_scale=1.0, extreme_policy=None):
    """Project QKV, using separate contiguous outputs for long inference."""
    tokens = int(proj_x.shape[0])
    enabled = bool(transformer_options.get(QKV_CHUNKING_OPTION_KEY, False))
    threshold = max(0, int(transformer_options.get(QKV_CHUNK_THRESHOLD_OPTION_KEY, 38000)))
    chunk_tokens = max(1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024)))
    if extreme_policy is None:
        extreme_policy = _extreme_qkv_policy(tokens, proj_x.device)
    extreme, extreme_chunk_tokens, extreme_threshold = extreme_policy
    if extreme:
        chunk_tokens = min(chunk_tokens, extreme_chunk_tokens)
    width = self.heads * self.head_dim
    if not enabled or tokens <= threshold:
        projection_input = proj_x
        if local_fp16_scale != 1.0:
            projection_input = (projection_input * (1.0 / local_fp16_scale)).half()
        return _ordinary_projection(self.qkv_proj, projection_input, transformer_options).split(width, dim=-1)
    trimmed = None
    chunks = math.ceil(tokens / chunk_tokens)
    reuse_active = False
    if torch.is_grad_enabled() and proj_x.requires_grad:
        result = torch.cat([self.qkv_proj((part * (1.0 / local_fp16_scale)).half() if local_fp16_scale != 1.0 else part) for part in proj_x.split(chunk_tokens, dim=0)], dim=0)
        q_out, k_out, v_out = result.split(width, dim=-1)
        layout = 'combined-autograd'
    else:
        reserved_outputs = None
        allocate_outputs = None
        trimmed = None
        if proj_x.device.type == 'cuda' and (proj_x.dtype == torch.float16 or local_fp16_scale != 1.0 or extreme):
            from .phase_allocation import allocate_with_recovery

            def allocate_raw_outputs():
                return tuple((torch.empty((tokens, width), dtype=torch.float16, device=proj_x.device) for _ in range(3)))

            def allocate_outputs():
                return allocate_with_recovery(allocate_raw_outputs, proj_x.device, transformer_options, required_bytes=3 * tokens * width * 2, reason='qkv-output-allocation')
            trimmed = request_cuda_headroom(proj_x.device, transformer_options, reason='qkv-native-stage', required_free_bytes=qkv_native_reserve_bytes(transformer_options, extreme=bool(extreme)), minimum_reclaimable_mib=128, allow_vbar_release=True)

        def execute(prepared_weight=None):
            if reserved_outputs is None:
                q_result = k_result = v_result = None
            else:
                q_result, k_result, v_result = reserved_outputs
            for start in range(0, tokens, chunk_tokens):
                input_part = proj_x[start:start + chunk_tokens]
                if local_fp16_scale != 1.0:
                    input_part = (input_part * (1.0 / local_fp16_scale)).half()
                elif extreme and input_part.dtype == torch.float32:
                    input_part = input_part.half()
                if prepared_weight is None:
                    part = _ordinary_projection(self.qkv_proj, input_part, transformer_options)
                else:
                    part = _prepared_projection(self.qkv_proj, input_part, prepared_weight)
                q_part, k_part, v_part = part.split(width, dim=-1)
                if q_result is None:
                    if allocate_outputs is not None:
                        q_result, k_result, v_result = allocate_outputs()
                    else:
                        output_shape = (tokens, width)
                        q_result = torch.empty(output_shape, dtype=part.dtype, device=part.device)
                        k_result = torch.empty_like(q_result)
                        v_result = torch.empty_like(q_result)
                stop = start + part.shape[0]
                q_result[start:stop].copy_(q_part)
                k_result[start:stop].copy_(k_part)
                v_result[start:stop].copy_(v_part)
                del part, q_part, k_part, v_part, input_part
            return (q_result, k_result, v_result)
        reuse_allowed, _free_before_mib, admission_reason = _projection_reuse_allowed(transformer_options, proj_x.device, chunks, snapshot=trimmed.get('after') or trimmed.get('before') if trimmed is not None else None)
        fallback_type = None
        fallback_message = None
        recover_resource_failure = False
        prepared_weight = None
        if reuse_allowed:
            prepared_dtype = torch.float16 if local_fp16_scale != 1.0 or (extreme and proj_x.dtype == torch.float32) else proj_x.dtype
            try:
                with _projection_weight_context(self.qkv_proj, proj_x.device, prepared_dtype, transformer_options) as prepared_weight:
                    if allocate_outputs is not None:
                        reserved_outputs = allocate_outputs()
                    q_out, k_out, v_out = execute(prepared_weight)
                reuse_active = True
            except _ProjectionWeightReuseUnsupported as exc:
                fallback_type = type(exc).__name__
                fallback_message = str(exc)
            except Exception as exc:
                if not _is_projection_weight_resource_error(exc):
                    raise
                fallback_type = type(exc).__name__
                fallback_message = str(exc)
                recover_resource_failure = True
        prepared_weight = None
        if fallback_type is not None:
            q_out = k_out = v_out = None
            reserved_outputs = None
            _report_projection_reuse_fallback('qkv', proj_x.device, chunks, fallback_type, fallback_message)
            if recover_resource_failure:
                _recover_projection_reuse_fallback(proj_x.device)
        if not reuse_active:
            q_out, k_out, v_out = execute()
        layout = 'split-contiguous-prepared-weight' if reuse_active else 'split-contiguous'
    return (q_out, k_out, v_out)

def _out_projection(self, out, transformer_options):
    """Bound the quantized output-projection accumulation above 38K tokens."""
    tokens = int(out.shape[0])
    enabled = bool(transformer_options.get(QKV_CHUNKING_OPTION_KEY, False))
    threshold = max(0, int(transformer_options.get(QKV_CHUNK_THRESHOLD_OPTION_KEY, 38000)))
    chunk_tokens = max(1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024)))
    extreme, extreme_chunk_tokens, _ = _extreme_qkv_policy(tokens, out.device)
    if extreme:
        chunk_tokens = min(chunk_tokens, extreme_chunk_tokens)

    def project(part, prepared_weight=None):
        if prepared_weight is None:
            return _ordinary_projection(self.out_proj, part, transformer_options)
        return _prepared_projection(self.out_proj, part, prepared_weight)
    if not enabled or tokens <= threshold:
        return project(out)
    trimmed, reuse_snapshot = _trim_qkv_cache_if_needed(out.device, transformer_options, reason='output-projection-entry', return_snapshot=True)
    chunks = math.ceil(tokens / chunk_tokens)
    reuse_active = False
    if torch.is_grad_enabled() and out.requires_grad:
        result = torch.cat([project(part) for part in out.split(chunk_tokens, dim=0)], dim=0)
    else:

        def execute(prepared_weight=None):
            projected = None
            for start in range(0, tokens, chunk_tokens):
                part = project(out[start:start + chunk_tokens], prepared_weight)
                if projected is None:
                    projected = torch.empty((tokens,) + tuple(part.shape[1:]), dtype=part.dtype, device=part.device)
                projected[start:start + part.shape[0]].copy_(part)
            return projected
        reuse_allowed, _free_before_mib, admission_reason = _projection_reuse_allowed(transformer_options, out.device, chunks, snapshot=reuse_snapshot)
        fallback_type = None
        fallback_message = None
        recover_resource_failure = False
        if reuse_allowed:
            try:
                with _projection_weight_context(self.out_proj, out.device, out.dtype, transformer_options) as prepared_weight:
                    result = execute(prepared_weight)
                reuse_active = True
            except _ProjectionWeightReuseUnsupported as exc:
                fallback_type = type(exc).__name__
                fallback_message = str(exc)
            except Exception as exc:
                if not _is_projection_weight_resource_error(exc):
                    raise
                fallback_type = type(exc).__name__
                fallback_message = str(exc)
                recover_resource_failure = True
        if fallback_type is not None:
            result = None
            _report_projection_reuse_fallback('out', out.device, chunks, fallback_type, fallback_message)
            if recover_resource_failure:
                _recover_projection_reuse_fallback(out.device)
        if not reuse_active:
            result = execute()
    return result

def _is_sol_range_stream(value):
    return bool(getattr(value, '_h3_v100_sol_range_stream', False))

def _replace_streamed_audio_rows(part, part_start, audio_outputs):
    """Overwrite overlapping sparse rows with the validated FP32 audio result."""
    part_stop = part_start + int(part.shape[0])
    for audio_start, audio_stop, audio_out in audio_outputs or ():
        overlap_start = max(part_start, int(audio_start))
        overlap_stop = min(part_stop, int(audio_stop))
        if overlap_start >= overlap_stop:
            continue
        part[overlap_start - part_start:overlap_stop - part_start].copy_(audio_out[0, overlap_start - int(audio_start):overlap_stop - int(audio_start)])

def _project_attention_chunks_into_input(self, target, chunks, transformer_options, *, audio_outputs=(), use_fp16=True):
    """Reuse fresh AdaLN input storage for projected attention chunks.

    MiniMax H3 no longer needs the normalized/modulated attention input after
    QKV projection has completed.  Writing the projected result back into that
    same FP32 tensor avoids both a full FP32 attention promotion and a second
    full hidden-width output.  The enclosing DiT block still performs its
    original gate/residual operation on the returned tensor.
    """
    started = False

    def execute(prepared_weight=None):
        nonlocal started
        for start, source in chunks:
            started = True
            rows = int(source.shape[2]) if source.ndim == 4 else int(source.shape[1])
            if source.ndim == 4:
                part = source.transpose(1, 2).reshape(rows, -1).float()
            else:
                part = source[0].float()
            _replace_streamed_audio_rows(part, int(start), audio_outputs)
            if prepared_weight is None:
                projected = _ordinary_projection(self.out_proj, part, transformer_options)
            else:
                projected = _prepared_projection(self.out_proj, part, prepared_weight)
            if use_fp16:
                projected = projected.float().mul_(_ATTENTION_FP16_SCALE)
            target[int(start):int(start) + rows].copy_(projected)
            del source, part, projected
        return target
    reuse_requested = bool(transformer_options.get(PROJECTION_WEIGHT_REUSE_OPTION_KEY, False) and transformer_options.get(EXPERIMENTAL_FP16_OPTION_KEY, False))
    if reuse_requested:
        recover_resource_failure = False
        try:
            with _projection_weight_context(self.out_proj, target.device, torch.float32, transformer_options) as prepared_weight:
                result = execute(prepared_weight)
            return result
        except _ProjectionWeightReuseUnsupported:
            pass
        except Exception as exc:
            if not _is_projection_weight_resource_error(exc):
                raise
            if started:
                raise
            recover_resource_failure = True
        if recover_resource_failure:
            _recover_projection_reuse_fallback(target.device)
    return execute()

def _consume_sol_range_stream(self, target, stream, transformer_options, *, audio_outputs=(), use_fp16=True):
    """Consume a bounded Sol stream transactionally into disposable H3 input."""

    def sparse_chunks():
        for start, out, lse in stream:
            del lse
            yield (start, out)
    failure = None
    try:
        return _project_attention_chunks_into_input(self, target, sparse_chunks(), transformer_options, audio_outputs=audio_outputs, use_fp16=use_fp16)
    except Exception as error:
        if _fatal_device_failure(error):
            raise
        failure = (type(error).__name__, str(error))
    if failure is not None:
        exact = stream.exact_fallback(failure)
        if _is_sol_range_stream(exact):
            return _consume_sol_range_stream(self, target, exact, transformer_options, audio_outputs=audio_outputs, use_fp16=use_fp16)
        try:
            chunk_tokens = max(64, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024)))

            def exact_chunks():
                for start in range(0, int(exact.shape[1]), chunk_tokens):
                    yield (start, exact[:, start:start + chunk_tokens])
            return _project_attention_chunks_into_input(self, target, exact_chunks(), transformer_options, audio_outputs=audio_outputs, use_fp16=use_fp16)
        finally:
            del exact

def h3_v100_attention_forward(self, x, rope_freqs=None, transformer_options={}):
    """Plan 2: FP16 QKV/attention with FP32 norm, RoPE and residual stream."""
    s = x.shape[0]
    residual_dtype = x.dtype
    v100_only = True
    if isinstance(transformer_options, dict):
        v100_only = transformer_options.get(OPTION_KEY, True)
    use_fp16 = x.device.type == 'cuda' and x.dtype == torch.float32 and (not v100_only or _is_v100_device(x.device))
    audio_ranges = ()
    if isinstance(transformer_options, dict):
        audio_ranges = transformer_options.get(AUDIO_RANGES_OPTION_KEY, ())
    use_fp32_audio_attention = use_fp16 and bool(audio_ranges) and (not model_management.in_training)
    missing_audio_metadata = use_fp16 and (not audio_ranges) and (not model_management.in_training) and (not bool(transformer_options.get(REFINER_OPTION_KEY, False)))
    extreme_policy, local_qkv = _qkv_input_policy(self, x, transformer_options, use_fp16=use_fp16)
    extreme_qkv, _, _ = extreme_policy
    if use_fp16 and (not local_qkv):
        proj_x = (x * (1.0 / _ATTENTION_FP16_SCALE)).half()
        local_fp16_scale = 1.0
    else:
        proj_x = x
        local_fp16_scale = _ATTENTION_FP16_SCALE if use_fp16 else 1.0
    q, k, v = _qkv_projection(self, proj_x, transformer_options, local_fp16_scale=local_fp16_scale, extreme_policy=extreme_policy)
    if proj_x is not x:
        del proj_x
    adaptive_qk = bool(use_fp16 and transformer_options.get(QKV_CHUNKING_OPTION_KEY, False) and (s > int(transformer_options.get(QKV_CHUNK_THRESHOLD_OPTION_KEY, 38000))) and (rope_freqs is not None))
    if use_fp16 and (not adaptive_qk):
        q = q.float()
        k = k.float()
    v = v.view(s, self.heads, self.head_dim)
    if adaptive_qk:
        q, k = _chunked_qk_rope(self, q, k, rope_freqs, transformer_options)
    elif rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        q, k = _normalize_rope_pair(self, q, k, rope_freqs, x.device)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    if adaptive_qk:
        _trim_and_log(x.device, transformer_options, 'contiguous-attention-output')
    precomputed_audio_outputs = None
    if use_fp32_audio_attention and adaptive_qk:
        precomputed_audio_outputs = _streaming_audio_attention(q, k, v, audio_ranges, self.heads, self.head_dim, key_chunk=max(1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024))), transformer_options=transformer_options)
    if use_fp32_audio_attention:
        with _audio_overwrite_contract(transformer_options):
            attention_out = optimized_attention(q.half(), k.half(), v.half(), self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
        if _is_sol_range_stream(attention_out):
            audio_outputs = precomputed_audio_outputs
            if audio_outputs is None:
                audio_outputs = _streaming_audio_attention(q, k, v, audio_ranges, self.heads, self.head_dim, key_chunk=max(1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024))), transformer_options=transformer_options)
            result = _consume_sol_range_stream(self, x, attention_out, transformer_options, audio_outputs=audio_outputs, use_fp16=True)
            attention_out = q = k = v = None
            audio_outputs = precomputed_audio_outputs = None
            _reserve_post_attention_fp32_activation(result, transformer_options, s)
            return result
        if adaptive_qk:
            audio_outputs = precomputed_audio_outputs
            if audio_outputs is None:
                audio_outputs = _streaming_audio_attention(q, k, v, audio_ranges, self.heads, self.head_dim, key_chunk=max(1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024))), transformer_options=transformer_options)
            del q, k, v
            _trim_and_log(x.device, transformer_options, 'fp32-attention-promotion')
            out = attention_out.to(residual_dtype)
            del attention_out
            for start, stop, audio_out in audio_outputs:
                out[:, start:stop] = audio_out.to(residual_dtype)
            del audio_outputs
        else:
            out = attention_out.to(residual_dtype)
            del attention_out
            audio_v = v.to(residual_dtype)
            for start, stop in audio_ranges:
                if 0 <= start < stop <= s:
                    out[:, start:stop] = attention_pytorch(q[:, :, start:stop], k, audio_v, self.heads, mask=None, skip_reshape=True)
    elif missing_audio_metadata:
        warning_key = (x.device.index, s)
        if warning_key not in _missing_audio_ranges_reported:
            _missing_audio_ranges_reported.add(warning_key)
            LOGGER.warning('H3 audio row metadata is missing or invalid for sequence=%d; falling back to full FP32 attention for audio safety.', s)
        out = optimized_attention(q.float(), k.float(), v.float(), self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options).to(residual_dtype)
    elif use_fp16:
        attention_out = optimized_attention(q.half(), k.half(), v.half(), self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
        if _is_sol_range_stream(attention_out):
            result = _consume_sol_range_stream(self, x, attention_out, transformer_options, audio_outputs=(), use_fp16=True)
            attention_out = q = k = v = None
            _reserve_post_attention_fp32_activation(result, transformer_options, s)
            return result
        if adaptive_qk:
            del q, k, v
            _trim_and_log(x.device, transformer_options, 'fp32-attention-promotion')
        out = attention_out.to(residual_dtype)
        del attention_out
    else:
        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
    result = _out_projection(self, out.squeeze(0), transformer_options)
    if use_fp16:
        result = result.float().mul_(_ATTENTION_FP16_SCALE)
    if adaptive_qk:
        out = q = k = v = None
        _reserve_post_attention_fp32_activation(result, transformer_options, s)
    return result
setattr(h3_v100_attention_forward, PATCH_MARKER, True)
h3_v100_attention_forward._uses_optimized_attention = True

def _audio_ranges_from_mod_segments(mod_segments, sequence_length):
    """Return H3 audio spans from its (start, stop, timestep*3+modality) table."""
    ranges = []
    for segment in mod_segments or ():
        if not isinstance(segment, (tuple, list)) or len(segment) != 3:
            continue
        start, stop, row = segment
        if not all((isinstance(value, int) for value in (start, stop, row))):
            continue
        if row % 3 == 2 and 0 <= start < stop <= sequence_length:
            ranges.append((start, stop))
    return tuple(ranges)

def _make_h3_block_forward(original_forward, block_index, block_count):
    """Expose audio row ranges only while the corresponding block is running."""
    from .block_lifetime import compatible, forward as lifetime_forward
    lifetime_compatible = compatible(getattr(original_forward, '__func__', original_forward))

    def h3_v100_block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        if not isinstance(transformer_options, dict):
            return original_forward(x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        missing = object()
        previous = transformer_options.get(AUDIO_RANGES_OPTION_KEY, missing)
        previous_block = transformer_options.get(BLOCK_INDEX_KEY, missing)
        previous_block_count = transformer_options.get(BLOCK_COUNT_KEY, missing)
        transformer_options[AUDIO_RANGES_OPTION_KEY] = _audio_ranges_from_mod_segments(mod_segments, x.shape[0])
        transformer_options[BLOCK_INDEX_KEY] = int(block_index)
        transformer_options[BLOCK_COUNT_KEY] = int(block_count)
        try:
            if x.is_cuda and x.dtype == torch.float16:
                x = x.float()

            def call_original():
                from .bounded_norm import CURRENT_OPTIONS, MARKER as norm_marker
                norm_forward = getattr(getattr(self, 'norm2', None), 'forward', None)
                active = bool(getattr(getattr(norm_forward, '__func__', norm_forward), norm_marker, False) and x.device.type == 'cuda' and (x.dtype == torch.float32) and (x.ndim == 2) and (not torch.is_grad_enabled()))
                token = CURRENT_OPTIONS.set(transformer_options) if active else None
                try:
                    if active and lifetime_compatible:
                        return lifetime_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options)
                    return original_forward(x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
                finally:
                    if token is not None:
                        CURRENT_OPTIONS.reset(token)
            return call_original()
        finally:
            if previous is missing:
                transformer_options.pop(AUDIO_RANGES_OPTION_KEY, None)
            else:
                transformer_options[AUDIO_RANGES_OPTION_KEY] = previous
            if previous_block is missing:
                transformer_options.pop(BLOCK_INDEX_KEY, None)
            else:
                transformer_options[BLOCK_INDEX_KEY] = previous_block
            if previous_block_count is missing:
                transformer_options.pop(BLOCK_COUNT_KEY, None)
            else:
                transformer_options[BLOCK_COUNT_KEY] = previous_block_count
    setattr(h3_v100_block_forward, BLOCK_PATCH_MARKER, True)
    setattr(h3_v100_block_forward, BLOCK_ORIGINAL_FORWARD_ATTR, _unwrap_our_block_forward(original_forward))
    return h3_v100_block_forward

def _make_refiner_block_forward(original_forward):
    """Keep the two pre-DiT text-refiner residual streams in FP32."""

    def h3_v100_refiner_forward(self, x, transformer_options={}):
        if x.is_floating_point() and x.dtype != torch.float32:
            x = x.float()
        if not isinstance(transformer_options, dict):
            return original_forward(x, transformer_options=transformer_options)
        missing = object()
        previous = transformer_options.get(REFINER_OPTION_KEY, missing)
        transformer_options[REFINER_OPTION_KEY] = True
        try:
            return original_forward(x, transformer_options=transformer_options)
        finally:
            if previous is missing:
                transformer_options.pop(REFINER_OPTION_KEY, None)
            else:
                transformer_options[REFINER_OPTION_KEY] = previous
    setattr(h3_v100_refiner_forward, REFINER_BLOCK_PATCH_MARKER, True)
    return h3_v100_refiner_forward

def _make_condition_forward(original_forward):
    """Protect the Qwen-to-H3 condition projection from global FP16 compute."""

    def h3_v100_condition_forward(self, x, *args, **kwargs):
        return original_forward(x.float(), *args, **kwargs)
    setattr(h3_v100_condition_forward, CONDITION_PATCH_MARKER, True)
    return h3_v100_condition_forward

def _is_our_patch(value):
    function = getattr(value, '__func__', value)
    return bool(getattr(function, PATCH_MARKER, False))

class H3V100MixedPrecision:
    """Apply the V100 precision split to one cloned MiniMax H3 MODEL."""

    def patch(self, model, enabled=True, v100_only=True):
        if not enabled:
            return (model,)
        patched_model = model.clone()
        try:
            diffusion_model = patched_model.get_model_object('diffusion_model')
        except Exception as exc:
            raise RuntimeError('MiniMax H3 V100 patch could not access diffusion_model.') from exc
        blocks = getattr(diffusion_model, 'blocks', None)
        if not blocks:
            raise RuntimeError('MiniMax H3 V100 patch expected diffusion_model.blocks, but none were found.')
        for index, block in enumerate(blocks):
            attention = getattr(block, 'attn', None)
            if attention is None or not hasattr(attention, 'qkv_proj'):
                raise RuntimeError(f'MiniMax H3 V100 patch rejected this model: block {index} has no compatible attention/QKV projection.')
        transformer_options = patched_model.model_options.setdefault('transformer_options', {})
        transformer_options[OPTION_KEY] = bool(v100_only)
        transformer_options[SOL_STREAM_OUTPUT_OPTION_KEY] = True
        token_refiner = getattr(diffusion_model, 'token_refiner', None)
        refiner_blocks = getattr(token_refiner, 'blocks', None)
        condition_proj = getattr(diffusion_model, 'condition_proj', None)
        if not refiner_blocks or condition_proj is None:
            raise RuntimeError('MiniMax H3 V100 patch expected condition_proj and token_refiner.blocks before the main DiT blocks.')
        condition_key = 'diffusion_model.condition_proj.forward'
        existing_condition = patched_model.object_patches.get(condition_key)
        condition_patched = 0
        if existing_condition is None:
            patched_model.add_object_patch(condition_key, types.MethodType(_make_condition_forward(condition_proj.forward), condition_proj))
            condition_patched = 1
        elif not getattr(getattr(existing_condition, '__func__', existing_condition), CONDITION_PATCH_MARKER, False):
            raise RuntimeError(f'MiniMax H3 V100 condition safety found another patch at {condition_key}.')
        for index, refiner_block in enumerate(refiner_blocks):
            refiner_key = f'diffusion_model.token_refiner.blocks.{index}.forward'
            existing_refiner = patched_model.object_patches.get(refiner_key)
            if existing_refiner is None:
                patched_model.add_object_patch(refiner_key, types.MethodType(_make_refiner_block_forward(refiner_block.forward), refiner_block))
            elif not getattr(getattr(existing_refiner, '__func__', existing_refiner), REFINER_BLOCK_PATCH_MARKER, False):
                raise RuntimeError(f'MiniMax H3 V100 refiner safety found another patch at {refiner_key}.')
            refiner_attention_key = f'diffusion_model.token_refiner.blocks.{index}.attn.forward'
            existing_refiner_attention = patched_model.object_patches.get(refiner_attention_key)
            if existing_refiner_attention is None:
                patched_model.add_object_patch(refiner_attention_key, types.MethodType(h3_v100_attention_forward, refiner_block.attn))
            elif not _is_our_patch(existing_refiner_attention):
                raise RuntimeError(f'MiniMax H3 V100 refiner attention found another patch at {refiner_attention_key}.')
        from .bounded_norm import make_bounded_norm, MARKER as norm_marker
        for index, block in enumerate(blocks):
            for norm_name in ('norm1', 'norm2'):
                norm = getattr(block, norm_name, None)
                if norm is not None:
                    norm_key = f'diffusion_model.blocks.{index}.{norm_name}.forward'
                    existing_norm = patched_model.object_patches.get(norm_key, norm.forward)
                    if not getattr(getattr(existing_norm, '__func__', existing_norm), norm_marker, False):
                        patched_model.add_object_patch(norm_key, types.MethodType(make_bounded_norm(existing_norm, stage=norm_name), norm))
            block_key = f'diffusion_model.blocks.{index}.forward'
            existing_block = patched_model.object_patches.get(block_key)
            if existing_block is not None:
                block_function = getattr(existing_block, '__func__', existing_block)
                if not getattr(block_function, BLOCK_PATCH_MARKER, False):
                    raise RuntimeError(f'MiniMax H3 V100 audio safety patch found another block patch at {block_key}. Remove that patch before this node.')
                base_block_forward = _unwrap_our_block_forward(existing_block)
            else:
                base_block_forward = _unwrap_our_block_forward(block.forward)
            patched_model.add_object_patch(block_key, types.MethodType(_make_h3_block_forward(base_block_forward, index, len(blocks)), block))
            key = f'diffusion_model.blocks.{index}.attn.forward'
            existing = patched_model.object_patches.get(key)
            if existing is not None:
                if _is_our_patch(existing):
                    continue
                raise RuntimeError(f'MiniMax H3 V100 patch found another Attention patch at {key}. Remove the other attention/low-VRAM patch before this node.')
            patched_model.add_object_patch(key, types.MethodType(h3_v100_attention_forward, block.attn))
        return (patched_model,)
