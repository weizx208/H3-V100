"""Bounded-memory execution for explicitly validated token-wise modules."""
import logging
import math
import types
from contextlib import ExitStack, contextmanager
import torch
from . import mlp_native
from .native_dynamic_vbar import CONTROLLER_KEY
from .phase_allocation import allocate_with_recovery
from .cast_failure_cleanup import owned_casts
from .runtime_memory import _runtime_state, RUNTIME_KEY, memory_snapshot, request_cuda_headroom
from .weight_profile import FP8_E4M3_PROFILE, weight_profile_from_options
LOGGER = logging.getLogger('V100TokenChunking')
PATCH_MARKER = '_v100_tokenwise_chunking'
ORIGINAL_FORWARD_ATTR = '_v100_tokenwise_original_forward'
OPTION_KEY = 'v100_mlp_chunk_tokens'
ADAPTIVE_KEY = 'v100_mlp_adaptive'
CACHE_TRIM_KEY = 'v100_mlp_cache_trim'
CACHE_TRIM_THRESHOLD_KEY = 'v100_mlp_cache_trim_threshold_mb'
QKV_CHUNKING_KEY = 'v100_h3_qkv_chunking'
QKV_CHUNK_TOKENS_KEY = 'v100_h3_qkv_chunk_tokens'
QKV_CHUNK_THRESHOLD_KEY = 'v100_h3_qkv_chunk_threshold'
QKV_CACHE_TRIM_THRESHOLD_KEY = 'v100_h3_qkv_cache_trim_threshold_mb'
EXPERIMENTAL_FP16_KEY = 'v100_h3_experimental_fp16_linear'
SCALED_FP16_SWIGLU_KEY = 'v100_h3_scaled_fp16_swiglu'
_SWIGLU_BRANCH_SCALE = 16.0
_SWIGLU_FC2_SCALE = 8.0
CONTROLLER_ATTR = '_h3_v100_dynamic_vbar_policy'
_UPGRADE_STABLE_CALLS = 3
_UPGRADE_BUDGET_MARGIN_TOKENS = 1024
_MLP_WEIGHT_PAIR_DRIVER_FLOOR_MIB = 1024
_FP8_MLP_WEIGHT_PAIR_SAFETY_RATIO = 1.1
_weight_pair_fallback_reported = set()
MLP_STEP_EPOCH_KEY = 'v100_h3_mlp_step_epoch'
MLP_STEP_FREEZE_KEY = 'v100_h3_mlp_step_freezes'

class _WeightPairUnsupported(TypeError):
    pass

def _unwrap_our_forward(value):
    """Return the clean callable beneath any previous copy of our wrapper."""
    current = value
    seen = set()
    while current is not None:
        function = getattr(current, '__func__', current)
        if not getattr(function, PATCH_MARKER, False):
            return current
        identity = id(function)
        if identity in seen:
            raise RuntimeError('H3 MLP chunking detected a cyclic V100 wrapper chain.')
        seen.add(identity)
        current = getattr(function, ORIGINAL_FORWARD_ATTR, None)
    raise RuntimeError('H3 MLP chunking could not recover its original forward.')

def _call_mlp(module, original_forward, x, transformer_options, block_index, prepared_weights=None, native_swiglu_out=None, native_result_output=None, native_result_element_offset=0):
    if not transformer_options.get(EXPERIMENTAL_FP16_KEY, False):
        return original_forward(x)
    if prepared_weights is None:
        up = module.fc1(x.half())
    else:
        fc1_weight, fc1_bias, fc2_weight, fc2_bias = prepared_weights
        from comfy.ops import run_every_op
        run_every_op()
        up = module.fc1._forward(x.half(), fc1_weight, fc1_bias)
    if transformer_options.get(SCALED_FP16_SWIGLU_KEY, False):
        native_active = mlp_native.supports_scaled_swiglu(up, native_swiglu_out)
        if native_active:
            if native_swiglu_out is None:
                native_swiglu_out = torch.empty((up.shape[0], up.shape[1] // 2), dtype=torch.float16, device=up.device)
            swiglu = mlp_native.scaled_swiglu_out(up, native_swiglu_out, _SWIGLU_BRANCH_SCALE, _SWIGLU_FC2_SCALE)
        else:
            gate, value = up.chunk(2, dim=-1)
            scaled_value = value * (1.0 / _SWIGLU_BRANCH_SCALE)
            swiglu = torch.nn.functional.silu(gate).mul_(scaled_value)
            del scaled_value
            swiglu.mul_(1.0 / _SWIGLU_FC2_SCALE)
        if prepared_weights is None:
            projected = module.fc2(swiglu)
        else:
            run_every_op()
            projected = module.fc2._forward(swiglu, fc2_weight, fc2_bias)
        restore_scale = _SWIGLU_BRANCH_SCALE * _SWIGLU_FC2_SCALE
        if native_active:
            if native_result_output is None:
                native_result_output = torch.empty(projected.shape, dtype=torch.float32, device=projected.device)
                native_result_element_offset = 0
            if mlp_native.supports_scale_store(projected, native_result_output, native_result_element_offset):
                mlp_native.scale_store_fp16_to_fp32(projected, native_result_output, native_result_element_offset, restore_scale)
                width = int(projected.shape[-1])
                row_start = int(native_result_element_offset) // width
                result = native_result_output[row_start:row_start + projected.shape[0]]
            else:
                result = projected.float().mul_(restore_scale)
        else:
            result = projected.float().mul_(restore_scale)
        return result
    gate, value = up.chunk(2, dim=-1)
    swiglu = torch.nn.functional.silu(gate.float()).mul_(value.float())
    fc2_input = (swiglu / 256.0).half()
    if prepared_weights is None:
        projected = module.fc2(fc2_input)
    else:
        run_every_op()
        projected = module.fc2._forward(fc2_input, fc2_weight, fc2_bias)
    result = projected.float().mul_(256.0)
    return result

def _tensor_nbytes(value):
    if value is None:
        return 0
    return int(value.numel()) * int(value.element_size())

def _linear_weight_elements(linear):
    weight = getattr(linear, 'weight', None)
    shape = getattr(weight, 'shape', None)
    if shape is not None:
        try:
            return int(math.prod((int(value) for value in shape)))
        except (TypeError, ValueError):
            pass
    return int(getattr(linear, 'in_features')) * int(getattr(linear, 'out_features'))

def _mlp_weight_pair_driver_target_bytes(module, transformer_options):
    """Return the complete driver-visible budget for one prepared MLP pair."""
    mib = 1024 ** 2
    if weight_profile_from_options(transformer_options) != FP8_E4M3_PROFILE:
        return _MLP_WEIGHT_PAIR_DRIVER_FLOOR_MIB * mib
    elements = [_linear_weight_elements(module.fc1), _linear_weight_elements(module.fc2)]
    bias_elements = sum((int(getattr(linear, 'out_features', 0)) for linear in (module.fc1, module.fc2) if getattr(linear, 'bias', None) is not None))
    source_bytes = sum(elements)
    prepared_bytes = 2 * sum(elements) + 2 * bias_elements
    decode_scratch_bytes = max(elements)
    estimated_peak = source_bytes + prepared_bytes + decode_scratch_bytes
    target = math.ceil(estimated_peak * _FP8_MLP_WEIGHT_PAIR_SAFETY_RATIO)
    return math.ceil(target / mib) * mib

def _request_fp8_mlp_pair_headroom(module, device, transformer_options, *, snapshot=None):
    """Try one exact allocator reclaim before an FP8 pair floor bypass."""
    target = _mlp_weight_pair_driver_target_bytes(module, transformer_options)
    if snapshot is None:
        snapshot = _runtime_memory_snapshot(device)
    if weight_profile_from_options(transformer_options) != FP8_E4M3_PROFILE:
        return {'requested': False, 'performed': False, 'satisfied': int(snapshot['free_bytes']) >= target, 'before': snapshot, 'after': snapshot, 'target_free_bytes': target, 'reason': 'non-fp8-profile'}
    free_bytes = int(snapshot['free_bytes'])
    if free_bytes >= target:
        return {'requested': False, 'performed': False, 'satisfied': True, 'before': snapshot, 'after': snapshot, 'target_free_bytes': target, 'reason': 'enough-driver-free'}
    deficit = target - free_bytes
    reclaimable_bytes = int(snapshot.get('reclaimable_bytes', max(0, int(snapshot.get('reserved_bytes', 0)) - int(snapshot.get('allocated_bytes', 0)))))
    if reclaimable_bytes < deficit:
        return {'requested': False, 'performed': False, 'satisfied': False, 'before': snapshot, 'after': snapshot, 'target_free_bytes': target, 'reason': 'insufficient-reclaimable-deficit'}
    result = request_cuda_headroom(device, transformer_options, reason='mlp-weight-pair', required_free_bytes=target, snapshot=snapshot, honor_cooldown=True, minimum_reclaimable_mib=max(1, math.ceil(deficit / 1024 ** 2)), demand_bypass_cooldown=True, allow_vbar_release=False)
    result['requested'] = True
    return result

def _allocate_mlp_output(shape, dtype, device, options):
    return allocate_with_recovery(lambda: torch.empty(shape, dtype=dtype, device=device), device, options, required_bytes=math.prod(shape) * dtype.itemsize, reason='mlp-output-allocation')

def _pressure_key(device, tokens):
    return f'{device}:{int(tokens)}'

def _pressure_limit(options, device, tokens, proposed):
    state = options.get(RUNTIME_KEY, {})
    return min(proposed, state.get('mlp_pressure_caps', {}).get(_pressure_key(device, tokens), proposed))

def _record_chunk_pressure(options, device, tokens, rows):
    if rows < 640:
        return
    state = _runtime_state(options)
    caps = state.setdefault('mlp_pressure_caps', {})
    key = _pressure_key(device, tokens)
    cap = max(640, rows // 2 // 256 * 256)
    caps[key] = min(caps.get(key, rows), cap)

def _call_mlp_chunk(module, original_forward, x, options, block_index, sequence_tokens=None, **kwargs):

    def compute():
        with owned_casts((module.fc1, module.fc2), x.device, options):
            return _call_mlp(module, original_forward, x, options, block_index, **kwargs)
    if not x.is_cuda or kwargs.get('prepared_weights') is not None:
        return _call_mlp(module, original_forward, x, options, block_index, **kwargs)
    fc1_width = int(getattr(module.fc1, 'out_features', x.shape[-1] * 2))
    fc2_width = int(getattr(module.fc2, 'out_features', x.shape[-1]))
    weight_elements = max(int(getattr(module.fc1, 'in_features', x.shape[-1])) * fc1_width, int(getattr(module.fc2, 'in_features', fc1_width // 2)) * fc2_width)
    required = max(_MLP_WEIGHT_PAIR_DRIVER_FLOOR_MIB * 1024 ** 2, 12 * weight_elements + int(x.shape[0]) * (fc1_width * 4 + fc2_width * 8))
    return allocate_with_recovery(compute, x.device, options, required_bytes=required, reason='mlp-chunk-execution', on_oom=(lambda: _record_chunk_pressure(options, x.device, sequence_tokens, int(x.shape[0]))) if sequence_tokens is not None and options.get(ADAPTIVE_KEY, False) else None)

def _is_weight_pair_resource_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any((marker in message for marker in ('out of memory', 'vbar_fault', 'vbar fault', 'result 2', 'cuda error: memory allocation', 'cudamalloc')))

@contextmanager
def _prepare_mlp_weight_pair(module, x, transformer_options):
    """Prepare and pin fc1/fc2 once for all chunks in this MLP invocation."""
    from comfy.ops import CastBiasWeightContext
    from .cast_failure_cleanup import dense_cast_weight
    if not all((hasattr(linear, '_forward') for linear in (module.fc1, module.fc2))):
        raise _WeightPairUnsupported('H3 MLP weight-pair reuse requires ComfyUI Linear._forward.')
    if any((getattr(linear, 'pre_quant_scale', None) is not None for linear in (module.fc1, module.fc2))):
        raise _WeightPairUnsupported('H3 MLP weight-pair reuse does not accept pre_quant_scale.')
    kwargs = {'input': None, 'dtype': torch.float16, 'device': x.device, 'bias_dtype': torch.float16, 'offloadable': True, 'compute_dtype': torch.float16, 'want_requant': False}
    with owned_casts((module.fc1, module.fc2), x.device, transformer_options), ExitStack() as stack:
        fc1_weight, fc1_bias = stack.enter_context(CastBiasWeightContext(module.fc1, **kwargs))
        fc1_weight = dense_cast_weight(fc1_weight, torch.float16)
        fc2_weight, fc2_bias = stack.enter_context(CastBiasWeightContext(module.fc2, **kwargs))
        fc2_weight = dense_cast_weight(fc2_weight, torch.float16)
        weights = (fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        yield (weights, sum((_tensor_nbytes(value) for value in weights)))

def _balanced_chunk_tokens(tokens, target, alignment=256, minimum=640):
    """Choose the largest balanced chunk that stays inside a live limit."""
    tokens = max(1, int(tokens))
    target = max(1, int(target))
    alignment = max(1, int(alignment))
    minimum = max(1, int(minimum))
    if tokens <= target:
        return tokens
    chunks = max(2, math.ceil(tokens / target))
    while chunks <= tokens:
        rows = math.ceil(tokens / chunks)
        aligned = math.ceil(rows / alignment) * alignment
        if aligned <= target:
            return max(minimum, aligned)
        chunks += 1
    return minimum

def _select_chunk_tokens(tokens, x, module, experimental_fp16=False, native_headroom_policy=False, runtime_memory=None):
    """Budget an H3 MLP call before the quantized fc1 allocation occurs."""
    device = x.device
    if device.type != 'cuda':
        return (tokens, 'non_cuda_full', {})
    mib = 1024 ** 2
    if runtime_memory is None:
        runtime_memory = _runtime_memory_snapshot(device)
    free_bytes = runtime_memory['free_bytes']
    total_bytes = runtime_memory['total_bytes']
    allocated_bytes = runtime_memory['allocated_bytes']
    reserved_bytes = runtime_memory['reserved_bytes']
    reusable_cache_bytes = max(0, reserved_bytes - allocated_bytes)
    credited_cache_bytes = 0 if native_headroom_policy else reusable_cache_bytes
    cache_credit_allowed = credited_cache_bytes > 0
    effective_bytes = free_bytes + credited_cache_bytes
    fc1_width = int(getattr(module.fc1, 'out_features', x.shape[-1] * 2))
    fc2_width = int(getattr(module.fc2, 'out_features', x.shape[-1]))
    chunk_scratch_bytes_per_token = fc1_width * 4 + fc1_width // 2 * 4
    persistent_output_bytes = int(tokens) * fc2_width * 4
    bytes_per_token = chunk_scratch_bytes_per_token + fc2_width * 4
    safety_bytes = max(int(total_bytes * 0.1), 1536 * mib)
    transfer_reserve_bytes = max(512 * mib, int(total_bytes * 0.04))
    reserve_bytes = safety_bytes + transfer_reserve_bytes
    allocator_effective_bytes = free_bytes + reusable_cache_bytes
    activation_usable_bytes = reusable_cache_bytes + max(0, free_bytes - reserve_bytes)
    chunk_usable_bytes = max(0, activation_usable_bytes - persistent_output_bytes)
    budget_tokens = max(1, int(chunk_usable_bytes // max(1, chunk_scratch_bytes_per_token)))
    reference_bytes = 16 * 1024 ** 3
    hard_full_limit = int(32768 * total_bytes / reference_bytes)
    hard_full_limit = max(8192, min(65536, hard_full_limit))
    hard_full_limit = hard_full_limit // 1024 * 1024
    maximum = min(tokens, hard_full_limit)
    target = min(maximum, budget_tokens)
    minimum = 640 if experimental_fp16 else 512
    if target >= tokens and tokens <= hard_full_limit:
        selected = tokens
        reason = 'runtime_budget_full'
    elif experimental_fp16:
        selected = _balanced_chunk_tokens(tokens, target, alignment=256, minimum=minimum)
        reason = 'balanced_measured_full_limit' if tokens > hard_full_limit else 'balanced_runtime_memory_budget'
    else:
        candidates = (16384, 8192, 4096, 2048, 1024, 512)
        selected = next((value for value in candidates if value <= target), minimum)
        reason = 'measured_full_limit' if tokens > hard_full_limit else 'runtime_memory_budget'
    details = {'driver_free_mib': free_bytes / mib, 'reusable_cache_mib': reusable_cache_bytes / mib, 'credited_cache_mib': credited_cache_bytes / mib, 'cache_credit_allowed': cache_credit_allowed, 'effective_mib': effective_bytes / mib, 'allocator_effective_mib': allocator_effective_bytes / mib, 'activation_usable_mib': activation_usable_bytes / mib, 'safety_mib': safety_bytes / mib, 'transfer_reserve_mib': transfer_reserve_bytes / mib, 'estimated_full_mib': tokens * bytes_per_token / mib, 'persistent_output_mib': persistent_output_bytes / mib, 'chunk_scratch_mib_per_token': chunk_scratch_bytes_per_token / mib, 'budget_tokens': budget_tokens, 'hard_full_limit': hard_full_limit, 'selection_cap': hard_full_limit}
    return (selected, reason, details)

def _runtime_memory_snapshot(device):
    return memory_snapshot(device)

def _mlp_step_key(transformer_options, block_index):
    """Return one stable key for every block in a denoising step."""
    sigmas = transformer_options.get('sigmas')
    try:
        sigma = float(torch.as_tensor(sigmas).reshape(-1)[0].item())
        return ('sigma', f'{sigma:.12g}')
    except (TypeError, ValueError, IndexError, RuntimeError):
        epoch = int(transformer_options.get(MLP_STEP_EPOCH_KEY, 0))
        if int(block_index) == 0:
            epoch += 1
            transformer_options[MLP_STEP_EPOCH_KEY] = epoch
        return ('epoch', epoch)

def _trim_if_needed(device, transformer_options, runtime_memory=None):
    if device.type != 'cuda' or not transformer_options.get(CACHE_TRIM_KEY, False):
        return False
    configured_threshold = max(0.0, float(transformer_options.get(CACHE_TRIM_THRESHOLD_KEY, 4096)))
    if runtime_memory is None:
        runtime_memory = _runtime_memory_snapshot(device)
    result = request_cuda_headroom(device, transformer_options, reason='mlp-soft-floor', required_free_bytes=None, snapshot=runtime_memory, honor_cooldown=True, soft_free_floor_mib=configured_threshold)
    return bool(result['performed'])

def _make_forward(original_forward, block_index, transformer_options, expected_dynamic_vbar_controller=None):
    """Chunk the leading token dimension without retaining chunk outputs."""

    def chunked_forward(self, x):
        runtime_memory = _runtime_memory_snapshot(x.device) if x.device.type == 'cuda' else None
        pre_trimmed = _trim_if_needed(x.device, transformer_options, runtime_memory=runtime_memory)
        if pre_trimmed:
            runtime_memory = _runtime_memory_snapshot(x.device)
        controller = transformer_options.get(CONTROLLER_KEY)
        if expected_dynamic_vbar_controller is not None:
            if controller is not expected_dynamic_vbar_controller:
                raise RuntimeError(f'H3 native Dynamic VBAR policy binding was lost before MLP block {block_index}.')
        adaptive = bool(transformer_options.get(ADAPTIVE_KEY, False))
        details = {}
        if adaptive:
            selections = transformer_options.setdefault('v100_mlp_auto_selections', {})
            experimental_fp16 = bool(transformer_options.get(EXPERIMENTAL_FP16_KEY, False))
            scaled_fp16_swiglu = bool(transformer_options.get(SCALED_FP16_SWIGLU_KEY, False))
            selection_key = (x.device.type, x.device.index, int(x.shape[0]), experimental_fp16, scaled_fp16_swiglu)
            selected, reason, details = _select_chunk_tokens(int(x.shape[0]), x, self, experimental_fp16, native_headroom_policy=controller is not None, runtime_memory=runtime_memory)
            selected = _pressure_limit(transformer_options, x.device, int(x.shape[0]), selected)
            if controller is not None and selected > 8448:
                selected = _balanced_chunk_tokens(int(x.shape[0]), 8448, alignment=256, minimum=640)
                reason = 'native_dynamic_vbar_correctness_cap'
            previous_selected = selections.get(selection_key)
            step_key = _mlp_step_key(transformer_options, block_index)
            step_freezes = transformer_options.setdefault(MLP_STEP_FREEZE_KEY, {})
            frozen = step_freezes.get(selection_key)
            same_step = bool(isinstance(frozen, dict) and frozen.get('step_key') == step_key)
            upgrade_states = transformer_options.setdefault('v100_mlp_upgrade_states', {})
            upgrade_state = upgrade_states.setdefault(selection_key, {'candidate': None, 'stable_calls': 0})
            if same_step and selected >= int(frozen['chunk_tokens']):
                selections[selection_key] = int(frozen['chunk_tokens'])
                reason = 'denoise_step_frozen_no_upgrade'
            elif previous_selected is None or selected < previous_selected:
                selections[selection_key] = selected
                upgrade_state['candidate'] = None
                upgrade_state['stable_calls'] = 0
            elif selected == previous_selected:
                upgrade_state['candidate'] = None
                upgrade_state['stable_calls'] = 0
            else:
                budget_has_margin = bool(details and details['budget_tokens'] >= selected + _UPGRADE_BUDGET_MARGIN_TOKENS)
                if budget_has_margin:
                    if upgrade_state['candidate'] == selected:
                        upgrade_state['stable_calls'] += 1
                    else:
                        upgrade_state['candidate'] = selected
                        upgrade_state['stable_calls'] = 1
                    if upgrade_state['stable_calls'] >= _UPGRADE_STABLE_CALLS:
                        selections[selection_key] = selected
                        reason = 'stable_runtime_budget_upgrade'
                        upgrade_state['candidate'] = None
                        upgrade_state['stable_calls'] = 0
                else:
                    upgrade_state['candidate'] = None
                    upgrade_state['stable_calls'] = 0
            applied = selections[selection_key]
            step_freezes[selection_key] = {'step_key': step_key, 'chunk_tokens': int(applied)}
            chunk_tokens = selections[selection_key]
        else:
            chunk_tokens = max(0, int(transformer_options.get(OPTION_KEY, 0)))
        full_tokens = int(x.shape[0]) if x.ndim > 0 else 0
        if chunk_tokens == 0 or x.ndim != 2 or x.shape[0] <= chunk_tokens:
            return _call_mlp(self, original_forward, x, transformer_options, block_index)
        tokens = int(x.shape[0])
        chunks = math.ceil(tokens / chunk_tokens)

        def execute_chunks(prepared_weights=None):
            from .runtime_memory import _runtime_state
            native_chunk_path = bool(prepared_weights is not None and transformer_options.get(SCALED_FP16_SWIGLU_KEY, False) and x.is_cuda and (x.dtype in (torch.float16, torch.float32)) and mlp_native.available_for(x.device))
            output = None
            swiglu_scratch = None
            if native_chunk_path:
                fc1_width = int(getattr(self.fc1, 'out_features', x.shape[-1] * 2))
                fc2_width = int(getattr(self.fc2, 'out_features', x.shape[-1]))
                output = _allocate_mlp_output((tokens, fc2_width), torch.float32, x.device, transformer_options)
                swiglu_scratch = torch.empty((min(tokens, chunk_tokens), fc1_width // 2), dtype=torch.float16, device=x.device)
            chunk_start = 0
            while chunk_start < tokens:
                current_limit = _pressure_limit(transformer_options, x.device, tokens, chunk_tokens)
                rows = min(current_limit, tokens - chunk_start)
                part = _call_mlp_chunk(self, original_forward, x[chunk_start:chunk_start + rows], transformer_options, block_index, sequence_tokens=tokens, prepared_weights=prepared_weights, native_swiglu_out=swiglu_scratch[:rows] if swiglu_scratch is not None else None, native_result_output=output, native_result_element_offset=chunk_start * output.shape[1] if output is not None else 0)
                if output is None:
                    output = _allocate_mlp_output((tokens,) + tuple(part.shape[1:]), part.dtype, part.device, transformer_options)
                target = output[chunk_start:chunk_start + part.shape[0]]
                if not (target.dtype == part.dtype and target.shape == part.shape and (target.data_ptr() == part.data_ptr())):
                    target.copy_(part)
                del part
                chunk_start += rows
            return output
        if torch.is_grad_enabled() and x.requires_grad:
            result = torch.cat([_call_mlp(self, original_forward, part, transformer_options, block_index) for part in x.split(chunk_tokens, dim=0)], dim=0)
        else:
            pair_eligible = bool(chunks > 1 and x.is_cuda and transformer_options.get(EXPERIMENTAL_FP16_KEY, False) and (transformer_options.get(CONTROLLER_KEY) is not None))
            if pair_eligible:
                driver_free_before_mib = details.get('driver_free_mib')
                if driver_free_before_mib is None:
                    if runtime_memory is None:
                        runtime_memory = _runtime_memory_snapshot(x.device)
                    driver_free_before_mib = runtime_memory['free_bytes'] / 1024 ** 2
                pair_target_bytes = _mlp_weight_pair_driver_target_bytes(self, transformer_options)
                pair_target_mib = pair_target_bytes / 1024 ** 2
                if weight_profile_from_options(transformer_options) == FP8_E4M3_PROFILE and driver_free_before_mib < pair_target_mib:
                    headroom_snapshot = runtime_memory
                    if headroom_snapshot is None:
                        headroom_snapshot = _runtime_memory_snapshot(x.device)
                    headroom = _request_fp8_mlp_pair_headroom(self, x.device, transformer_options, snapshot=headroom_snapshot)
                    after = headroom.get('after') or headroom_snapshot
                    driver_free_before_mib = int(after['free_bytes']) / 1024 ** 2
                if driver_free_before_mib < pair_target_mib:
                    from .runtime_memory import _runtime_state
                    result = execute_chunks()
                else:
                    fallback_exc = None
                    try:
                        with _prepare_mlp_weight_pair(self, x, transformer_options) as (prepared_weights, expanded_bytes):
                            expanded_mib = None
                            result = execute_chunks(prepared_weights)
                    except _WeightPairUnsupported as exc:
                        fallback_exc = (type(exc).__name__, str(exc), True)
                    except Exception as exc:
                        if not _is_weight_pair_resource_error(exc):
                            raise
                        fallback_exc = (type(exc).__name__, str(exc), False)
                    if fallback_exc is not None:
                        prepared_weights = None
                        fallback_key = (x.device.index, block_index, fallback_exc[0], fallback_exc[1][:160])
                        if fallback_key not in _weight_pair_fallback_reported:
                            _weight_pair_fallback_reported.add(fallback_key)
                            LOGGER.warning('H3 Dynamic MLP weight-pair reuse fallback: block=%d chunks=%d reason=%s. Restoring the validated per-chunk cast path.', block_index, chunks, fallback_exc[1])
                        torch.cuda.synchronize(x.device)
                        torch.cuda.empty_cache()
                        result = execute_chunks()
            else:
                result = execute_chunks()
        _trim_if_needed(x.device, transformer_options)
        return result
    setattr(chunked_forward, PATCH_MARKER, True)
    setattr(chunked_forward, ORIGINAL_FORWARD_ATTR, _unwrap_our_forward(original_forward))
    setattr(chunked_forward, CONTROLLER_ATTR, expected_dynamic_vbar_controller)
    return chunked_forward

class H3TokenwiseMLPChunking:
    """Adapter for H3's validated [tokens, hidden] token-independent MLPs."""

    def patch(self, model, chunk_tokens=512, cache_trim=True, cache_trim_threshold_mb=2048, adaptive=False, experimental_fp16=False, scaled_fp16_swiglu=False, dynamic_vbar_controller=None):
        chunk_tokens = max(0, int(chunk_tokens))
        if chunk_tokens == 0:
            return (model,)
        patched = model.clone()
        diffusion_model = patched.get_model_object('diffusion_model')
        blocks = getattr(diffusion_model, 'blocks', None)
        if not blocks:
            raise RuntimeError('H3 MLP chunking expected diffusion_model.blocks.')
        transformer_options = patched.model_options.setdefault('transformer_options', {})
        transformer_options[OPTION_KEY] = chunk_tokens
        transformer_options[ADAPTIVE_KEY] = bool(adaptive)
        transformer_options[CACHE_TRIM_KEY] = bool(cache_trim)
        transformer_options[CACHE_TRIM_THRESHOLD_KEY] = max(0, int(cache_trim_threshold_mb))
        transformer_options[QKV_CHUNKING_KEY] = bool(adaptive)
        transformer_options[QKV_CHUNK_TOKENS_KEY] = 1024
        transformer_options[QKV_CHUNK_THRESHOLD_KEY] = 38000
        transformer_options[QKV_CACHE_TRIM_THRESHOLD_KEY] = max(0, int(cache_trim_threshold_mb))
        transformer_options[EXPERIMENTAL_FP16_KEY] = bool(experimental_fp16)
        transformer_options[SCALED_FP16_SWIGLU_KEY] = bool(experimental_fp16 and scaled_fp16_swiglu)
        if dynamic_vbar_controller is not None:
            transformer_options[CONTROLLER_KEY] = dynamic_vbar_controller
        count = 0
        for index, block in enumerate(blocks):
            mlp = getattr(block, 'mlp', None)
            if mlp is None or not all((hasattr(mlp, name) for name in ('fc1', 'fc2'))):
                raise RuntimeError(f'H3 MLP chunking rejected block {index}: incompatible MLP.')
            key = f'diffusion_model.blocks.{index}.mlp.forward'
            existing = patched.object_patches.get(key)
            if existing is not None:
                function = getattr(existing, '__func__', existing)
                if not getattr(function, PATCH_MARKER, False):
                    raise RuntimeError(f'H3 MLP chunking found another patch at {key}.')
                base_forward = _unwrap_our_forward(existing)
            else:
                base_forward = _unwrap_our_forward(mlp.forward)
            patched.add_object_patch(key, types.MethodType(_make_forward(base_forward, index, transformer_options, dynamic_vbar_controller), mlp))
            count += 1
        if dynamic_vbar_controller is not None:
            for index in range(count):
                key = f'diffusion_model.blocks.{index}.mlp.forward'
                function = getattr(patched.object_patches[key], '__func__', None)
                if getattr(function, CONTROLLER_ATTR, None) is not dynamic_vbar_controller:
                    raise RuntimeError(f'H3 native Dynamic VBAR integration check failed at MLP block {index}.')
        return (patched,)
