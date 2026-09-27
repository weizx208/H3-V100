"""Calibrated fused Sol speed route for the stable H3 V100 node."""
from __future__ import annotations
import logging
import math
from collections import OrderedDict
from copy import deepcopy
import threading
import torch
from .dual_runtime import _fatal_device_failure
from .device_support import is_sm70_device
from .flash_attention import SOL_EXACT_RANGE_KEY, SOL_EXACT_RANGE_REQUIRED_KEY
from .sol_range import CORRECTED_STREAM_CHUNK_TOKENS, CorrectedSolRangeStream, run_fused_sol_active
from .exact_flash_range import make_exact_flash_range_stream
from .runtime_memory import POLICY_KEY, request_cuda_headroom
from .sol_calibration import AUDIO_OVERWRITE_ACTIVE_KEY, AUDIO_RANGES_KEY, CALIBRATION_LAYERS, HardSparseSpeedState, PROTECTED_LAYERS, STATE_KEY, STABLE_VBAR_KEY, STREAM_OUTPUT_KEY, _calibrate, _cuda_time, _MemoryBudgetExceeded
LOGGER = logging.getLogger('H3V100FusedSolSpeed')
STABILITY_POLICY_KEY = 'v100_h3_sol_stability_policy'
ADAPTIVE_BUDGET_POLICY_KEY = 'v100_h3_sol_adaptive_budget_policy'
CORRECTED_STREAM_BASE_TOKENS = 38000
CORRECTED_STREAM_SAFETY_MIB = 512
SOL_ROUTE_PROJECTION_RESERVE_MIB = 576
SOL_ROUTE_TARGET_OFFSETS_MIB = 256
SOURCE_MODEL_FINGERPRINT_KEY = 'v100_h3_source_model_fingerprint'
ADMISSION_CACHE_VERSION = 6
ADMISSION_CACHE_MAX_ENTRIES = 32
ADMISSION_CACHE_REUSES = 1
AUDIO_GUARD_MIN_ROWS = 768
AUDIO_GUARD_END_PERCENT = 0.7
_admission_cache = OrderedDict()
_admission_cache_lock = threading.Lock()
DUAL_ROUTE_EXACT = 'dual-exact-owner'
DUAL_ROUTE_SOL = 'sol-owner'
DUAL_ROUTE_SOL_SPARSE = 'dual-sol-sparse-owner'

def _admission_cache_key(state, signature):
    fingerprint = getattr(state, 'source_model_fingerprint', None)
    schedule = getattr(state, 'sigma_schedule', ())
    if not fingerprint or not schedule:
        return None
    fingerprint_key = tuple((str(value) for value in fingerprint)) if isinstance(fingerprint, (tuple, list)) else (str(fingerprint),)
    return (ADMISSION_CACHE_VERSION, fingerprint_key, signature, int(state.min_tokens), float(state.minimum_gain), int(state.memory_limit_mib), float(state.start_percent), float(state.end_percent), int(getattr(state, 'step_count', 0)), schedule, str(getattr(state, 'quality_profile', 'quality')), int(getattr(state, 'audio_guard_min_rows', 0)), getattr(state, 'audio_guard_end_percent', None))

def _take_cached_admission(state, signature):
    key = _admission_cache_key(state, signature)
    if key is None:
        return None
    with _admission_cache_lock:
        entry = _admission_cache.get(key)
        if entry is None:
            return None
        remaining = int(entry.get('remaining_reuses', 0))
        if remaining <= 0:
            _admission_cache.pop(key, None)
            return None
        admission = deepcopy(entry['admission'])
        remaining -= 1
        if remaining <= 0:
            _admission_cache.pop(key, None)
        else:
            entry['remaining_reuses'] = remaining
            _admission_cache.move_to_end(key)
    admission['cache_hit'] = True
    admission['cache_policy'] = 'reuse-once-then-recalibrate'
    return admission

def _store_cached_admission(state, signature, admission):
    if admission.get('status') != 'ready':
        return False
    key = _admission_cache_key(state, signature)
    if key is None:
        return False
    compact = {'status': 'ready', 'admitted': bool(admission.get('admitted', False)), 'tokens': int(admission['tokens']), 'tau': float(admission['tau']), 'calibration_step': int(admission.get('calibration_step', -1)), 'calibration_phase': admission.get('calibration_phase', 'active-window'), 'active_start_step': admission.get('active_start_step'), 'calibration_layers': list(admission.get('calibration_layers', ())), 'stream_output': bool(admission.get('stream_output', False)), 'stream_reason': admission.get('stream_reason'), 'estimated_full_intermediate_peak_mib': admission.get('estimated_full_intermediate_peak_mib'), 'stream_chunk_tokens': admission.get('stream_chunk_tokens'), 'mean_ratio': admission.get('mean_ratio'), 'worst_ratio': admission.get('worst_ratio'), 'samples': []}
    with _admission_cache_lock:
        _admission_cache[key] = {'admission': compact, 'remaining_reuses': ADMISSION_CACHE_REUSES}
        _admission_cache.move_to_end(key)
        while len(_admission_cache) > ADMISSION_CACHE_MAX_ENTRIES:
            _admission_cache.popitem(last=False)
    return True

def _cleanup_rejected_calibration(device):
    """Release only unused PyTorch blocks left by a rejected calibration."""
    if getattr(device, 'type', None) != 'cuda':
        return False
    cached_bytes = max(0, int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device)))
    if cached_bytes < 512 * 2 ** 20:
        return False
    torch.cuda.empty_cache()
    return True

def _signature(q, prefix_stop, video_grid, tau, stream_output=None, block_layout=None, stability_policy_tag=None, route_policy_tag=None):
    return (str(q.device), tuple((int(value) for value in q.shape)), int(prefix_stop), tuple((int(value) for value in video_grid)), float(tau), str(block_layout or 'row_major'), stream_output, str(stability_policy_tag or 'base'), str(route_policy_tag or 'base-route'))

def _adaptive_budget_policy(transformer_options, tokens=None):
    value = transformer_options.get(ADAPTIVE_BUDGET_POLICY_KEY)
    if not isinstance(value, dict) or not value.get('enabled'):
        return None
    if value.get('owner') == 'main' and (transformer_options.get(STABILITY_POLICY_KEY) is not None or any((key.startswith('v100_h3_') and key.endswith('_trace') for key in transformer_options))):
        return None
    if tokens is not None and (not int(value.get('min_tokens', 0)) <= int(tokens) <= int(value.get('max_tokens', 2 ** 31 - 1))):
        return None
    if not callable(value.get('route_builder')):
        return None
    return value

def _release_adaptive_route_history(route_policy):
    if not isinstance(route_policy, dict):
        return
    release = getattr(route_policy.get('state'), 'release_history', None)
    if callable(release):
        release()

def _stability_policy(transformer_options, tokens=None):
    value = transformer_options.get(STABILITY_POLICY_KEY)
    if not isinstance(value, dict) or not value.get('enabled'):
        return None
    if tokens is not None and (not int(value.get('min_tokens', 0)) <= int(tokens) <= int(value.get('max_tokens', 2 ** 31 - 1))):
        return None
    return value

def _additional_protected_layers(transformer_options, tokens=None):
    policy = _stability_policy(transformer_options, tokens)
    if policy is None:
        return ()
    try:
        return tuple((int(value) for value in policy.get('protected_layers', ())))
    except (TypeError, ValueError):
        return ()

def _layer_tau_delta(transformer_options, block_index, tokens=None):
    policy = _stability_policy(transformer_options, tokens)
    if policy is None or block_index is None:
        return 0.0
    block = int(block_index)
    for start, stop, delta in policy.get('tau_bands', ()):
        if int(start) <= block <= int(stop):
            return float(delta)
    return 0.0

def _audio_row_count(transformer_options):
    total = 0
    for value in transformer_options.get(AUDIO_RANGES_KEY, ()) or ():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            continue
        start, stop = value
        if isinstance(start, int) and isinstance(stop, int) and (stop > start):
            total += stop - start
    return total

def _effective_sol_end_percent(state, transformer_options):
    configured = float(state.end_percent)
    guard_end = getattr(state, 'audio_guard_end_percent', None)
    minimum_rows = int(getattr(state, 'audio_guard_min_rows', 0))
    if guard_end is not None and minimum_rows > 0 and (str(getattr(state, 'quality_profile', 'quality')) != 'manual') and (_audio_row_count(transformer_options) >= minimum_rows):
        return (min(configured, float(guard_end)), True)
    return (configured, False)

def _sol_window_phase(state, transformer_options):
    """Return active, speed-precalibration, or outside for this denoise step.

    Speed/Ultra pre-calibration is limited to the single non-edge schedule ordinal
    immediately before the configured active window.  It may measure the three
    admission layers, but callers must keep its returned attention output exact.
    """
    if state.current_step is None or not state.step_count:
        return 'outside'
    step = int(state.current_step)
    step_count = int(state.step_count)
    denominator = max(1, step_count - 1)
    progress = step / denominator
    effective_end, _audio_guard_active = _effective_sol_end_percent(state, transformer_options)
    active_start = float(state.start_percent)
    if active_start <= progress <= effective_end:
        return 'active'
    next_progress = (step + 1) / denominator
    if str(getattr(state, 'quality_profile', 'quality')) in ('speed', 'ultra') and 0 < step < step_count - 1 and (progress < active_start <= next_progress <= effective_end):
        return 'speed-precalibration'
    return 'outside'

def _first_active_step(state, transformer_options):
    if not state.step_count:
        return None
    denominator = max(1, int(state.step_count) - 1)
    effective_end, _audio_guard_active = _effective_sol_end_percent(state, transformer_options)
    for step in range(int(state.step_count)):
        progress = step / denominator
        if float(state.start_percent) <= progress <= effective_end:
            return step
    return None

def dual_high_level_route(state, transformer_options, tokens):
    """Choose an owner before QKV allocation for a Sol-configured call.

    Calls already proven exact may move to the dual adapter. Calibration stays
    on Sol's validated primary-device owner because it depends on projected
    Q/K/V and native timing. Once base Sol is admitted and no alternate route
    policy owns its metadata, sparse work may be split by the dual adapter.
    This performs no CUDA allocation or device probe.
    """
    if not isinstance(state, HardSparseSpeedState):
        return (DUAL_ROUTE_SOL, 'missing-sol-state')
    if not isinstance(transformer_options, dict):
        return (DUAL_ROUTE_SOL, 'invalid-transformer-options')
    block_index = transformer_options.get('v100_h3_block_index')
    block_count = transformer_options.get('v100_h3_block_count')
    state.observe_step(transformer_options, block_index)
    tokens = int(tokens)
    try:
        normalized_block = None if block_index is None else int(block_index)
    except (TypeError, ValueError):
        normalized_block = None
    route_policy = _adaptive_budget_policy(transformer_options, tokens)
    if route_policy is not None and normalized_block == 0 and (state.current_step is not None) and state.step_count:
        progress = state.current_step / max(1, state.step_count - 1)
        effective_end, _audio_guard = _effective_sol_end_percent(state, transformer_options)
        if progress > effective_end:
            _release_adaptive_route_history(route_policy)
    if tokens < int(state.min_tokens):
        return (DUAL_ROUTE_EXACT, 'below-sol-min-tokens')
    if block_index is None or block_count is None:
        return (DUAL_ROUTE_EXACT, 'missing-sol-block-metadata')
    try:
        block_index = int(block_index)
        block_count = int(block_count)
    except (TypeError, ValueError):
        return (DUAL_ROUTE_EXACT, 'invalid-sol-block-metadata')
    if not 0 <= block_index < block_count:
        return (DUAL_ROUTE_EXACT, 'invalid-sol-block-index')
    protected_layers = {*(int(value) for value in PROTECTED_LAYERS), *(int(value) for value in _additional_protected_layers(transformer_options, tokens))}
    if block_index in protected_layers:
        return (DUAL_ROUTE_EXACT, 'sol-protected-layer')
    phase = _sol_window_phase(state, transformer_options)
    if phase == 'outside':
        if state.current_step is not None and state.step_count:
            progress = state.current_step / max(1, state.step_count - 1)
            effective_end, audio_guard_active = _effective_sol_end_percent(state, transformer_options)
        return (DUAL_ROUTE_EXACT, 'outside-sol-window')
    if phase == 'speed-precalibration':
        return (DUAL_ROUTE_SOL, 'sol-precalibration')
    prefix_stop = int(transformer_options.get('v100_sol_attention_prefix_stop', 0))
    video_grid = transformer_options.get('v100_sol_h3_video_grid')
    try:
        complete_layout = bool(0 <= prefix_stop < tokens and isinstance(video_grid, (tuple, list)) and (len(video_grid) == 3) and all((int(value) > 0 for value in video_grid)) and (math.prod((int(value) for value in video_grid)) == tokens - prefix_stop))
    except (TypeError, ValueError):
        complete_layout = False
    if not complete_layout:
        return (DUAL_ROUTE_EXACT, 'incomplete-sol-layout')
    matching = [admission for admission in state.admissions.values() if int(admission.get('tokens', -1)) == tokens]
    if not matching:
        return (DUAL_ROUTE_SOL, 'sol-admission-pending')
    if any((admission.get('status') == 'calibrating' or int(admission.get('calibration_step', -1)) == int(state.current_step) for admission in matching)):
        return (DUAL_ROUTE_SOL, 'sol-calibration-owner')
    if any((bool(admission.get('admitted', False)) for admission in matching)):
        if route_policy is None:
            return (DUAL_ROUTE_SOL_SPARSE, 'sol-sparse-dual')
        return (DUAL_ROUTE_SOL, 'sol-sparse-policy-owner')
    if all((admission.get('status') in ('ready', 'failed') for admission in matching)):
        return (DUAL_ROUTE_EXACT, 'sol-admission-rejected')
    return (DUAL_ROUTE_SOL, 'sol-admission-unresolved')

def _eligible(state, q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs, transformer_options):
    if mask is not None or not skip_reshape or skip_output_reshape:
        return False
    if kwargs.get('enable_gqa', False):
        return False
    if not (q.ndim == k.ndim == v.ndim == 4 and q.shape == k.shape == v.shape and (int(q.shape[1]) == int(heads)) and (int(q.shape[-1]) == 128) and (int(q.shape[2]) >= int(state.min_tokens))):
        return False
    if any((value.dtype != torch.float16 for value in (q, k, v))):
        return False
    if any((value.requires_grad for value in (q, k, v))):
        return False
    if not q.is_cuda or not is_sm70_device(q.device):
        return False
    if any((value.stride(-1) != 1 for value in (q, k, v))):
        return False
    block_index = transformer_options.get('v100_h3_block_index')
    block_count = transformer_options.get('v100_h3_block_count')
    if block_index is None or block_count is None:
        return False
    protected_layers = {*(int(value) for value in PROTECTED_LAYERS), *(int(value) for value in _additional_protected_layers(transformer_options, int(q.shape[2])))}
    if int(block_index) in protected_layers:
        return False
    if state.current_step is None or not state.step_count:
        return False
    progress = state.current_step / max(1, state.step_count - 1)
    effective_end, audio_guard_active = _effective_sol_end_percent(state, transformer_options)
    if _sol_window_phase(state, transformer_options) == 'outside':
        return False
    prefix_stop = int(transformer_options.get('v100_sol_attention_prefix_stop', 0))
    video_grid = transformer_options.get('v100_sol_h3_video_grid')
    return bool(0 <= prefix_stop < int(q.shape[2]) and isinstance(video_grid, (tuple, list)) and (len(video_grid) == 3) and (math.prod((int(value) for value in video_grid)) == int(q.shape[2]) - prefix_stop))

def _stream_decision(q, transformer_options):
    """Choose bounded output when the normal full-output peak is wasteful."""
    if not transformer_options.get(STREAM_OUTPUT_KEY, False):
        return (False, 'consumer-unavailable', None, None)
    tokens = int(q.shape[2])
    estimated_peak = int(q.numel()) * 9
    driver_free = None
    if q.is_cuda:
        driver_free = int(torch.cuda.mem_get_info(q.device)[0])
    configured_threshold = max(CORRECTED_STREAM_BASE_TOKENS, int(transformer_options.get('v100_h3_qkv_chunk_threshold', 38000)))
    if tokens >= configured_threshold:
        reason = 'projection-chunking-tier'
        enabled = True
    elif driver_free is not None and driver_free < estimated_peak + CORRECTED_STREAM_SAFETY_MIB * 2 ** 20:
        required = estimated_peak + CORRECTED_STREAM_SAFETY_MIB * 2 ** 20
        if POLICY_KEY in transformer_options:
            reclaim = request_cuda_headroom(q.device, transformer_options, reason='sol-full-output-admission', required_free_bytes=required, minimum_reclaimable_mib=256)
            driver_free = int(reclaim['after']['free_bytes'])
        enabled = driver_free < required
        reason = 'runtime-memory-budget' if enabled else 'full-output-after-cache-reclaim'
    else:
        reason = 'full-output-profitable'
        enabled = False
    return (enabled, reason, estimated_peak / 2 ** 20, None if driver_free is None else driver_free / 2 ** 20)

def _prepare_sol_route_headroom(q, state, transformer_options):
    """Acquire route headroom from allocator cache, then unpinned native pages.

    QKV has already been projected on both weight profiles. Keep its live
    tensors and all native pins intact; the shared owner may evict only the
    remaining driver-visible deficit from unpinned resident weights. This is
    scoped to route preparation, not the optional full-output upgrade.
    """
    if not q.is_cuda or POLICY_KEY not in transformer_options:
        return None
    batch, heads, tokens, width = (int(value) for value in q.shape)
    blocks = math.ceil(tokens / 64)
    words = math.ceil(blocks / 32)
    rows = batch * heads * blocks
    fixed_bytes = batch * heads * blocks * words * 4 + (rows + 1) * 4 + batch * heads * blocks * width * 2 * 2
    prepare_bytes = batch * heads * blocks * width * 4 * 3 + rows * 4
    route_cap_bytes = max(128, int(state.memory_limit_mib)) * 2 ** 20
    desired_route_bytes = min(route_cap_bytes, fixed_bytes + SOL_ROUTE_TARGET_OFFSETS_MIB * 2 ** 20)
    required = desired_route_bytes + prepare_bytes + SOL_ROUTE_PROJECTION_RESERVE_MIB * 2 ** 20
    reclaim = request_cuda_headroom(q.device, transformer_options, reason='sol-route-prepare', required_free_bytes=int(required), minimum_reclaimable_mib=256, allow_vbar_release=True)
    available = int(reclaim['after']['free_bytes'])
    if available < required:
        raise _MemoryBudgetExceeded(f'Sol route preflight needs {required / 2 ** 20:.1f} MiB including prepare, route and projection reserve, only {available / 2 ** 20:.1f} MiB driver-visible free')
    return reclaim

def _make_override(state, stable_override):

    def override(original, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        transformer_options = kwargs.get('transformer_options') or {}
        block_index = transformer_options.get('v100_h3_block_index')
        state.observe_step(transformer_options, block_index)
        route_policy_for_lifetime = _adaptive_budget_policy(transformer_options, int(q.shape[2]))
        if route_policy_for_lifetime is not None and block_index is not None and (int(block_index) == 0) and (state.current_step is not None) and state.step_count:
            progress = state.current_step / max(1, state.step_count - 1)
            effective_end, _audio_guard = _effective_sol_end_percent(state, transformer_options)
            if progress > effective_end:
                _release_adaptive_route_history(route_policy_for_lifetime)

        def exact_call(*, range_allowed=True, range_required=False):
            missing = object()
            previous = transformer_options.get(SOL_EXACT_RANGE_KEY, missing)
            previous_required = transformer_options.get(SOL_EXACT_RANGE_REQUIRED_KEY, missing)
            transformer_options[SOL_EXACT_RANGE_KEY] = bool(range_allowed)
            transformer_options[SOL_EXACT_RANGE_REQUIRED_KEY] = bool(range_required)
            try:
                return stable_override(original, q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
            finally:
                if previous is missing:
                    transformer_options.pop(SOL_EXACT_RANGE_KEY, None)
                else:
                    transformer_options[SOL_EXACT_RANGE_KEY] = previous
                if previous_required is missing:
                    transformer_options.pop(SOL_EXACT_RANGE_REQUIRED_KEY, None)
                else:
                    transformer_options[SOL_EXACT_RANGE_REQUIRED_KEY] = previous_required
        if not _eligible(state, q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs, transformer_options):
            return exact_call()
        prefix_stop = int(transformer_options.get('v100_sol_attention_prefix_stop', 0))
        video_grid = transformer_options['v100_sol_h3_video_grid']
        stability_policy = _stability_policy(transformer_options, int(q.shape[2]))
        stability_policy_tag = stability_policy.get('tag') if stability_policy is not None else None
        route_policy = _adaptive_budget_policy(transformer_options, int(q.shape[2]))
        route_policy_tag = route_policy.get('tag') if route_policy is not None else None
        base_signature = _signature(q, prefix_stop, video_grid, state.tau, stream_output=None, block_layout=transformer_options.get('v100_sol_block_layout'), stability_policy_tag=stability_policy_tag, route_policy_tag=route_policy_tag)
        density_holder = {}
        stream_modes = getattr(state, 'stream_modes', None)
        if stream_modes is None:
            stream_modes = {}
            state.stream_modes = stream_modes
        stream_policy = stream_modes.get(base_signature)
        if stream_policy is None:
            stream_policy = _stream_decision(q, transformer_options)
            stream_modes[base_signature] = stream_policy
        stream_output, stream_reason, estimated_peak_mib, driver_free_mib = stream_policy
        signature = _signature(q, prefix_stop, video_grid, state.tau, stream_output=bool(stream_output), block_layout=transformer_options.get('v100_sol_block_layout'), stability_policy_tag=stability_policy_tag, route_policy_tag=route_policy_tag)
        audio_overwrite_active = bool(transformer_options.get(AUDIO_OVERWRITE_ACTIVE_KEY, False))
        audio_ranges = transformer_options.get(AUDIO_RANGES_KEY, ()) if audio_overwrite_active else ()
        effective_tau = max(0.0, min(4.0, float(state.tau) + _layer_tau_delta(transformer_options, block_index, int(q.shape[2]))))

        def candidate_call(stream_probe=False):
            _prepare_sol_route_headroom(q, state, transformer_options)
            output, density = run_fused_sol_active(q, k, v, tau=effective_tau, prefix_stop=prefix_stop, scale=kwargs.get('scale'), stream_output=stream_output, stream_chunk_tokens=CORRECTED_STREAM_CHUNK_TOKENS, memory_limit_mib=state.memory_limit_mib, audio_ranges=audio_ranges, audio_overwrite_active=audio_overwrite_active, route_builder=route_policy.get('route_builder') if route_policy is not None else None, route_context={'step_index': int(state.current_step), 'block_index': int(block_index)})
            density_holder['value'] = density
            if stream_probe and isinstance(output, CorrectedSolRangeStream):
                return output.consume_probe()
            return output
        admission = state.admissions.get(signature)
        if admission is None:
            admission = _take_cached_admission(state, signature)
            if admission is None:
                admission = {'status': 'calibrating', 'admitted': False, 'tokens': int(q.shape[2]), 'tau': float(state.tau), 'stability_policy': stability_policy_tag, 'calibration_step': int(state.current_step), 'calibration_phase': _sol_window_phase(state, transformer_options), 'active_start_step': _first_active_step(state, transformer_options), 'calibration_layers': list(CALIBRATION_LAYERS), 'stream_output': bool(stream_output), 'stream_reason': stream_reason, 'estimated_full_intermediate_peak_mib': estimated_peak_mib, 'driver_free_mib': driver_free_mib, 'stream_chunk_tokens': CORRECTED_STREAM_CHUNK_TOKENS if stream_output else None, 'samples': []}
            state.admissions[signature] = admission
        if admission['status'] == 'calibrating':
            if int(state.current_step) != int(admission['calibration_step']):
                return exact_call()
            layer = int(block_index)
            sampled = {int(sample['block_index']) for sample in admission['samples']}
            if layer not in CALIBRATION_LAYERS or layer in sampled:
                return exact_call()

            def finish_sample(sample):
                sample['block_index'] = layer
                sample['route_density'] = density_holder.get('value')
                admission['samples'].append(sample)
                if len(admission['samples']) != len(CALIBRATION_LAYERS):
                    return
                ratios = [float(value['ratio']) for value in admission['samples']]
                finite = all((bool(value['finite']) for value in admission['samples']))
                admission['mean_ratio'] = sum(ratios) / len(ratios)
                admission['worst_ratio'] = max(ratios)
                admission['admitted'] = bool(finite and admission['mean_ratio'] <= 1.0 - state.minimum_gain and (admission['worst_ratio'] <= 1.02))
                admission['status'] = 'ready'
                _store_cached_admission(state, signature, admission)
                if not admission['admitted']:
                    _release_adaptive_route_history(route_policy)
                    _cleanup_rejected_calibration(q.device)
            try:
                if stream_output:
                    if not admission['samples']:
                        warm_output = candidate_call(stream_probe=True)
                        torch.cuda.synchronize(q.device)
                        del warm_output
                    candidate, candidate_ms = _cuda_time(lambda: candidate_call(stream_probe=True), q.device)
                    finite = bool(torch.isfinite(candidate).all().item())
                    del candidate
                    exact = exact_call(range_required=True)
                    attach_timing = getattr(exact, 'attach_kernel_timing', None)
                    if not callable(attach_timing):
                        raise RuntimeError('bounded Sol calibration requires an exact range stream')
                    route_density = density_holder.get('value')

                    def exact_timing_complete(exact_ms, timing_error):
                        if timing_error is not None or exact_ms is None:
                            admission.update({'status': 'failed', 'admitted': False, 'reason': f'bounded-exact-timing-{type(timing_error).__name__}'})
                            LOGGER.warning('H3 V100 bounded Sol calibration failed open: %s', admission['reason'])
                            return
                        ratio = float(candidate_ms) / max(float(exact_ms), 1e-06)
                        density_holder['value'] = route_density
                        finish_sample({'admitted': bool(finite and ratio <= 1.0 - state.minimum_gain), 'candidate_ms': float(candidate_ms), 'flash_ms': float(exact_ms), 'ratio': ratio, 'gain': 1.0 - ratio, 'finite': finite, 'calibration_semantics': 'bounded-range-vs-range'})
                    attach_timing(exact_timing_complete)
                    return exact
                exact, sample = _calibrate(candidate_call, lambda: exact_call(range_allowed=False), state.minimum_gain, q.device, bool(not admission['samples']))
                finish_sample(sample)
                return exact
            except Exception as error:
                if _fatal_device_failure(error):
                    raise
                admission.update({'status': 'failed', 'admitted': False, 'reason': type(error).__name__})
                LOGGER.warning('H3 V100 fused Sol calibration failed open to Flash: %s', admission['reason'])
            warm_output = candidate = exact = attach_timing = None
            _release_adaptive_route_history(route_policy)
            _cleanup_rejected_calibration(q.device)
            return exact_call()
        if int(state.current_step) == int(admission.get('calibration_step', -1)):
            if admission.get('cache_hit', False) and int(block_index) in CALIBRATION_LAYERS:
                # A cache hit skips timing, not the cold calibration's exact
                # math path. Otherwise warm runs can switch full Flash to
                # bounded dense attention at these layers under pressure.
                bounded = bool(admission.get('stream_output', False))
                return exact_call(range_allowed=bounded, range_required=bounded)
            return exact_call()
        if _sol_window_phase(state, transformer_options) == 'speed-precalibration':
            return exact_call()
        if not admission.get('admitted', False):
            return exact_call()
        try:
            output = candidate_call()
            density = density_holder.get('value')
            if isinstance(output, CorrectedSolRangeStream):

                def mark_stream_failure(error):
                    _release_adaptive_route_history(route_policy)
                    admission['admitted'] = False
                    admission['reason'] = f'runtime-{type(error).__name__}' if not isinstance(error, tuple) else 'runtime-StreamFailure'
                    LOGGER.warning('H3 V100 corrected Sol range stream failed open to Flash: %s', admission['reason'])
                output.attach_fallback(lambda: make_exact_flash_range_stream(q, k, v, scale=kwargs.get('scale'), chunk_tokens=CORRECTED_STREAM_CHUNK_TOKENS, audio_ranges=audio_ranges, audio_overwrite_active=audio_overwrite_active), mark_stream_failure)
            return output
        except Exception as error:
            if _fatal_device_failure(error):
                raise
            admission.update({'admitted': False, 'reason': f'runtime-{type(error).__name__}'})
            LOGGER.warning('H3 V100 fused Sol runtime failed open to Flash: %s', admission['reason'])
        output = None
        _release_adaptive_route_history(route_policy)
        if q.is_cuda:
            torch.cuda.empty_cache()
        return exact_call()
    override._v100_fused_sol_speed_override = True
    override._v100_fused_sol_speed_stable = stable_override
    return override

def fused_sol_speed_outer_sample_wrapper(executor, *args, **kwargs):
    """Reset calibrated fused/streamed Sol admission for one sample."""
    guider = getattr(executor, 'class_obj', None)
    model_options = getattr(guider, 'model_options', {})
    transformer_options = model_options.get('transformer_options', {})
    state = transformer_options.get(STATE_KEY)
    if not isinstance(state, HardSparseSpeedState):
        return executor(*args, **kwargs)
    sample_sigmas = kwargs.get('sigmas')
    if sample_sigmas is None and len(args) > 3:
        sample_sigmas = args[3]
    state.reset_run(transformer_options, sample_sigmas=sample_sigmas)
    state.stream_modes = {}
    route_policy = transformer_options.get(ADAPTIVE_BUDGET_POLICY_KEY)
    if isinstance(route_policy, dict):
        reset_route = getattr(route_policy.get('state'), 'reset', None)
        if callable(reset_route):
            reset_route()
    try:
        return executor(*args, **kwargs)
    finally:
        _release_adaptive_route_history(route_policy)

def patch_model_for_fused_sol_speed(model, *, tau=1.0, min_tokens=16384, minimum_gain_percent=5.0, start_percent=0.2, end_percent=0.8, memory_limit_mib=1024, quality_profile='quality'):
    import comfy.patcher_extension
    patched = model.clone()
    patched.model_options = dict(patched.model_options)
    options = dict(patched.model_options.get('transformer_options', {}))
    patched.model_options['transformer_options'] = options
    if STABLE_VBAR_KEY not in options:
        raise RuntimeError('Fused Sol requires the stable H3 V100 policy.')
    if options.get('v100_attention_backend', 'flash_attn') != 'flash_attn':
        raise RuntimeError('Fused Sol wraps the exact Flash backend.')
    stable = options.get('optimized_attention_override')
    if not callable(stable) or not getattr(stable, '_v100_flash_attention_override', False):
        raise RuntimeError('Fused Sol requires the stable exact Flash override.')
    if not 0.0 <= float(start_percent) < float(end_percent) <= 1.0:
        raise ValueError('Fused Sol requires 0 <= start < end <= 1')
    state = HardSparseSpeedState(tau=float(tau), topk_tail_blocks=0, min_tokens=max(1024, int(min_tokens)), minimum_gain=float(minimum_gain_percent) / 100.0, memory_limit_mib=max(128, int(memory_limit_mib)))
    state.start_percent = float(start_percent)
    state.end_percent = float(end_percent)
    state.quality_profile = str(quality_profile)
    state.audio_guard_min_rows = AUDIO_GUARD_MIN_ROWS
    state.audio_guard_end_percent = AUDIO_GUARD_END_PERCENT
    state.source_model_fingerprint = options.get(SOURCE_MODEL_FINGERPRINT_KEY)
    options[STATE_KEY] = state
    options['optimized_attention_override'] = _make_override(state, stable)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, 'v100_h3_fused_sol_speed', fused_sol_speed_outer_sample_wrapper)
    return (patched, state)
