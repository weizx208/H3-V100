"""Two-V100 exact and calibrated base-Sol adapter for the main optimize node.

It composes with the validated H3 V100 mixed-precision patch and uses ComfyUI
once to prepare each FP8 or INT8-ConvRot/LoRA
projection on the primary device, and then runs independent head shards on two
V100s.  All cross-device traffic is explicitly staged through pinned CPU RAM.
"""
from __future__ import annotations
import concurrent.futures
from contextlib import ExitStack
from dataclasses import dataclass, field
import logging
import math
import threading
import types
import torch
import torch.nn.functional as F
from .weight_profile import FP8_E4M3_PROFILE, INT8_CONVROT_PROFILE, dual_attention_decode_extra_bytes
from .dual_runtime import DualAdaptiveAdmission, HostMemory, OperationTransaction, system_host_memory, torch_device_memory, validate_dual_v100_pair
LOGGER = logging.getLogger('H3V100DualExactAttention')
OPTION_KEY = 'v100_h3_dual_exact_attention_state'
PATCH_MARKER = '_h3_v100_dual_exact_attention'
BLOCK_PATCH_MARKER = '_h3_v100_dual_exact_block'
SELECTED_BACKEND_KEY = 'v100_h3_selected_attention_backend'
LIFECYCLE_WRAPPER_KEY = 'v100_h3_dual_attention_lifecycle'
ADAPTIVE_BUDGET_POLICY_KEY = 'v100_h3_sol_adaptive_budget_policy'
WDDM_REBALANCE_MIN_TOKENS = 65536
WDDM_REBALANCE_MIN_HEAD_GAIN = 4
WDDM_REBALANCE_MIN_RECLAIMABLE_MIB = 512
WDDM_LEASE_MIN_DRIVER_FREE_MIB = 512
RESOURCE_REPROBES_PER_SAMPLE = 2
_DUAL_RUN_LOCK = threading.Lock()
from .dual_runtime import _fatal_device_failure

def _resource_failure(error):
    """Only allocation pressure may be retried on the next sample."""
    return isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) or (isinstance(error, RuntimeError) and any((text in str(error).lower() for text in ('out of memory', "can't allocate memory", 'cannot allocate memory', 'cublas_status_alloc_failed', 'cuda error: memory allocation'))))

def _recoverable_probe_failure(error):
    if _fatal_device_failure(error):
        return False
    return _resource_failure(error) or isinstance(error, OSError) or (isinstance(error, RuntimeError) and any((text in str(error).lower() for text in ('invalid device ordinal', 'devices are busy or unavailable', 'device is busy', 'device is unavailable'))))

class _ProbeUnavailable(RuntimeError):
    pass

def _parse_secondary_device(value):
    text = str(value or 'auto').strip().lower()
    if text == 'auto':
        return None
    if text.startswith('cuda:'):
        text = text.split(':', 1)[1]
    try:
        index = int(text)
    except ValueError as error:
        raise ValueError("dual_gpu_secondary must be 'auto' or a CUDA device such as cuda:1") from error
    if index < 0:
        raise ValueError('dual_gpu_secondary CUDA index must be non-negative')
    return index

def available_secondary_device_options():
    """List visible compatible V100 devices, excluding ComfyUI's primary."""
    choices = ['auto']
    if not torch.cuda.is_available():
        return tuple(choices)
    primary_index = None
    try:
        import comfy.model_management as model_management
        primary = model_management.get_torch_device()
        if getattr(primary, 'type', None) == 'cuda':
            primary_index = primary.index
    except Exception:
        pass
    if primary_index is None:
        try:
            primary_index = int(torch.cuda.current_device())
        except Exception:
            primary_index = None
    for index in range(torch.cuda.device_count()):
        if primary_index is not None and int(index) == int(primary_index):
            continue
        try:
            capability = tuple(torch.cuda.get_device_capability(index))
            name = str(torch.cuda.get_device_name(index))
        except Exception:
            continue
        if capability == (7, 0) and 'V100' in name.upper():
            choices.append(f'cuda:{index}')
    return tuple(choices)

def qkv_head_rows(weight, start_head, stop_head, heads, head_dim):
    """Slice Q, K and V row groups for one contiguous head interval."""
    start_head = int(start_head)
    stop_head = int(stop_head)
    heads = int(heads)
    head_dim = int(head_dim)
    if weight.ndim != 2 or not 0 <= start_head < stop_head <= heads:
        raise ValueError('invalid QKV head interval')
    inner = heads * head_dim
    if int(weight.shape[0]) != 3 * inner:
        raise ValueError('QKV weight has incompatible row count')
    start = start_head * head_dim
    stop = stop_head * head_dim
    return torch.cat((weight[start:stop], weight[inner + start:inner + stop], weight[2 * inner + start:2 * inner + stop]), dim=0).contiguous()

def out_head_columns(weight, start_head, stop_head, heads, head_dim):
    """Slice output-projection columns for one contiguous head interval."""
    start_head = int(start_head)
    stop_head = int(stop_head)
    heads = int(heads)
    head_dim = int(head_dim)
    if weight.ndim != 2 or not 0 <= start_head < stop_head <= heads:
        raise ValueError('invalid output-projection head interval')
    if int(weight.shape[1]) != heads * head_dim:
        raise ValueError('output projection has incompatible column count')
    return weight[:, start_head * head_dim:stop_head * head_dim].contiguous()

class DualAttentionRetry(RuntimeError):
    """Tell the enclosing block to rebuild its disposable AdaLN input."""

    def __init__(self, failure_type, failure_message, *, partial_target_write, fallback, devices, on_recovery=None):
        super().__init__(f'{failure_type}: {failure_message}')
        self.failure_type = str(failure_type)
        self.failure_message = str(failure_message)
        self.partial_target_write = bool(partial_target_write)
        self.fallback = fallback
        self.devices = tuple((int(value) for value in devices))
        self.on_recovery = on_recovery

class _DualAdmissionFallback(RuntimeError):
    """Leave the dual transaction before invoking the single-device path."""

class _PinnedPool:
    """One bounded host pool reused sequentially across transfer phases."""

    def __init__(self, input_chunk, query_chunk, hidden, weight_bytes):
        self.input_chunk = int(input_chunk)
        self.query_chunk = int(query_chunk)
        self.hidden = int(hidden)
        self.weight_bytes = int(weight_bytes)
        self.input = torch.empty((self.input_chunk, self.hidden), dtype=torch.float16, device='cpu', pin_memory=True)
        self.partial = torch.empty((self.query_chunk, self.hidden), dtype=torch.float32, device='cpu', pin_memory=True)
        self.raw = torch.empty(self.weight_bytes, dtype=torch.uint8, device='cpu', pin_memory=True)

    @property
    def allocated_bytes(self):
        return sum((int(value.numel()) * int(value.element_size()) for value in (self.input, self.partial, self.raw)))

    def admits(self, input_chunk, query_chunk, hidden, weight_bytes):
        return bool(self.input_chunk >= int(input_chunk) and self.query_chunk >= int(query_chunk) and (self.hidden == int(hidden)) and (self.weight_bytes >= int(weight_bytes)))

    def raw_view(self, dtype, shape):
        elements = math.prod((int(value) for value in shape))
        bytes_needed = elements * torch.empty((), dtype=dtype).element_size()
        if bytes_needed > self.weight_bytes:
            raise RuntimeError('pinned raw staging view exceeds admitted pool')
        return self.raw[:bytes_needed].view(dtype).view(*shape)

def _prepared_weight_bytes(secondary_heads, hidden, head_dim):
    qkv = 3 * int(secondary_heads) * int(head_dim) * int(hidden) * 2
    out = int(hidden) * int(secondary_heads) * int(head_dim) * 4
    return max(qkv, out, 2 * int(head_dim) * 4)

def _rope_chunk_bytes(rope, input_chunk):
    if rope is None:
        return 0
    if rope.ndim < 2 or int(rope.shape[1]) <= 0:
        raise ValueError('dual H3 RoPE must have a positive token dimension')
    shape = [int(value) for value in rope.shape]
    shape[1] = min(shape[1], int(input_chunk))
    return math.prod(shape) * int(rope.element_size())

def _pool_matches_admission(pool, admission, hidden, head_dim, query_chunk, rope=None, *, sol_route=False, total_heads=56):
    """Check the reusable host pool against the final head split and RAM cap."""
    if pool is None or not admission.enabled:
        return False
    required = max(_prepared_weight_bytes(int(total_heads) if sol_route else admission.attention.secondary_units, hidden, head_dim), _rope_chunk_bytes(rope, admission.staging.input_chunk_tokens))
    return bool(pool.admits(admission.staging.input_chunk_tokens, query_chunk, hidden, required) and pool.allocated_bytes <= admission.staging.pinned_limit_bytes)

def _copy_via_host(source, destination, host_view):
    """Copy CUDA -> pinned CPU -> CUDA with no peer-device transfer."""
    if source.device.type != 'cuda' or destination.device.type != 'cuda':
        raise ValueError('host-staged transfer requires two CUDA tensors')
    if source.device == destination.device:
        raise ValueError('host-staged transfer requires distinct CUDA devices')
    host_view.copy_(source, non_blocking=True)
    torch.cuda.synchronize(source.device)
    destination.copy_(host_view, non_blocking=True)
    torch.cuda.synchronize(destination.device)

def _stage_tensor(source, destination_device, pool, *, use_input=False):
    shape = tuple((int(value) for value in source.shape))
    if use_input:
        if source.ndim != 2 or shape[1] != pool.hidden or shape[0] > pool.input_chunk:
            raise RuntimeError('activation staging view exceeds admitted pool')
        host = pool.input[:shape[0]]
    else:
        host = pool.raw_view(source.dtype, shape)
    destination = torch.empty(shape, dtype=source.dtype, device=destination_device)
    _copy_via_host(source, destination, host)
    return destination

def _stage_rope(rope, destination_device, pool):
    destination = torch.empty_like(rope, device=destination_device)
    tokens = int(rope.shape[1])
    for start in range(0, tokens, pool.input_chunk):
        stop = min(tokens, start + pool.input_chunk)
        source = rope[:, start:stop].contiguous()
        shape = tuple((int(value) for value in source.shape))
        host = pool.raw_view(source.dtype, shape)
        _copy_via_host(source, destination[:, start:stop], host)
    return destination

def _stage_scaled_input(value, destination_device, pool):
    destination = torch.empty_like(value, dtype=torch.float16, device=destination_device)
    tokens = int(value.shape[0])
    for start in range(0, tokens, pool.input_chunk):
        stop = min(tokens, start + pool.input_chunk)
        source = value[start:stop].mul(1.0 / 16.0).half()
        _copy_via_host(source, destination[start:stop], pool.input[:stop - start])
        del source
    return destination

def _rms_norm_fp16(value, weight, epsilon):
    fp32 = value.float()
    return fp32.mul(torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + float(epsilon))).mul_(weight).half()

def _normalize_rope(q, k, rope, q_weight, k_weight, epsilon):
    from . import qk_native
    rot = int(rope.shape[-3]) * 2
    if qk_native.supports(q, k, rope, q_weight, k_weight, rot):
        return qk_native.rms_rope_split_half(q, k, rope, q_weight, k_weight, epsilon, rot, output_fp16=True)
    import comfy.quant_ops
    q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope, q_weight, k_weight, epsilon=float(epsilon), rot_dim=rot)
    return (q.half(), k.half())

def _local_qkv_chunk(attention, value, qkv_weight, q_weight, k_weight, rope, local_heads, *, already_scaled):
    device = qkv_weight.device
    torch.cuda.set_device(device)
    head_dim = int(attention.head_dim)
    source = value
    if not already_scaled:
        source = source.mul(1.0 / 16.0).half()
    projected = F.linear(source, qkv_weight)
    width = int(local_heads) * head_dim
    q, k, v = projected.split(width, dim=-1)
    count = int(value.shape[0])
    q = q.view(1, count, local_heads, head_dim).float()
    k = k.view(1, count, local_heads, head_dim).float()
    if rope is not None:
        q, k = _normalize_rope(q.contiguous(), k.contiguous(), rope.contiguous(), q_weight, k_weight, attention.q_norm.eps)
    else:
        q = _rms_norm_fp16(q, q_weight, attention.q_norm.eps)
        k = _rms_norm_fp16(k, k_weight, attention.k_norm.eps)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.view(1, count, local_heads, head_dim).transpose(1, 2).contiguous()
    del source, projected
    return (q, k, v)

def _local_qkv(attention, value, qkv_weight, q_weight, k_weight, rope, local_heads, query_chunk, *, already_scaled, outputs=None):
    device = qkv_weight.device
    torch.cuda.set_device(device)
    tokens = int(value.shape[0])
    head_dim = int(attention.head_dim)
    if outputs is None:
        q_out = torch.empty((1, local_heads, tokens, head_dim), dtype=torch.float16, device=device)
        k_out = torch.empty_like(q_out)
        v_out = torch.empty_like(q_out)
    else:
        q_out, k_out, v_out = outputs
        if any(t.shape != (1, local_heads, tokens, head_dim) or t.dtype != torch.float16 or t.device != device for t in outputs):
            raise RuntimeError('acquired QKV workspace does not match the execution plan')
    for start in range(0, tokens, query_chunk):
        stop = min(tokens, start + query_chunk)
        q, k, v = _local_qkv_chunk(attention, value[start:stop], qkv_weight, q_weight, k_weight, None if rope is None else rope[:, start:stop], local_heads, already_scaled=already_scaled)
        q_out[:, :, start:stop].copy_(q)
        k_out[:, :, start:stop].copy_(k)
        v_out[:, :, start:stop].copy_(v)
        del q, k, v
    torch.cuda.synchronize(device)
    return (q_out, k_out, v_out)

def _local_audio(qkv, audio_ranges, heads, head_dim, key_chunk):
    if not audio_ranges:
        return ()
    from .h3_mixed_precision import _streaming_audio_attention
    q, k, v = qkv
    return _streaming_audio_attention(q, k, v, audio_ranges, heads, head_dim, key_chunk=key_chunk, transformer_options=None)


def _release_exact_qkv_inputs(values):
    """Drop preparation-only references after both synchronized QKV workers.

    Q/K/V, FP32 audio and output weights belong to the next phase. Do not flush
    allocators or lower admission reserves based on reference release.
    """
    for key in ('weights', 'norms', 'x1', 'rope1'):
        values[key] = None

def _exact_audio_skip_ranges(audio_outputs, tokens):
    """Skip only complete query blocks covered by existing FP32 audio rows."""
    from .sol_calibration import _full_audio_query_block_ranges
    return tuple(((first * 64, min(last * 64, int(tokens))) for first, last in _full_audio_query_block_ranges([(start, stop) for start, stop, _out in audio_outputs or ()], tokens)))

def _local_range(ops, qkv, out_weight, start, stop, tokens, scale, audio_outputs, *, audio_skip_ranges=None):
    from .h3_mixed_precision import _replace_streamed_audio_rows
    q, k, v = qkv
    if audio_skip_ranges is None:
        audio_skip_ranges = _exact_audio_skip_ranges(audio_outputs, tokens)
    skipped = [(max(int(start), first), min(int(stop), last)) for first, last in audio_skip_ranges if first < stop and last > start]
    if not skipped:
        out, lse = ops.sol_dense_rect(q, k, v, int(start), int(stop - start), int(tokens), float(scale))
        del lse
        flat = out.permute(0, 2, 1, 3).reshape(stop - start, -1).float()
    else:
        flat = torch.empty((stop - start, q.shape[1] * q.shape[3]), dtype=torch.float32, device=q.device)
        cursor = int(start)
        for first, last in (*skipped, (int(stop), int(stop))):
            if cursor < first:
                out, lse = ops.sol_dense_rect(q, k, v, cursor, first - cursor, int(tokens), float(scale))
                flat[cursor - start:first - start].copy_(out.permute(0, 2, 1, 3).reshape(first - cursor, -1))
                del out, lse
            cursor = last
    _replace_streamed_audio_rows(flat, int(start), audio_outputs)
    return F.linear(flat, out_weight)

def _local_sol_stream(qkv, sol_config):
    """Build one head-local corrected Sol stream on its owning device."""
    from .sol_range import run_fused_sol_active
    q, k, v = qkv
    torch.cuda.set_device(q.device)
    return run_fused_sol_active(q, k, v, tau=float(sol_config['tau']), prefix_stop=int(sol_config['prefix_stop']), scale=float(sol_config['scale']), stream_output=True, stream_chunk_tokens=int(sol_config['stream_chunk_tokens']), memory_limit_mib=int(sol_config['memory_limit_mib']), audio_ranges=tuple(sol_config['audio_ranges']), audio_overwrite_active=bool(sol_config['audio_overwrite_active']), route_builder=None)

def _local_sol_next(stream_iterator, out_weight, audio_outputs):
    """Consume and project one corrected Sol range on the local device."""
    from .h3_mixed_precision import _replace_streamed_audio_rows
    torch.cuda.set_device(out_weight.device)
    start, out, lse = next(stream_iterator)
    count = int(out.shape[2])
    flat = out.permute(0, 2, 1, 3).reshape(count, -1).float()
    del out, lse
    _replace_streamed_audio_rows(flat, int(start), audio_outputs)
    return (int(start), F.linear(flat, out_weight))

def _inference_worker(function, *args, **kwargs):
    """Restore PyTorch inference mode inside executor threads."""
    with torch.inference_mode():
        return function(*args, **kwargs)


@dataclass
class DualExactAttentionState:
    secondary_index: int | None = None
    performance_floor_tokens: int = 16384
    query_chunk: int = 2048
    weight_profile: str = FP8_E4M3_PROFILE
    _pool: _PinnedPool | None = field(default=None, init=False, repr=False)
    _executor: concurrent.futures.ThreadPoolExecutor = field(default_factory=lambda: concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='h3-dual-v100'), init=False, repr=False)
    _graph_executor: concurrent.futures.ThreadPoolExecutor = field(default_factory=lambda: concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='h3-dual-graph-owner'), init=False, repr=False)
    _lock: threading.Lock = field(default_factory=lambda: _DUAL_RUN_LOCK, init=False, repr=False)
    _disabled_signatures: set = field(default_factory=set, init=False, repr=False)
    _transient_local: threading.local = field(default_factory=threading.local, init=False, repr=False)
    _validated_secondaries: dict[tuple[int, int | None], tuple[int, ...]] = field(default_factory=dict, init=False, repr=False)

    def begin_sample(self):
        self._transient_local.failures = set()
        self._transient_local.resource_reprobes = {}
        self._transient_local.admission_warnings = set()
        self._transient_local.cache_recovery_backoff = {}
        self._transient_local.wddm_rebalance_attempted = set()
        self._transient_local.wddm_balanced_leases = {}

    def _warn_admission_fallback(self, tokens, mode, reason):
        warned = getattr(self._transient_local, 'admission_warnings', None)
        if warned is None:
            warned = self._transient_local.admission_warnings = set()
        key = (int(tokens), mode)
        if key not in warned:
            warned.add(key)
            LOGGER.warning('H3 dual attention using single-device fallback: '
                           'tokens=%d mode=%s reason=%s. Further identical warnings '
                           'are suppressed for this sample.', int(tokens), mode, reason)

    def _base_bypass_reason(self, x, transformer_options):
        from .sol_attention import MODE_FLASH, MODE_SOL
        if x.device.type != 'cuda':
            return 'not-cuda'
        if x.device.index is None:
            return 'missing-device-index'
        if x.dtype != torch.float32:
            return 'non-fp32-input'
        if torch.is_grad_enabled():
            return 'grad-enabled'
        if int(x.shape[0]) < int(self.performance_floor_tokens):
            return 'below-performance-floor'
        if transformer_options.get(SELECTED_BACKEND_KEY) not in (MODE_FLASH, MODE_SOL):
            return 'backend-not-supported'
        return None

    def _transient_failures(self):
        if not hasattr(self._transient_local, 'failures'):
            self._transient_local.failures = set()
        return self._transient_local.failures

    def _failure_key(self, primary, secondary, tokens, hidden, sol_route):
        return (int(primary), int(secondary), int(tokens), int(hidden), 'sol_sparse' if sol_route else 'exact')

    def _quarantine(self, key, error, *, preflight=False):
        transient = _resource_failure(error) or (preflight and _recoverable_probe_failure(error))
        target = self._transient_failures() if transient else self._disabled_signatures
        target.add(key)
        retries = getattr(self._transient_local, 'resource_reprobes', {})
        if key in retries:
            retries[key] = (retries[key][0], None)  # Invalidate older recovery.
        if not preflight and _resource_failure(error):
            if not hasattr(self._transient_local, 'resource_reprobes'):
                self._transient_local.resource_reprobes = {}
            retries = self._transient_local.resource_reprobes
            count = retries.get(key, (0, None))[0] + 1
            ticket = object()
            retries[key] = (count, ticket)
            if count <= RESOURCE_REPROBES_PER_SAMPLE:
                # Captures metadata only. No CUDA traceback/target survives here.
                return lambda: self._allow_resource_reprobe(key, ticket)
        return None

    def _allow_resource_reprobe(self, key, ticket):
        """A completed fallback block permits fresh admission, never a bypass."""
        retries = getattr(self._transient_local, 'resource_reprobes', {})
        count, current = retries.get(key, (0, None))
        if (current is not ticket or key in self._disabled_signatures
                or key not in self._transient_failures()):
            return False
        retries[key] = (count, None)  # Consume once; keep the sample-wide budget.
        self._transient_failures().discard(key)
        LOGGER.info('H3 dual attention resource recovery: single-device block '
                    'completed; fresh dual admission allowed (%d/%d).',
                    count, RESOURCE_REPROBES_PER_SAMPLE)
        return True

    def _host_snapshot(self):
        live = system_host_memory()
        owned = 0 if self._pool is None else self._pool.allocated_bytes
        return HostMemory(live.total_bytes, min(live.total_bytes, live.available_bytes + owned))

    def _runtime(self, primary, secondary, snapshots, host):
        remaining = dict(snapshots)
        initial_host = [host]

        def gpu_snapshot(index):
            cached = remaining.pop(int(index), None)
            return cached if cached is not None else torch_device_memory(index)

        def host_snapshot():
            return initial_host.pop() if initial_host else self._host_snapshot()
        return DualAdaptiveAdmission(primary, secondary, performance_floor_tokens=self.performance_floor_tokens, gpu_snapshot=gpu_snapshot, host_snapshot=host_snapshot)

    def _select_admission(self, primary_index, tokens, hidden, **requirements):
        """Rank complete feasible plans, sharing one initial GPU/RAM snapshot."""
        primary_index = int(primary_index)
        primary_memory = torch_device_memory(primary_index)
        probe_errors = []
        cache_key = (primary_index, self.secondary_index)
        eligible_indices = self._validated_secondaries.get(cache_key)
        if eligible_indices is None:
            eligible = []
            requested = range(torch.cuda.device_count()) if self.secondary_index is None else (int(self.secondary_index),)
            for index in requested:
                if index == primary_index:
                    continue
                try:
                    accepted, _receipt = validate_dual_v100_pair(primary_index, index)
                except Exception as error:
                    if not _recoverable_probe_failure(error):
                        raise
                    probe_errors.append(f'cuda:{index}: {type(error).__name__}')
                    continue
                if accepted:
                    eligible.append(int(index))
            eligible_indices = tuple(eligible)
            if not probe_errors:
                self._validated_secondaries[cache_key] = eligible_indices
        candidates = []
        blocked = False
        host = None
        for index in eligible_indices:
            failure_key = self._failure_key(primary_index, index, tokens, hidden, requirements.get('sol_route', False))
            if failure_key in self._disabled_signatures or failure_key in self._transient_failures():
                blocked = True
                continue
            try:
                secondary_memory = torch_device_memory(index)
            except Exception as error:
                if not _recoverable_probe_failure(error):
                    raise
                probe_errors.append(f'cuda:{index}: {type(error).__name__}')
                continue
            if host is None:
                host = self._host_snapshot()
            runtime = self._runtime(primary_index, index, {primary_index: primary_memory, index: secondary_memory}, host)
            admission = runtime.prepare(tokens, **requirements)
            plan = admission.attention
            rank = (not admission.enabled, max(0, -plan.primary_margin_bytes) + max(0, -plan.secondary_margin_bytes), max(plan.primary_units, plan.secondary_units), -min(plan.primary_margin_bytes / max(1, primary_memory.total_bytes), plan.secondary_margin_bytes / max(1, secondary_memory.total_bytes)), index)
            candidates.append((rank, runtime, admission))
        if candidates:
            _rank, runtime, admission = min(candidates, key=lambda item: item[0])
            return (runtime, admission, None)
        if probe_errors:
            raise _ProbeUnavailable('; '.join(probe_errors))
        return (None, None, 'signature-disabled-from-prior-failure' if blocked else 'no-secondary-v100')

    def _cache_recovery_backoff(self):
        if not hasattr(self._transient_local, 'cache_recovery_backoff'):
            self._transient_local.cache_recovery_backoff = {}
        return self._transient_local.cache_recovery_backoff

    @staticmethod
    def _cache_recovery_key(admission, tokens, requirements):
        return (admission.primary.index, admission.secondary.index, int(tokens), bool(requirements.get('sol_route', False)))

    def _recover_primary_admission_cache(self, admission, options, tokens, requirements):
        """Try shared allocator reclaim only to rescue a rejected dual plan.

        Cached bytes are a feasibility hint, never an admission. No VBAR
        eviction, extra synchronization, or retries to improve an admitted
        head split. Both GPUs and host staging must fit before trying a trim.
        """
        from .runtime_memory import POLICY_KEY, H3RuntimeMemoryPolicy, _runtime_state, memory_snapshot, request_cuda_headroom
        if not isinstance(options.get(POLICY_KEY), H3RuntimeMemoryPolicy) or admission.enabled or admission.reason != 'no-two-device-capacity':
            return False
        plan = admission.attention
        if plan.primary_margin_bytes >= 0 or plan.secondary_margin_bytes < 0:
            return False
        primary, secondary = (admission.primary, admission.secondary)
        device = torch.device('cuda', primary.index)
        snapshot = memory_snapshot(device)
        target = int(plan.primary_required_bytes + plan.primary_reserve_bytes)
        deficit = max(0, target - int(snapshot['free_bytes']))
        if not deficit:
            return True
        reclaimable = int(snapshot['reclaimable_bytes'])
        if reclaimable < deficit:
            return False
        from dataclasses import replace
        potential_primary = replace(primary, free_bytes=min(int(snapshot['total_bytes']), int(snapshot['free_bytes']) + reclaimable))
        potential = self._runtime(primary.index, secondary.index, {primary.index: potential_primary, secondary.index: secondary}, admission.host).prepare(tokens, **requirements)
        if not potential.enabled:
            return False
        memory_state = _runtime_state(options)
        backoff = self._cache_recovery_backoff()
        key = self._cache_recovery_key(admission, tokens, requirements)
        remaining = backoff.get(key, 0)
        if remaining > 0:
            backoff[key] = remaining - 1
            return False
        backoff[key] = max(0, int(options[POLICY_KEY].cooldown_checks))
        result = request_cuda_headroom(device, options, reason='dual-admission-cache', snapshot=snapshot, required_free_bytes=target, minimum_reclaimable_mib=0, demand_bypass_cooldown=False, honor_cooldown=False, allow_vbar_release=False)
        if not result.get('performed'):
            return False
        return True

    def _wddm_rebalance_attempted(self):
        if not hasattr(self._transient_local, 'wddm_rebalance_attempted'):
            self._transient_local.wddm_rebalance_attempted = set()
        return self._transient_local.wddm_rebalance_attempted

    def _balanced_lease_key(self, primary, tokens, hidden, requirements):
        return (
            int(primary), int(tokens), int(hidden), self.weight_profile,
            int(requirements.get('query_chunk', self.query_chunk)),
            int(requirements.get('audio_rows', 0)),
            int(requirements.get('audio_key_chunk', 0)),
            int(requirements.get('attention_decode_extra_bytes', 0)),
        )

    def _balanced_leases(self):
        if not hasattr(self._transient_local, 'wddm_balanced_leases'):
            self._transient_local.wddm_balanced_leases = {}
        return self._transient_local.wddm_balanced_leases

    def _remember_balanced_lease(self, selected, primary, tokens, hidden, requirements):
        """Lease only a balanced exact plan that completed successfully."""
        runtime, admission, _reason = selected
        if runtime is None or admission is None or not admission.enabled:
            return
        plan = admission.attention
        if (
            bool(requirements.get('sol_route'))
            or int(tokens) < WDDM_REBALANCE_MIN_TOKENS
            or int(plan.primary_units) != int(plan.secondary_units)
        ):
            return
        key = self._balanced_lease_key(primary, tokens, hidden, requirements)
        self._balanced_leases()[key] = selected

    def _drop_balanced_lease(self, primary, tokens, hidden, requirements):
        key = self._balanced_lease_key(primary, tokens, hidden, requirements)
        self._balanced_leases().pop(key, None)

    def _validated_balanced_lease(
        self, primary, tokens, hidden, head_dim, rope, total_heads, requirements
    ):
        """Reuse a proven same-shape plan while allocator-owned capacity remains.

        Driver-free memory alone drops after PyTorch caches recurring QKV/range
        workspaces.  A completed plan proves those shapes can execute.  The
        lease still checks current driver-free plus reclaimable allocator bytes,
        the secondary's live capacity, the pinned pool, and a hard free floor.
        """
        if bool(requirements.get('sol_route')) or int(tokens) < WDDM_REBALANCE_MIN_TOKENS:
            return None
        key = self._balanced_lease_key(primary, tokens, hidden, requirements)
        selected = self._balanced_leases().get(key)
        if selected is None:
            return None
        runtime, admission, _reason = selected
        plan = admission.attention
        if not _pool_matches_admission(
            self._pool, admission, hidden, head_dim, self.query_chunk, rope,
            sol_route=False, total_heads=total_heads,
        ):
            self._balanced_leases().pop(key, None)
            return None
        try:
            from .runtime_memory import memory_snapshot
            primary_snapshot = memory_snapshot(torch.device('cuda', int(primary)))
            secondary_memory = torch_device_memory(runtime.secondary_index)
            host = self._host_snapshot()
        except Exception as error:
            if not _recoverable_probe_failure(error):
                raise
            self._balanced_leases().pop(key, None)
            return None
        primary_effective = int(primary_snapshot['free_bytes']) + int(primary_snapshot['reclaimable_bytes'])
        primary_needed = int(plan.primary_required_bytes) + int(plan.primary_reserve_bytes)
        secondary_needed = int(plan.secondary_required_bytes) + int(plan.secondary_reserve_bytes)
        driver_floor = max(
            WDDM_LEASE_MIN_DRIVER_FREE_MIB * 1024 ** 2,
            int(primary_snapshot['total_bytes']) * 3 // 100,
        )
        host_reserve = max(2 * 1024 ** 3, int(host.total_bytes * 0.05))
        host_usable = max(0, int(host.available_bytes) - host_reserve)
        host_limit = min(int(host.total_bytes * 0.08), int(host_usable * 0.2))
        valid = (
            int(primary_snapshot['free_bytes']) >= driver_floor
            and primary_effective >= primary_needed
            and int(secondary_memory.free_bytes) >= secondary_needed
            and host_usable >= int(admission.staging.required_bytes)
            and int(admission.staging.required_bytes) <= host_limit
        )
        if not valid:
            self._balanced_leases().pop(key, None)
            return None
        return selected

    def _rebalance_admitted_exact_cache(self, selected, options, tokens, hidden, requirements):
        """Reclaim inactive primary cache once when it materially improves exact balance.

        This is a WDDM performance optimization, not a capacity escape hatch.
        The existing admitted plan remains valid throughout.  It never releases
        DynamicVRAM pages, changes reserves, or retries repeatedly within one
        sampler segment.
        """
        runtime, admission, _reason = selected
        if runtime is None or admission is None or not admission.enabled:
            return selected
        if bool(requirements.get('sol_route')) or int(tokens) < WDDM_REBALANCE_MIN_TOKENS:
            return selected
        plan = admission.attention
        if plan.primary_units >= plan.secondary_units:
            return selected
        current_bottleneck = max(int(plan.primary_units), int(plan.secondary_units))
        key = self._cache_recovery_key(admission, tokens, requirements)
        attempted = self._wddm_rebalance_attempted()
        if key in attempted:
            return selected
        from .runtime_memory import POLICY_KEY, H3RuntimeMemoryPolicy, memory_snapshot, request_cuda_headroom
        policy = options.get(POLICY_KEY)
        if not isinstance(policy, H3RuntimeMemoryPolicy):
            return selected
        primary, secondary = admission.primary, admission.secondary
        device = torch.device('cuda', primary.index)
        snapshot = memory_snapshot(device)
        reclaimable = int(snapshot['reclaimable_bytes'])
        minimum_reclaimable = WDDM_REBALANCE_MIN_RECLAIMABLE_MIB * 1024 ** 2
        if reclaimable < minimum_reclaimable:
            return selected
        from dataclasses import replace
        potential_primary = replace(
            primary,
            free_bytes=min(
                int(snapshot['total_bytes']),
                int(snapshot['free_bytes']) + reclaimable,
            ),
        )
        potential = self._runtime(
            primary.index,
            secondary.index,
            {primary.index: potential_primary, secondary.index: secondary},
            admission.host,
        ).prepare(tokens, **requirements)
        if not potential.enabled:
            return selected
        potential_plan = potential.attention
        potential_bottleneck = max(
            int(potential_plan.primary_units), int(potential_plan.secondary_units)
        )
        head_gain = current_bottleneck - potential_bottleneck
        if head_gain < WDDM_REBALANCE_MIN_HEAD_GAIN:
            return selected
        attempted.add(key)
        target = int(potential_plan.primary_required_bytes + potential_plan.primary_reserve_bytes)
        result = request_cuda_headroom(
            device,
            options,
            reason='wddm-dual-balance-cache',
            snapshot=snapshot,
            required_free_bytes=target,
            minimum_reclaimable_mib=WDDM_REBALANCE_MIN_RECLAIMABLE_MIB,
            demand_bypass_cooldown=True,
            honor_cooldown=True,
            allow_vbar_release=False,
        )
        if not result.get('performed'):
            return selected
        refreshed = self._select_admission(primary.index, tokens, hidden, **requirements)
        refreshed_admission = refreshed[1]
        if refreshed_admission is None or not refreshed_admission.enabled:
            return selected
        refreshed_plan = refreshed_admission.attention
        refreshed_bottleneck = max(
            int(refreshed_plan.primary_units), int(refreshed_plan.secondary_units)
        )
        if refreshed_bottleneck >= current_bottleneck:
            return selected
        return refreshed

    def _coordinated_admission(self, primary, tokens, hidden, options, **requirements):
        selected = self._select_admission(primary, tokens, hidden, **requirements)
        runtime, admission, _reason = selected
        if admission is not None and admission.enabled:
            selected = self._rebalance_admitted_exact_cache(
                selected, options, tokens, hidden, requirements
            )
            admission = selected[1]
            backoff = self._cache_recovery_backoff()
            if backoff:
                backoff.pop(self._cache_recovery_key(admission, tokens, requirements), None)
            return selected
        if runtime is None or not self._recover_primary_admission_cache(admission, options, tokens, requirements):
            return selected
        selected = self._select_admission(primary, tokens, hidden, **requirements)
        key = 'readmitted' if selected[1] is not None and selected[1].enabled else 'still_rejected'
        if key == 'readmitted':
            backoff = self._cache_recovery_backoff()
            for recovered in (admission, selected[1]):
                backoff.pop(self._cache_recovery_key(recovered, tokens, requirements), None)
        return selected

    def _ensure_pool(self, admission, hidden, head_dim, rope=None, *, sol_route=False, total_heads=56):
        if _pool_matches_admission(self._pool, admission, hidden, head_dim, self.query_chunk, rope, sol_route=sol_route, total_heads=total_heads):
            return False
        required = max(_prepared_weight_bytes(int(total_heads) if sol_route else admission.attention.secondary_units, hidden, head_dim), _rope_chunk_bytes(rope, admission.staging.input_chunk_tokens))
        self._pool = None
        self._pool = _PinnedPool(admission.staging.input_chunk_tokens, self.query_chunk, hidden, required)
        return True

    def release_host_pool(self):
        """Release idle pinned staging RAM without invalidating this state."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self._pool = None
            return True
        finally:
            self._lock.release()

    def _prepare_weights(self, attention, transformer_options, primary, secondary, primary_heads, pool):
        from .h3_mixed_precision import _projection_weight_context
        import comfy.model_management as model_management
        heads = int(attention.heads)
        head_dim = int(attention.head_dim)
        ranges = ((0, primary_heads), (primary_heads, heads))
        with _projection_weight_context(attention.qkv_proj, primary, torch.float16, transformer_options) as prepared:
            weight, bias = prepared
            if bias is not None:
                raise RuntimeError('dual H3 attention requires bias-free QKV')
            qkv0 = qkv_head_rows(weight, *ranges[0], heads, head_dim)
            source1 = qkv_head_rows(weight, *ranges[1], heads, head_dim)
            qkv1 = _stage_tensor(source1, secondary, pool)
            del source1
        del prepared, weight, bias
        with _projection_weight_context(attention.out_proj, primary, torch.float32, transformer_options) as prepared:
            weight, bias = prepared
            if bias is not None:
                raise RuntimeError('dual H3 attention requires bias-free output projection')
            out0 = out_head_columns(weight, *ranges[0], heads, head_dim)
            source1 = out_head_columns(weight, *ranges[1], heads, head_dim)
            out1 = _stage_tensor(source1, secondary, pool)
            del source1
        del prepared, weight, bias
        q_weight0 = model_management.cast_to(attention.q_norm.weight, dtype=torch.float32, device=primary).contiguous()
        k_weight0 = model_management.cast_to(attention.k_norm.weight, dtype=torch.float32, device=primary).contiguous()
        q_weight1 = _stage_tensor(q_weight0, secondary, pool)
        k_weight1 = _stage_tensor(k_weight0, secondary, pool)
        return ((qkv0, qkv1), (out0, out1), ((q_weight0, k_weight0), (q_weight1, k_weight1)))

    def _prepare_sol_qkv_and_output(self, attention, target, rope_freqs, transformer_options, primary, secondary, primary_heads, key_chunk, pool):
        """Replicate full-width QKV projection and retain one local head shard.

        Sol routing is sensitive to the full-width GEMM shape.  Projecting the
        same token chunk on both V100s keeps that shape and lets each device
        retain only its assigned heads.  This deliberately duplicates QKV
        compute to avoid sending three full-sequence Q/K/V head complements
        through host RAM after every chunk.
        """
        from .h3_mixed_precision import _projection_weight_context
        import comfy.model_management as model_management
        heads = int(attention.heads)
        head_dim = int(attention.head_dim)
        ranges = ((0, primary_heads), (primary_heads, heads))
        with _projection_weight_context(attention.out_proj, primary, torch.float32, transformer_options) as prepared:
            weight, bias = prepared
            if bias is not None:
                raise RuntimeError('dual H3 attention requires bias-free output projection')
            out0 = out_head_columns(weight, *ranges[0], heads, head_dim)
            source1 = out_head_columns(weight, *ranges[1], heads, head_dim)
            out1 = _stage_tensor(source1, secondary, pool)
            del source1
        del prepared, weight, bias
        q_weight = model_management.cast_to(attention.q_norm.weight, dtype=torch.float32, device=primary).contiguous()
        k_weight = model_management.cast_to(attention.k_norm.weight, dtype=torch.float32, device=primary).contiguous()
        q_weight1 = _stage_tensor(q_weight, secondary, pool)
        k_weight1 = _stage_tensor(k_weight, secondary, pool)
        rope1 = None if rope_freqs is None else _stage_rope(rope_freqs, secondary, pool)
        with ExitStack() as weight_owner:
            prepared = None
            if self.weight_profile == INT8_CONVROT_PROFILE:
                prepared = weight_owner.enter_context(_projection_weight_context(attention.qkv_proj, primary, torch.float16, transformer_options))
            tokens = int(target.shape[0])
            secondary_heads = heads - int(primary_heads)
            primary_qkv = tuple((torch.empty((1, primary_heads, tokens, head_dim), dtype=torch.float16, device=primary) for _ in range(3)))
            secondary_qkv = tuple((torch.empty((1, secondary_heads, tokens, head_dim), dtype=torch.float16, device=secondary) for _ in range(3)))
            secondary_input = torch.empty((min(tokens, key_chunk), int(target.shape[1])), dtype=torch.float16, device=secondary)
            if prepared is None:
                prepared = weight_owner.enter_context(_projection_weight_context(attention.qkv_proj, primary, torch.float16, transformer_options))
            qkv_weight, bias = prepared
            if bias is not None:
                raise RuntimeError('dual H3 attention requires bias-free QKV')
            qkv_weight1 = _stage_tensor(qkv_weight, secondary, pool)
            for start in range(0, tokens, key_chunk):
                stop = min(tokens, start + key_chunk)
                count = stop - start
                scaled = target[start:stop].mul(1.0 / 16.0).half()
                input1 = secondary_input[:count]
                _copy_via_host(scaled, input1, pool.input[:count])
                future1 = self._executor.submit(_inference_worker, _local_qkv_chunk, attention, input1, qkv_weight1, q_weight1, k_weight1, None if rope1 is None else rope1[:, start:stop], heads, already_scaled=True)
                chunk0 = chunk1 = None
                try:
                    chunk0 = _local_qkv_chunk(attention, scaled, qkv_weight, q_weight, k_weight, None if rope_freqs is None else rope_freqs[:, start:stop], heads, already_scaled=True)
                    chunk1 = future1.result()
                except BaseException:
                    try:
                        future1.result()
                    except Exception:
                        pass
                    raise
                for index, tensor in enumerate(chunk0):
                    primary_qkv[index][:, :, start:stop].copy_(tensor[:, :primary_heads])
                for index, tensor in enumerate(chunk1):
                    secondary_qkv[index][:, :, start:stop].copy_(tensor[:, primary_heads:])
                del chunk0, chunk1, input1, scaled, future1
        return ((primary_qkv, secondary_qkv), (out0, out1))

    def run(self, attention, target, rope_freqs, transformer_options, fallback, *, sol_config=None):
        dual_mode = 'sol_sparse' if isinstance(sol_config, dict) else 'exact'
        primary_index = int(target.device.index)
        signature = (primary_index, int(target.shape[0]), int(target.shape[1]))
        from .h3_mixed_precision import AUDIO_RANGES_OPTION_KEY
        audio_ranges = tuple(transformer_options.get(AUDIO_RANGES_OPTION_KEY, ()))
        audio_rows = sum((max(0, int(stop) - int(start)) for start, stop in audio_ranges))
        key_chunk = max(1, int(transformer_options.get('v100_h3_qkv_chunk_tokens', 1024)))
        requirements = dict(query_chunk=self.query_chunk, active_phases=('attention',), sol_route=dual_mode == 'sol_sparse', audio_rows=audio_rows, audio_key_chunk=key_chunk, host_attention_slots=1, attention_minimum_heads=8 if dual_mode == 'sol_sparse' else 19)
        hidden = int(target.shape[1])
        head_dim = int(attention.head_dim)
        if not self._lock.acquire(blocking=False):
            return fallback(target, rope_freqs=rope_freqs, transformer_options=transformer_options)
        runtime = admission = None
        selected = None
        rejection_reason = None
        from .graph_capacity import active_primary_graph, acquire as acquire_graph_workspace
        graph_context = None
        acquired_qkv = None
        try:
            graph_context = active_primary_graph(target.device)
            if self.weight_profile == INT8_CONVROT_PROFILE:
                requirements['attention_decode_extra_bytes'] = dual_attention_decode_extra_bytes(self.weight_profile, heads=int(attention.heads), head_dim=int(attention.head_dim), hidden=int(target.shape[1]))
            selected = None if graph_context is not None else self._validated_balanced_lease(
                primary_index, int(target.shape[0]), hidden, head_dim,
                rope_freqs, int(attention.heads), requirements,
            )
            if graph_context is not None:
                selected, acquired_qkv, requirements = acquire_graph_workspace(
                    self, primary_index, int(target.shape[0]), hidden, head_dim,
                    transformer_options, requirements,
                )
            elif selected is None:
                selected = self._coordinated_admission(
                    primary_index, int(target.shape[0]), hidden,
                    transformer_options, **requirements,
                )
            runtime, admission, rejection_reason = selected
        except Exception as error:
            if not isinstance(error, _ProbeUnavailable) and (not _recoverable_probe_failure(error)):
                raise
            rejection_reason = 'device-probe-unavailable'
        finally:
            if runtime is None:
                self._lock.release()
        if runtime is None:
            return fallback(target, rope_freqs=rope_freqs, transformer_options=transformer_options)
        secondary_index = runtime.secondary_index
        failure_key = self._failure_key(primary_index, secondary_index, target.shape[0], target.shape[1], dual_mode == 'sol_sparse')
        transaction = OperationTransaction(f'dual-{dual_mode}-attention')
        transaction.start()
        failure = None
        receipt = None
        on_recovery = None
        admission_rejected = False
        qkv_futures = None
        audio_futures = None
        secondary_future = None
        primary_part = None
        cuda_started = False
        graph_owned = False
        propagating_error = False
        primary = torch.device('cuda', primary_index)
        secondary = torch.device('cuda', secondary_index)
        values = {'weights': None, 'out_weights': None, 'norms': None, 'x1': None, 'rope1': None, 'qkv': None, 'audio': None, 'returned': None, 'sol_streams': None, 'sol_iterators': None, 'pending': []}
        values['primary_qkv'] = acquired_qkv
        acquired_qkv = None
        try:
            for _attempt in range(2):
                if not admission.enabled or _pool_matches_admission(self._pool, admission, hidden, head_dim, self.query_chunk, rope_freqs, sol_route=dual_mode == 'sol_sparse', total_heads=int(attention.heads)):
                    break
                self._ensure_pool(admission, hidden, head_dim, rope_freqs, sol_route=dual_mode == 'sol_sparse', total_heads=int(attention.heads))
                admission = runtime.prepare(int(target.shape[0]), **requirements)
            if not admission.enabled or not _pool_matches_admission(self._pool, admission, hidden, head_dim, self.query_chunk, rope_freqs, sol_route=dual_mode == 'sol_sparse', total_heads=int(attention.heads)):
                raise _DualAdmissionFallback
            primary_heads = int(admission.attention.primary_units)
            secondary_heads = int(admission.attention.secondary_units)
            tokens = int(target.shape[0])
            pool = self._pool
            cuda_started = True
            values['returned'] = torch.empty((self.query_chunk, int(target.shape[1])), dtype=torch.float32, device=primary)
            if graph_context is not None:
                from .graph_attention import run as run_graph_owned
                graph_owned = True
                run_graph_owned(
                    self, attention, target, rope_freqs, transformer_options,
                    primary, secondary, primary_heads, secondary_heads,
                    key_chunk, pool, audio_ranges, values, transaction,
                    sol_config=sol_config,
                )
                transaction.commit()
                return target
            if dual_mode == 'sol_sparse':
                values['qkv'], values['out_weights'] = self._prepare_sol_qkv_and_output(attention, target, rope_freqs, transformer_options, primary, secondary, primary_heads, key_chunk, pool)
            else:
                values['weights'], values['out_weights'], values['norms'] = self._prepare_weights(attention, transformer_options, primary, secondary, primary_heads, pool)
                values['x1'] = _stage_scaled_input(target, secondary, pool)
                if rope_freqs is not None:
                    values['rope1'] = _stage_rope(rope_freqs, secondary, pool)
                qkv_futures = (self._executor.submit(_inference_worker, _local_qkv, attention, target, values['weights'][0], *values['norms'][0], rope_freqs, primary_heads, key_chunk, already_scaled=False), self._executor.submit(_inference_worker, _local_qkv, attention, values['x1'], values['weights'][1], *values['norms'][1], values['rope1'], secondary_heads, key_chunk, already_scaled=True))
                values['pending'] = list(qkv_futures)
                values['qkv'] = tuple(future.result() for future in qkv_futures)
                values['pending'].clear()
                qkv_futures = None
                _release_exact_qkv_inputs(values)
            audio_futures = (self._executor.submit(_inference_worker, _local_audio, values['qkv'][0], audio_ranges, primary_heads, int(attention.head_dim), key_chunk), self._executor.submit(_inference_worker, _local_audio, values['qkv'][1], audio_ranges, secondary_heads, int(attention.head_dim), key_chunk))
            values['pending'] = list(audio_futures)
            values['audio'] = tuple(future.result() for future in audio_futures)
            values['pending'].clear()
            audio_futures = None
            scale = int(attention.head_dim) ** (-0.5)
            if dual_mode == 'sol_sparse':
                effective_config = dict(sol_config)
                effective_config['scale'] = scale
                stream_futures = (self._executor.submit(_inference_worker, _local_sol_stream, values['qkv'][0], effective_config), self._executor.submit(_inference_worker, _local_sol_stream, values['qkv'][1], effective_config))
                values['pending'] = list(stream_futures)
                stream_results = tuple((future.result() for future in stream_futures))
                values['pending'].clear()
                values['sol_streams'] = tuple((result[0] for result in stream_results))
                values['sol_iterators'] = tuple((iter(stream) for stream in values['sol_streams']))
                cursor = 0
                while cursor < tokens:

                    def secondary_sol_range():
                        start1, part1 = _local_sol_next(values['sol_iterators'][1], values['out_weights'][1], values['audio'][1])
                        host = pool.partial[:int(part1.shape[0])]
                        host.copy_(part1, non_blocking=True)
                        torch.cuda.synchronize(secondary)
                        return (start1, int(part1.shape[0]))
                    secondary_future = self._executor.submit(_inference_worker, secondary_sol_range)
                    values['pending'] = [secondary_future]
                    start0, primary_part = _local_sol_next(values['sol_iterators'][0], values['out_weights'][0], values['audio'][0])
                    start1, count1 = secondary_future.result()
                    values['pending'].clear()
                    secondary_future = None
                    count0 = int(primary_part.shape[0])
                    if start0 != cursor or start1 != cursor or count1 != count0:
                        raise RuntimeError('dual Sol head streams returned different query ranges')
                    stop = cursor + count0
                    if stop > tokens or count0 > int(pool.partial.shape[0]):
                        raise RuntimeError('dual Sol query range exceeds admitted pool')
                    returned = values['returned'][:count0]
                    returned.copy_(pool.partial[:count0], non_blocking=True)
                    torch.cuda.synchronize(primary)
                    primary_part.add_(returned).mul_(16.0)
                    target[cursor:stop].copy_(primary_part)
                    transaction.mark_partial_target_write()
                    del primary_part
                    primary_part = None
                    cursor = stop
            else:
                from .sol_native import load_sol_ops
                ops = load_sol_ops(require_dense=True)
                audio_skip_ranges = _exact_audio_skip_ranges(values['audio'][0], tokens)
                for start in range(0, tokens, self.query_chunk):
                    stop = min(tokens, start + self.query_chunk)

                    def secondary_range():
                        torch.cuda.set_device(secondary)
                        part = _local_range(ops, values['qkv'][1], values['out_weights'][1], start, stop, tokens, scale, values['audio'][1], audio_skip_ranges=audio_skip_ranges)
                        host = pool.partial[:stop - start]
                        host.copy_(part, non_blocking=True)
                        torch.cuda.synchronize(secondary)
                        return None
                    secondary_future = self._executor.submit(_inference_worker, secondary_range)
                    values['pending'] = [secondary_future]
                    torch.cuda.set_device(primary)
                    primary_part = _local_range(ops, values['qkv'][0], values['out_weights'][0], start, stop, tokens, scale, values['audio'][0], audio_skip_ranges=audio_skip_ranges)
                    secondary_future.result()
                    values['pending'].clear()
                    secondary_future = None
                    returned = values['returned'][:stop - start]
                    returned.copy_(pool.partial[:stop - start], non_blocking=True)
                    torch.cuda.synchronize(primary)
                    primary_part.add_(returned).mul_(16.0)
                    target[start:stop].copy_(primary_part)
                    transaction.mark_partial_target_write()
                    del primary_part
                    primary_part = None
            transaction.commit()
            try:
                self._remember_balanced_lease(
                    (runtime, admission, rejection_reason),
                    primary_index, tokens, hidden, requirements,
                )
            except Exception:
                LOGGER.debug('H3 WDDM balanced lease update failed.', exc_info=True)
            return target
        except _DualAdmissionFallback:
            admission_rejected = True
            self._warn_admission_fallback(int(target.shape[0]), dual_mode,
                admission.reason if not admission.enabled else 'host-pool-unavailable')
            self._drop_balanced_lease(
                primary_index, int(target.shape[0]), hidden, requirements
            )
            transaction.abort()
        except Exception as error:
            failure = (type(error).__name__, 'dual attention execution failed')
            receipt = transaction.abort()
            self._drop_balanced_lease(
                primary_index, int(target.shape[0]), hidden, requirements
            )
            fatal = _fatal_device_failure(error)
            LOGGER.warning(
                'H3 dual attention %s: mode=%s tokens=%d error=%s.',
                'aborted' if fatal else 'quarantined for single-device retry',
                dual_mode, int(target.shape[0]), type(error).__name__,
            )
            if fatal:
                propagating_error = True
                raise
            on_recovery = self._quarantine(failure_key, error, preflight=not cuda_started)
        except BaseException:
            propagating_error = True
            raise
        finally:
            cleanup_failure = None

            def record_cleanup_failure(error):
                nonlocal cleanup_failure
                if cleanup_failure is None and _fatal_device_failure(error):
                    # Keep the diagnostic, not a traceback retaining CUDA locals.
                    cleanup_failure = RuntimeError(str(error))

            try:
                for future in values.get('pending') or ():
                    try:
                        future.result()
                    except Exception as error:
                        record_cleanup_failure(error)
                for stream in values.get('sol_streams') or ():
                    try:
                        stream.close()
                    except Exception as error:
                        record_cleanup_failure(error)
                qkv_futures = audio_futures = secondary_future = None
                primary_part = None
                workspace = getattr(values.get('primary_qkv'), 'secondary', None)
                if workspace is not None:
                    from .graph_workspace import release_secondary
                    try:
                        release_secondary(self, workspace)
                    except Exception as error:
                        record_cleanup_failure(error)
                        LOGGER.warning('H3 secondary QKV cleanup failed.', exc_info=True)
                for key in tuple(values):
                    values[key] = None
                if cuda_started:
                    try:
                        torch.cuda.synchronize(primary)
                    except Exception as error:
                        record_cleanup_failure(error)
                    if not graph_owned:
                        try:
                            torch.cuda.synchronize(secondary)
                        except Exception as error:
                            record_cleanup_failure(error)
            finally:
                self._lock.release()
            if cleanup_failure is not None and not propagating_error:
                raise cleanup_failure from None
        if admission_rejected:
            return fallback(target, rope_freqs=rope_freqs, transformer_options=transformer_options)
        for device in (primary, secondary):
            try:
                if graph_owned and device == secondary:
                    from .graph_attention import release_cached_secondary
                    self._graph_executor.submit(_inference_worker, release_cached_secondary, secondary).result()
                    continue
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            except Exception:
                pass
        raise DualAttentionRetry(*failure, partial_target_write=receipt['fresh_target_required'], fallback=fallback, devices=(primary_index, secondary_index), on_recovery=on_recovery) from None

def make_retrying_block_core(single_attention):
    """Recreate disposable norm1 input before a transactional fallback."""

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
        shifts = self.adaln_proj(t_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = shifts
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        retry_info = None
        try:
            attention_out = self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        except DualAttentionRetry as retry:
            retry_info = (retry.fallback, retry.devices, retry.on_recovery)
            retry.__traceback__ = None
        if retry_info is not None:
            fallback, devices, on_recovery = retry_info
            del h
            for device_index in devices:
                try:
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
            attention_out = fallback(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        x = _mod_gate(x, gate_msa, attention_out, mod_segments)
        del h, attention_out
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        result = _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
        if retry_info is not None and on_recovery is not None:
            on_recovery()
        return result
    setattr(forward, BLOCK_PATCH_MARKER, True)
    return forward

def make_dual_attention_forward(single_attention, state):

    def forward(self, x, rope_freqs=None, transformer_options={}):
        sol_state = None
        if not isinstance(transformer_options, dict):
            return single_attention(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        bypass_reason = state._base_bypass_reason(x, transformer_options)
        if bypass_reason is not None:
            return single_attention(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        from .sol_attention import MODE_SOL
        dual_mode = 'exact'
        sol_config = None
        if transformer_options.get(SELECTED_BACKEND_KEY) == MODE_SOL:
            from .fused_sol_speed import DUAL_ROUTE_EXACT, DUAL_ROUTE_SOL_SPARSE, STATE_KEY as SOL_STATE_KEY, _layer_tau_delta, dual_high_level_route
            from .sol_calibration import AUDIO_RANGES_KEY
            sol_state = transformer_options.get(SOL_STATE_KEY)
            owner, reason = dual_high_level_route(sol_state, transformer_options, int(x.shape[0]))
            if owner not in (DUAL_ROUTE_EXACT, DUAL_ROUTE_SOL_SPARSE):
                return single_attention(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
            if owner == DUAL_ROUTE_SOL_SPARSE:
                dual_mode = 'sol_sparse'
                block_index = int(transformer_options.get('v100_h3_block_index', 0))
                effective_tau = max(0.0, min(4.0, float(sol_state.tau) + _layer_tau_delta(transformer_options, block_index, int(x.shape[0]))))
                dual_audio_ranges = tuple(transformer_options.get(AUDIO_RANGES_KEY, ()))
                sol_config = {'tau': effective_tau, 'prefix_stop': int(transformer_options.get('v100_sol_attention_prefix_stop', 0)), 'memory_limit_mib': int(sol_state.memory_limit_mib), 'stream_chunk_tokens': int(state.query_chunk), 'audio_ranges': dual_audio_ranges, 'audio_overwrite_active': bool(dual_audio_ranges)}
        result = state.run(self, x, rope_freqs, transformer_options, single_attention, sol_config=sol_config)
        return result
    setattr(forward, PATCH_MARKER, True)
    return forward

def dual_exact_attention_outer_sample_wrapper(executor, *args, **kwargs):
    options = getattr(getattr(executor, 'class_obj', None), 'model_options', {}).get('transformer_options', {})
    state = options.get(OPTION_KEY)
    if not isinstance(state, DualExactAttentionState):
        return executor(*args, **kwargs)
    try:
        state.begin_sample()
        return executor(*args, **kwargs)
    finally:
        state.release_host_pool()

class _DualAttentionInstaller:
    """Compatibility implementation used internally by H3V100Optimize."""

    def patch(self, model, run_nonce=0, secondary_device='auto'):
        import comfy.patcher_extension
        from .h3_mixed_precision import BLOCK_PATCH_MARKER as MAIN_BLOCK_MARKER, PATCH_MARKER as MAIN_ATTENTION_MARKER, _make_h3_block_forward
        from .sol_attention import MODE_FLASH, MODE_SOL
        from .weight_profile import FP8_E4M3_PROFILE, WEIGHT_PROFILE_OPTION_KEY
        patched = model.clone()
        patched.model_options = dict(patched.model_options)
        options = dict(patched.model_options.get('transformer_options', {}))
        patched.model_options['transformer_options'] = options
        if options.get(SELECTED_BACKEND_KEY) not in (MODE_FLASH, MODE_SOL):
            raise RuntimeError('Dual Attention requires the main H3 V100 Optimize node with attention_backend=flash_attn or sol_attn.')
        if options.get(SELECTED_BACKEND_KEY) == MODE_SOL:
            from .fused_sol_speed import STATE_KEY as SOL_STATE_KEY
            from .sol_calibration import HardSparseSpeedState
            if not isinstance(options.get(SOL_STATE_KEY), HardSparseSpeedState):
                raise RuntimeError('Dual Attention requires the integrated Sol state when attention_backend=sol_attn.')
        adaptive_budget_forced_off = bool(options.pop(ADAPTIVE_BUDGET_POLICY_KEY, None) is not None)
        if options.get(WEIGHT_PROFILE_OPTION_KEY) not in (FP8_E4M3_PROFILE, INT8_CONVROT_PROFILE):
            raise RuntimeError('Dual Attention requires a validated FP8 E4M3 scaled or INT8-ConvRot H3 core.')
        if options.get('prefetch_dynamic_vbars', False):
            raise RuntimeError('Dual Exact Attention requires Dynamic VBAR prefetch to remain disabled.')
        if OPTION_KEY in options:
            raise RuntimeError('Only one dual exact-attention state may be active.')
        diffusion_model = patched.get_model_object('diffusion_model')
        blocks = getattr(diffusion_model, 'blocks', None)
        if not blocks:
            raise RuntimeError('Dual Exact Attention expected MiniMax H3 blocks.')
        state = DualExactAttentionState(secondary_index=_parse_secondary_device(secondary_device), weight_profile=options[WEIGHT_PROFILE_OPTION_KEY])
        options[OPTION_KEY] = state
        for index, block in enumerate(blocks):
            attention_key = f'diffusion_model.blocks.{index}.attn.forward'
            single_attention = patched.object_patches.get(attention_key)
            function = getattr(single_attention, '__func__', single_attention)
            if single_attention is None or not getattr(function, MAIN_ATTENTION_MARKER, False):
                raise RuntimeError(f'block {index} is missing the validated H3 V100 attention patch')
            block_key = f'diffusion_model.blocks.{index}.forward'
            main_block = patched.object_patches.get(block_key)
            block_function = getattr(main_block, '__func__', main_block)
            if main_block is None or not getattr(block_function, MAIN_BLOCK_MARKER, False):
                raise RuntimeError(f'block {index} is missing the validated H3 V100 block patch')
            core = types.MethodType(make_retrying_block_core(single_attention), block)
            wrapped = _make_h3_block_forward(core, index, len(blocks))
            setattr(wrapped, BLOCK_PATCH_MARKER, True)
            patched.add_object_patch(block_key, types.MethodType(wrapped, block))
            patched.add_object_patch(attention_key, types.MethodType(make_dual_attention_forward(single_attention, state), block.attn))
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, LIFECYCLE_WRAPPER_KEY, dual_exact_attention_outer_sample_wrapper)
        return (patched,)

def patch_model_for_dual_attention(model, *, secondary_device='auto'):
    """Install the integrated main-node dual route and return its model clone."""
    patched, = _DualAttentionInstaller().patch(model, run_nonce=0, secondary_device=secondary_device)
    return patched
