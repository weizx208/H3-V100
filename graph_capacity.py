"""Acquire graph-owned QKV before budgeting the remaining attention peak.

Only live tensors supplied to the executor earn capacity credit. Graph physical
bytes and historical peaks never count as free space. Credits die with this
one attention call; no graph storage escapes a block allocation scope.
"""
import threading
from dataclasses import replace
import torch
from .dual_runtime import DualAdaptiveAdmission, torch_device_memory
from .graph_workspace import AcquiredQKV, acquire_secondary, release_secondary


def active_primary_graph(device):
    if torch.cuda.get_allocator_backend() != 'cudaMallocAsync':
        return None
    try:
        import comfy.model_prefetch as prefetch
    except ImportError:
        return None
    graph = prefetch.MALLOC_GRAPHS.get(threading.get_ident())
    if graph is None or not getattr(graph, '_comfy_active', False):
        return None
    owner = getattr(getattr(graph, '_stream', None), 'device', None)
    if owner != device:
        raise RuntimeError('H3 allocation graph belongs to another device')
    return graph


def _trim_secondary_cache(index):
    # Called only on the secondary owner, between completed attention calls.
    # Never infer capacity from its cached-byte total: remeasure after trimming.
    with torch.cuda.device(index):
        torch.cuda.empty_cache()


def _candidate_heads(total, minimum):
    """Search every supported split, preferring the least imbalanced work.

    The preliminary planner has no credit for graph-reusable QKV. Its chosen
    split therefore cannot define the feasible set of this acquired planner.
    Both directions must remain available when either device is constrained.
    """
    return sorted(range(minimum, total-minimum+1),
                  key=lambda heads: (max(heads, total-heads), heads))


def acquire(state, primary, tokens, hidden, head_dim, options, requirements):
    from .runtime_memory import request_cuda_headroom
    initial = state._select_admission(primary, tokens, hidden, **requirements)
    old_runtime, old_admission, reason = initial
    if old_runtime is None:
        return initial, None, requirements
    secondary = old_runtime.secondary_index
    if head_dim != 128 or hidden != 5376:
        raise RuntimeError('H3 acquired-workspace planner requires validated geometry')
    device = torch.device('cuda', primary)
    total_heads = 56
    minimum = int(requirements['attention_minimum_heads'])
    # Prefer balanced work, then examine BOTH redistribution directions. Each
    # candidate still needs fresh secondary/RAM checks and actual primary QKV.
    candidates = _candidate_heads(total_heads, minimum)
    trimmed_secondary = False
    for heads in candidates:
        req = dict(requirements, attention_primary_heads=heads,
                   attention_qkv_preallocated=True)
        def potential_snapshot(index):
            memory = torch_device_memory(index)
            # Only reject impossible physical geometry/RAM here. Both devices'
            # QKV will actually be acquired before admitting any computation.
            return replace(memory, free_bytes=memory.total_bytes)
        def plan_potential():
            return DualAdaptiveAdmission(primary, secondary,
                gpu_snapshot=potential_snapshot, host_snapshot=state._host_snapshot,
                performance_floor_tokens=state.performance_floor_tokens).prepare(tokens, **req)
        potential = plan_potential()
        if not potential.enabled:
            continue  # Even nominal physical capacity or host staging cannot fit.
        required = potential.attention.primary_required_bytes + potential.attention.primary_reserve_bytes
        qkv = None
        try:
            qkv = tuple(torch.empty((1, heads, tokens, head_dim),
                                   dtype=torch.float16, device=device) for _ in range(3))
        except torch.cuda.OutOfMemoryError:
            pass
        if qkv is None:
            request_cuda_headroom(device, options, reason='dual-graph-qkv',
                required_free_bytes=required, minimum_reclaimable_mib=0,
                honor_cooldown=False, allow_vbar_release=True)
            try:
                qkv = tuple(torch.empty((1, heads, tokens, head_dim),
                                       dtype=torch.float16, device=device) for _ in range(3))
            except torch.cuda.OutOfMemoryError:
                continue
        secondary_owner = None
        transferred = False
        try:
            secondary_owner = acquire_secondary(state, secondary, total_heads-heads, tokens, head_dim)
            if secondary_owner is None and not trimmed_secondary:
                state._graph_executor.submit(_trim_secondary_cache, secondary).result()
                trimmed_secondary = True
                secondary_owner = acquire_secondary(state, secondary, total_heads-heads, tokens, head_dim)
            if secondary_owner is None:
                continue
            owned_bytes = sum(t.numel() * t.element_size() for t in qkv)
            secondary_bytes = secondary_owner.nbytes
            def owned_snapshot(index, primary_credit=owned_bytes, secondary_credit=secondary_bytes):
                memory = torch_device_memory(index)
                credit = primary_credit if index == primary else secondary_credit
                return replace(memory, free_bytes=min(memory.total_bytes, memory.free_bytes + credit))
            runtime = DualAdaptiveAdmission(primary, secondary,
                gpu_snapshot=owned_snapshot, host_snapshot=state._host_snapshot,
                performance_floor_tokens=state.performance_floor_tokens)
            admission = runtime.prepare(tokens, **req)
            if not admission.enabled and admission.attention.secondary_margin_bytes < 0 and not trimmed_secondary:
                state._graph_executor.submit(_trim_secondary_cache, secondary).result()
                trimmed_secondary = True
                admission = runtime.prepare(tokens, **req)
            if not admission.enabled and admission.attention.primary_margin_bytes < 0:
                request_cuda_headroom(device, options, reason='dual-graph-remainder',
                    required_free_bytes=max(0, required-owned_bytes),
                    minimum_reclaimable_mib=0, honor_cooldown=False,
                    allow_vbar_release=True)
                admission = runtime.prepare(tokens, **req)
            if admission.enabled:
                result = AcquiredQKV(qkv, secondary_owner)
                transferred = True
                return (runtime, admission, None), result, req
        finally:
            if not transferred:
                release_secondary(state, secondary_owner)
            del qkv
    # No allocation-backed plan fitted. Caller retains its normal safe fallback.
    return (old_runtime, replace(old_admission, enabled=False,
            reason='graph-workspace-unavailable'), reason), None, requirements
