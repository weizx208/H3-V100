"""Shared CUDA allocator-cache policy for the stable H3 V100 pipeline.

The Dynamic VBAR loader can only use driver-visible free memory.  PyTorch's
inactive allocator blocks are reusable by PyTorch, but are invisible to that
loader and to direct CUDA allocations.  This module gives QKV, projection,
MLP, Flash and Sol one demand-aware reclaim decision instead of several
uncoordinated ``empty_cache`` heuristics.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import torch
from .native_dynamic_vbar import CONTROLLER_KEY
LOGGER = logging.getLogger('H3V100Memory')
POLICY_KEY = 'v100_h3_runtime_memory_policy'
RUNTIME_KEY = 'v100_h3_runtime_memory_state'
MIB = 1024 ** 2

@dataclass(frozen=True)
class H3RuntimeMemoryPolicy:
    soft_free_floor_mib: int = 2048
    hard_free_floor_mib: int = 512
    minimum_reclaimable_mib: int = 512
    cooldown_checks: int = 12

def install_runtime_memory_policy(transformer_options, *, soft_free_floor_mib=2048, hard_free_floor_mib=512, minimum_reclaimable_mib=512, cooldown_checks=12):
    policy = H3RuntimeMemoryPolicy(max(0, int(soft_free_floor_mib)), max(0, int(hard_free_floor_mib)), max(0, int(minimum_reclaimable_mib)), max(0, int(cooldown_checks)))
    transformer_options[POLICY_KEY] = policy
    transformer_options.setdefault(RUNTIME_KEY, {'cooldown_remaining': 0})
    return policy

def memory_snapshot(device):
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = int(torch.cuda.memory_allocated(device))
    reserved_bytes = int(torch.cuda.memory_reserved(device))
    return {'free_bytes': int(free_bytes), 'total_bytes': int(total_bytes), 'allocated_bytes': allocated_bytes, 'reserved_bytes': reserved_bytes, 'reclaimable_bytes': max(0, reserved_bytes - allocated_bytes)}

def _runtime_state(transformer_options):
    return transformer_options.setdefault(RUNTIME_KEY, {'cooldown_remaining': 0})

def _request_cuda_headroom_impl(device, transformer_options, *, reason, required_free_bytes=None, snapshot=None, honor_cooldown=True, minimum_reclaimable_mib=None, soft_free_floor_mib=None, demand_bypass_cooldown=True, allow_vbar_release=False):
    """Release inactive allocator blocks only when a known demand needs them.

    ``required_free_bytes`` is the complete driver-visible budget needed by the
    caller, including its safety reserve. Inactive allocator blocks are not
    credited here: their total size does not prove that a sufficiently large
    contiguous block exists for the next full-sequence allocation. When the
    requirement is omitted, the shared soft floor is used.
    """
    if getattr(device, 'type', None) != 'cuda':
        return {'performed': False, 'reason': 'non-cuda', 'before': None, 'after': None, 'target_free_bytes': 0}
    policy = transformer_options.get(POLICY_KEY)
    if not isinstance(policy, H3RuntimeMemoryPolicy):
        policy = H3RuntimeMemoryPolicy()
    state = _runtime_state(transformer_options)
    before = memory_snapshot(device) if snapshot is None else snapshot
    total_bytes = int(before['total_bytes'])
    demand_request = required_free_bytes is not None
    if demand_request:
        target_free_bytes = max(0, int(required_free_bytes))
    else:
        soft_floor_mib = policy.soft_free_floor_mib if soft_free_floor_mib is None else max(0, int(soft_free_floor_mib))
        target_free_bytes = max(int(soft_floor_mib * MIB), int(total_bytes * 0.125))
    if int(before['free_bytes']) >= target_free_bytes:
        state['cooldown_remaining'] = max(0, int(state['cooldown_remaining']) - 1)
        return {'performed': False, 'reason': 'enough-driver-free', 'before': before, 'after': before, 'target_free_bytes': target_free_bytes}
    minimum_mib = policy.minimum_reclaimable_mib if minimum_reclaimable_mib is None else max(0, int(minimum_reclaimable_mib))
    if int(before['reclaimable_bytes']) < minimum_mib * MIB and (not allow_vbar_release):
        return {'performed': False, 'reason': 'insufficient-reclaimable-cache', 'before': before, 'after': before, 'target_free_bytes': target_free_bytes}
    hard_floor_bytes = max(int(policy.hard_free_floor_mib * MIB), int(total_bytes * 0.03))
    if (not demand_request or not demand_bypass_cooldown) and honor_cooldown and (int(before['free_bytes']) >= hard_floor_bytes) and (int(state['cooldown_remaining']) > 0):
        state['cooldown_remaining'] -= 1
        return {'performed': False, 'reason': 'cooldown', 'before': before, 'after': before, 'target_free_bytes': target_free_bytes}
    allocator_trimmed = int(before['reclaimable_bytes']) > 0 and int(before['reclaimable_bytes']) >= minimum_mib * MIB
    if allocator_trimmed:
        torch.cuda.empty_cache()
        after_allocator = memory_snapshot(device)
    else:
        after_allocator = before
    vbar_released = 0
    vbar_requested = False
    if allow_vbar_release and int(after_allocator['free_bytes']) < target_free_bytes:
        controller = transformer_options.get(CONTROLLER_KEY)
        release_unpinned = getattr(controller, 'release_unpinned', None)
        if callable(release_unpinned):
            vbar_requested = True
            missing_bytes = max(0, target_free_bytes - int(after_allocator['free_bytes']))
            vbar_released = max(0, int(release_unpinned(device, missing_bytes)))
    after = memory_snapshot(device) if vbar_requested else after_allocator
    performed = bool(allocator_trimmed or vbar_released)
    if performed:
        state['cooldown_remaining'] = int(policy.cooldown_checks)
    return {'performed': performed, 'reason': 'released-allocator-and-vbar' if allocator_trimmed and vbar_released else 'released-vbar' if vbar_released else 'released-allocator-cache' if allocator_trimmed else 'insufficient-releasable-headroom', 'before': before, 'after': after, 'target_free_bytes': target_free_bytes, 'vbar_released_bytes': vbar_released}

def request_cuda_headroom(device, transformer_options, **kwargs):
    """Reclaim activity is not proof that the allocation budget is satisfied."""
    result = _request_cuda_headroom_impl(device, transformer_options, **kwargs)
    after = result.get('after')
    if after is None:
        result.update(satisfied=None, shortfall_bytes=None)
    else:
        shortfall = max(0, int(result.get('target_free_bytes', 0)) - int(after['free_bytes']))
        result.update(satisfied=shortfall == 0, shortfall_bytes=shortfall)
    return result

def runtime_memory_outer_sample_wrapper(executor, *args, **kwargs):
    guider = getattr(executor, 'class_obj', None)
    options = getattr(guider, 'model_options', {}).get('transformer_options', {})
    options[RUNTIME_KEY] = {'cooldown_remaining': 0}
    options.pop('v100_h3_mlp_step_freezes', None)
    options['v100_h3_mlp_step_epoch'] = 0
    return executor(*args, **kwargs)

def patch_model_for_runtime_memory_lifecycle(model):
    import comfy.patcher_extension
    patched = model.clone()
    patched.model_options = dict(patched.model_options)
    patched.model_options['transformer_options'] = dict(patched.model_options.get('transformer_options', {}))
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, 'v100_h3_memory_lifecycle', runtime_memory_outer_sample_wrapper)
    return patched
