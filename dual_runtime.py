"""Runtime admission and transaction contract for dual-V100 execution.

This module does not patch ComfyUI.  It isolates the state transitions that a
future unregistered adapter must obey before it owns CUDA tensors or writes an
attention/MLP result.
"""
from __future__ import annotations
from dataclasses import dataclass
import ctypes
import os
from typing import Callable
try:
    from .dual_gpu_plan import DeviceMemory, HostMemory, HostStagingPlan, ParallelPlan, plan_attention_heads, plan_host_staging, plan_mlp_channels
except ImportError:
    from dual_gpu_adaptive_plan import DeviceMemory, HostMemory, HostStagingPlan, ParallelPlan, plan_attention_heads, plan_host_staging, plan_mlp_channels

@dataclass(frozen=True)
class RuntimeAdmission:
    enabled: bool
    epoch: int
    tokens: int
    primary: DeviceMemory
    secondary: DeviceMemory
    host: HostMemory
    attention: ParallelPlan
    mlp: ParallelPlan
    staging: HostStagingPlan
    active_phases: tuple[str, ...]
    reason: str

@dataclass
class OperationTransaction:
    """Track whether a fallback needs a fresh output target."""
    name: str
    state: str = 'prepared'
    partial_target_write: bool = False

    def start(self):
        if self.state != 'prepared':
            raise RuntimeError(f'{self.name}: transaction is {self.state}')
        self.state = 'running'

    def mark_partial_target_write(self):
        if self.state != 'running':
            raise RuntimeError(f'{self.name}: cannot write while {self.state}')
        self.partial_target_write = True

    def commit(self):
        if self.state != 'running':
            raise RuntimeError(f'{self.name}: cannot commit while {self.state}')
        self.state = 'committed'

    def abort(self):
        if self.state not in ('prepared', 'running'):
            raise RuntimeError(f'{self.name}: cannot abort while {self.state}')
        self.state = 'aborted'
        return {'fallback_allowed': True, 'fresh_target_required': bool(self.partial_target_write), 'restart_scope': self.name}

def system_host_memory() -> HostMemory:
    """Read live host memory without adding a psutil dependency."""
    if os.name == 'nt':

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = (('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong))
        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            raise OSError('GlobalMemoryStatusEx failed')
        return HostMemory(int(state.ullTotalPhys), int(state.ullAvailPhys))
    page = os.sysconf('SC_PAGE_SIZE')
    total = page * os.sysconf('SC_PHYS_PAGES')
    available = page * os.sysconf('SC_AVPHYS_PAGES')
    return HostMemory(int(total), int(available))

def torch_device_memory(index: int) -> DeviceMemory:
    """Read driver-visible free/total bytes without allocating a CUDA tensor."""
    import torch
    free, total = torch.cuda.mem_get_info(torch.device('cuda', int(index)))
    return DeviceMemory(int(index), int(total), int(free))

def validate_dual_v100_pair(primary_index: int, secondary_index: int):
    """Return an eligibility receipt for the exact first implementation."""
    import torch
    if primary_index == secondary_index:
        return (False, 'same-device')
    if torch.cuda.device_count() <= max(primary_index, secondary_index):
        return (False, 'missing-device')
    devices = []
    for index in (primary_index, secondary_index):
        devices.append({'index': index, 'name': torch.cuda.get_device_name(index), 'capability': tuple(torch.cuda.get_device_capability(index))})
    if any((device['capability'] != (7, 0) for device in devices)):
        return (False, 'unsupported-compute-capability')
    if any(('V100' not in device['name'].upper() for device in devices)):
        return (False, 'not-v100')
    return (True, devices)

class DualAdaptiveAdmission:
    """Create a fresh plan before any dual-device allocation or target write."""

    def __init__(self, primary_index: int, secondary_index: int, *, gpu_snapshot: Callable[[int], DeviceMemory]=torch_device_memory, host_snapshot: Callable[[], HostMemory]=system_host_memory, performance_floor_tokens: int=16384):
        if primary_index == secondary_index:
            raise ValueError('primary and secondary devices must differ')
        self.primary_index = int(primary_index)
        self.secondary_index = int(secondary_index)
        self.gpu_snapshot = gpu_snapshot
        self.host_snapshot = host_snapshot
        self.performance_floor_tokens = max(1, int(performance_floor_tokens))
        self._epoch = 0

    def prepare(self, tokens: int, *, query_chunk: int=2048, mlp_chunk: int=640, sol_route: bool=False, active_phases: tuple[str, ...]=('attention', 'mlp'), audio_rows: int=0, audio_key_chunk: int=1024, host_attention_slots: int=2, attention_minimum_heads: int=19, attention_decode_extra_bytes: int=0) -> RuntimeAdmission:
        """Sample both GPUs and RAM once, then bind every plan to one epoch."""
        active_phases = tuple(dict.fromkeys((str(value) for value in active_phases)))
        unknown = set(active_phases).difference(('attention', 'mlp'))
        if unknown or not active_phases:
            raise ValueError(f'invalid active phases: {active_phases}')
        self._epoch += 1
        primary = self.gpu_snapshot(self.primary_index)
        secondary = self.gpu_snapshot(self.secondary_index)
        host = self.host_snapshot()
        if tokens < self.performance_floor_tokens:
            disabled = ParallelPlan(False, 0, 0, 0, 0, 0, 0, 0, 0, 'below-performance-floor')
            staging = HostStagingPlan(False, 0, 0, 0, 0, 0, 0, 0, 'gpu-plan-disabled')
            return RuntimeAdmission(False, self._epoch, tokens, primary, secondary, host, disabled, disabled, staging, active_phases, 'below-performance-floor')
        attention = plan_attention_heads(tokens, query_chunk, primary, secondary, sol_route=sol_route, audio_rows=audio_rows, audio_key_chunk=audio_key_chunk, minimum_heads=attention_minimum_heads, primary_decode_extra_bytes=attention_decode_extra_bytes)
        mlp = plan_mlp_channels(tokens, mlp_chunk, primary, secondary) if 'mlp' in active_phases else ParallelPlan(False, 0, 0, 0, 0, 0, 0, 0, 0, 'phase-inactive')
        staging = plan_host_staging(tokens, query_chunk, host, attention, mlp, mlp_chunk_tokens=mlp_chunk, sol_route=sol_route, use_attention='attention' in active_phases, use_mlp='mlp' in active_phases, attention_slot_options=(max(1, int(host_attention_slots)),))
        attention_required = 'attention' in active_phases
        mlp_required = 'mlp' in active_phases
        enabled = (attention.enabled or not attention_required) and (mlp.enabled or not mlp_required) and staging.enabled
        reason = 'admitted' if enabled else attention.reason if attention_required and (not attention.enabled) else mlp.reason if mlp_required and (not mlp.enabled) else staging.reason
        return RuntimeAdmission(enabled, self._epoch, tokens, primary, secondary, host, attention, mlp, staging, active_phases, reason)

    def capacity_still_valid(self, admission: RuntimeAdmission, *, committed_host_bytes: int=0) -> bool:
        """Check a lease immediately before allocating its first CUDA buffer.

        Callers must prepare a new admission if any CUDA allocation, release,
        allocator trim, VBAR page transition, or host-pool change occurs
        between ``prepare`` and this check.  Allocations and VBAR transitions
        that belong to the admitted operation begin only after this check and
        are covered by that operation's conservative peak estimate.
        """
        if not admission.enabled or admission.epoch != self._epoch:
            return False
        primary = self.gpu_snapshot(self.primary_index)
        secondary = self.gpu_snapshot(self.secondary_index)
        host = self.host_snapshot()
        host_reserve = max(2 * 1024 ** 3, int(host.total_bytes * 0.05))
        host_usable = max(0, host.available_bytes - host_reserve)
        current_pinned_limit = min(int(host.total_bytes * 0.08), int(host_usable * 0.2))
        primary_required = max((getattr(admission, phase).primary_required_bytes for phase in admission.active_phases), default=0)
        secondary_required = max((getattr(admission, phase).secondary_required_bytes for phase in admission.active_phases), default=0)
        primary_reserve = max((getattr(admission, phase).primary_reserve_bytes for phase in admission.active_phases), default=0)
        secondary_reserve = max((getattr(admission, phase).secondary_reserve_bytes for phase in admission.active_phases), default=0)
        return primary.free_bytes >= primary_required + primary_reserve and secondary.free_bytes >= secondary_required + secondary_reserve and (host.available_bytes >= admission.staging.required_bytes + host_reserve) and (max(admission.staging.required_bytes, max(0, int(committed_host_bytes))) <= current_pinned_limit)


def _fatal_device_failure(error):
    return any((text in str(error).lower() for text in ('illegal memory access', 'device-side assert', 'misaligned address', 'unspecified launch failure', 'device has been lost', 'device is lost')))
