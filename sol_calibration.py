"""Opt-in lossy hard-sparse attention for the H3 V100 speed profile.

The stable H3 V100 override remains exact Flash Attention.  This adapter is a
runtime admission layer: one eligible layer calibrates the complete returned
operator against Flash, then the current prompt/shape is allowed to use the
lossy path only when the measured gain clears the configured floor.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable
import torch
LOGGER = logging.getLogger('H3V100SolSpeed')
STATE_KEY = 'v100_h3_hard_sparse_speed_state'
STABLE_VBAR_KEY = 'v100_h3_dynamic_vbar_controller'
AUDIO_RANGES_KEY = 'minimax_h3_fp32_audio_ranges'
AUDIO_OVERWRITE_ACTIVE_KEY = 'v100_h3_fp32_audio_overwrite_active'
PROTECTED_LAYERS = (0, 1, 40, 49)
CALIBRATION_LAYERS = (2, 24, 32)
STREAM_OUTPUT_KEY = 'v100_h3_sol_stream_output'

class _MemoryBudgetExceeded(RuntimeError):
    pass

def _merged_audio_ranges(audio_ranges, tokens: int):
    """Normalize valid half-open audio intervals without trusting metadata."""
    valid = []
    for value in audio_ranges or ():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            continue
        start, stop = value
        if not isinstance(start, int) or not isinstance(stop, int):
            continue
        if 0 <= start < stop <= int(tokens):
            valid.append((int(start), int(stop)))
    valid.sort()
    merged = []
    for start, stop in valid:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(merged)

def _full_audio_query_block_ranges(audio_ranges, tokens: int, block_size=64):
    """Return block-index spans wholly covered by merged audio intervals."""
    tokens = int(tokens)
    block_size = int(block_size)
    blocks = math.ceil(tokens / block_size)
    result = []
    for start, stop in _merged_audio_ranges(audio_ranges, tokens):
        first = math.ceil(start / block_size)
        last = stop // block_size
        if stop == tokens and tokens % block_size:
            last = blocks
        if first < last:
            result.append((first, last))
    return tuple(result)

@dataclass
class HardSparseSpeedState:
    tau: float
    topk_tail_blocks: int
    min_tokens: int
    minimum_gain: float
    memory_limit_mib: int
    admissions: dict[tuple, dict[str, Any]] = field(default_factory=dict)
    sigma_ordinals: dict[str, int] = field(default_factory=dict)
    schedule_ordinals: dict[str, int] = field(default_factory=dict)
    sigma_schedule: tuple[float, ...] = ()
    current_step: int | None = None
    step_count: int | None = None

    def reset_run(self, transformer_options: dict[str, Any], sample_sigmas=None) -> None:
        self.admissions.clear()
        self.sigma_ordinals.clear()
        self.schedule_ordinals.clear()
        self.current_step = None
        if sample_sigmas is None:
            sample_sigmas = transformer_options.get('sample_sigmas')
        from .sampling_schedule import sample_sigma_boundaries
        self.sigma_schedule = sample_sigma_boundaries(sample_sigmas)
        self.step_count = len(self.sigma_schedule) - 1 if self.sigma_schedule else None
        for ordinal, sigma in enumerate(self.sigma_schedule[:-1]):
            self.schedule_ordinals[f'{sigma:.12g}'] = ordinal

    def observe_step(self, transformer_options: dict[str, Any], block_index) -> None:
        if block_index not in (0, None) and self.current_step is not None:
            return
        sigmas = transformer_options.get('sigmas')
        if sigmas is None or not len(sigmas):
            self.current_step = None
            return
        sigma = float(sigmas[0])
        key = f'{sigma:.12g}'
        if self.schedule_ordinals:
            self.current_step = self.schedule_ordinals.get(key)
            if self.current_step is not None:
                self.sigma_ordinals.setdefault(key, self.current_step)
            return
        if key not in self.sigma_ordinals:
            self.sigma_ordinals[key] = len(self.sigma_ordinals)
        self.current_step = self.sigma_ordinals[key]

def _cuda_time(call: Callable[[], Any], device):
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = call()
        end.record()
        end.synchronize()
    return (value, float(start.elapsed_time(end)))

def _calibrate(candidate_call, exact_call, minimum_gain: float, device, warm=True, profiled_candidate_call=None):
    if warm:
        warm_output = candidate_call()
        torch.cuda.synchronize(device)
        del warm_output
    profile = None
    if profiled_candidate_call is None:
        candidate, candidate_ms = _cuda_time(candidate_call, device)
    else:
        candidate, profile = profiled_candidate_call()
        candidate_ms = float(profile['candidate_ms'])
    finite = bool(torch.isfinite(candidate).all().item())
    del candidate
    exact, exact_ms = _cuda_time(exact_call, device)
    ratio = candidate_ms / max(exact_ms, 1e-06)
    admitted = finite and ratio <= 1.0 - float(minimum_gain)
    sample = {'admitted': bool(admitted), 'candidate_ms': candidate_ms, 'flash_ms': exact_ms, 'ratio': ratio, 'gain': 1.0 - ratio, 'finite': finite}
    if profile is not None:
        sample['profile'] = profile
    return (exact, sample)
