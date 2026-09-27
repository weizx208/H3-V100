"""Resource-adaptive two-device partition planner for H3.

The planner uses live free bytes as the authoritative capacity signal.  Total
VRAM only scales the reserve, so 16+16, 16+32 and 32+32 pairs share one policy.
It does not allocate CUDA memory and does not import ComfyUI.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
MIB = 1024 ** 2

@dataclass(frozen=True)
class DeviceMemory:
    index: int
    total_bytes: int
    free_bytes: int

@dataclass(frozen=True)
class HostMemory:
    total_bytes: int
    available_bytes: int

@dataclass(frozen=True)
class ParallelPlan:
    enabled: bool
    primary_units: int
    secondary_units: int
    primary_required_bytes: int
    secondary_required_bytes: int
    primary_reserve_bytes: int
    secondary_reserve_bytes: int
    primary_margin_bytes: int
    secondary_margin_bytes: int
    reason: str

@dataclass(frozen=True)
class HostStagingPlan:
    enabled: bool
    input_chunk_tokens: int
    attention_slots: int
    mlp_slots: int
    required_bytes: int
    reserve_bytes: int
    pinned_limit_bytes: int
    margin_bytes: int
    reason: str

def _reserve(memory: DeviceMemory) -> int:
    return max(256 * MIB, int(memory.total_bytes * 0.06))

def _best_partition(total_units: int, unit_alignment: int, primary: DeviceMemory, secondary: DeviceMemory, primary_fixed: int, secondary_fixed: int, primary_per_unit: int, secondary_per_unit: int, *, minimum_units: int, primary_preparation: tuple[int, int] | None=None, fixed_primary_units=None) -> ParallelPlan:
    primary_reserve = _reserve(primary)
    secondary_reserve = _reserve(secondary)
    candidates = []
    rejected = []
    for primary_units in range(minimum_units, total_units - minimum_units + 1, unit_alignment):
        if fixed_primary_units is not None and primary_units != fixed_primary_units:
            continue
        secondary_units = total_units - primary_units
        if secondary_units < minimum_units or secondary_units % unit_alignment:
            continue
        required0 = primary_fixed + primary_units * primary_per_unit
        if primary_preparation is not None:
            preparation_fixed, preparation_per_unit = primary_preparation
            required0 = max(required0, preparation_fixed + primary_units * preparation_per_unit)
        required1 = secondary_fixed + secondary_units * secondary_per_unit
        margin0 = primary.free_bytes - primary_reserve - required0
        margin1 = secondary.free_bytes - secondary_reserve - required1
        if margin0 < 0 or margin1 < 0:
            rejected.append((max(0, -margin0) / max(1, primary.total_bytes) + max(0, -margin1) / max(1, secondary.total_bytes), max(max(0, -margin0), max(0, -margin1)), max(primary_units, secondary_units), abs(primary_units - secondary_units), primary_units, secondary_units, required0, required1, margin0, margin1))
            continue
        slowest = max(primary_units, secondary_units)
        imbalance = abs(primary_units - secondary_units)
        normalized_margin = min(margin0 / max(1, primary.total_bytes), margin1 / max(1, secondary.total_bytes))
        candidates.append((slowest, imbalance, -normalized_margin, -margin0 - margin1, primary_units, secondary_units, required0, required1, margin0, margin1))
    if not candidates:
        if rejected:
            nearest = min(rejected)
            return ParallelPlan(False, nearest[4], nearest[5], nearest[6], nearest[7], primary_reserve, secondary_reserve, nearest[8], nearest[9], 'no-two-device-capacity')
        return ParallelPlan(False, 0, 0, 0, 0, primary_reserve, secondary_reserve, primary.free_bytes - primary_reserve - primary_fixed, secondary.free_bytes - secondary_reserve - secondary_fixed, 'no-two-device-capacity')
    best = min(candidates)
    return ParallelPlan(True, best[4], best[5], best[6], best[7], primary_reserve, secondary_reserve, best[8], best[9], 'balanced-live-capacity' if best[4] == best[5] else 'capacity-shifted')

def plan_attention_heads(tokens: int, query_chunk: int, primary: DeviceMemory, secondary: DeviceMemory, *, heads: int=56, head_dim: int=128, hidden: int=5376, sol_route: bool=False, pipeline_slots: int=2, minimum_heads: int=19, rope_table_elements_per_token: int=192, audio_rows: int=0, audio_key_chunk: int=1024, audio_query_chunk: int=512, primary_decode_extra_bytes: int=0, fixed_primary_heads=None, primary_qkv_preallocated=False) -> ParallelPlan:
    """Plan a head split from current memory rather than nominal card size."""
    if tokens <= 0 or query_chunk <= 0:
        raise ValueError('tokens and query_chunk must be positive')
    if heads <= 1 or head_dim <= 0 or hidden <= 0:
        raise ValueError('invalid attention geometry')
    if rope_table_elements_per_token < 0:
        raise ValueError('invalid RoPE geometry')
    if audio_rows < 0 or audio_key_chunk <= 0 or audio_query_chunk <= 0:
        raise ValueError('invalid audio geometry')
    if not 1 <= minimum_heads <= heads // 2:
        raise ValueError('minimum_heads must fit on both devices')
    if fixed_primary_heads is not None and not minimum_heads <= fixed_primary_heads <= heads - minimum_heads:
        raise ValueError('fixed primary heads do not fit the partition')
    if primary_decode_extra_bytes < 0:
        raise ValueError('primary decode workspace must be nonnegative')
    pipeline_slots = max(1, int(pipeline_slots))
    qkv_weight_per_head = 3 * head_dim * hidden * 2
    weight_per_head = qkv_weight_per_head + hidden * head_dim * 4 + 4 * head_dim * hidden
    qkv_per_head = tokens * 3 * head_dim * 2
    qk_scratch_per_head = query_chunk * head_dim * (3 * 2 + 2 * 4 + 2 * 2)
    attention_scratch_per_head = query_chunk * head_dim * (2 + 4)
    range_per_head = max(qk_scratch_per_head, attention_scratch_per_head)
    audio_per_head = 0
    if audio_rows:
        raw_audio_per_head = 2 * audio_rows * head_dim * 4 + 2 * audio_rows * 4 + 2 * audio_key_chunk * head_dim * 4 + 2 * audio_query_chunk * audio_key_chunk * 4 + audio_query_chunk * head_dim * 4
        audio_per_head = math.ceil(raw_audio_per_head * 1.6)
    blocks = math.ceil(tokens / 64)
    sol_per_head = 0
    if sol_route:
        route_score_rows = min(blocks, 512)
        sol_per_head = blocks * blocks + route_score_rows * blocks * 4 + blocks * blocks + blocks * head_dim * 4
    per_head = weight_per_head + qkv_per_head + range_per_head + audio_per_head + sol_per_head
    result_slots = pipeline_slots * query_chunk * hidden * 4
    full_prepared_weight = max(3 * heads * head_dim * hidden * (2 + 1), hidden * heads * head_dim * (4 + 1))
    primary_fixed = result_slots + full_prepared_weight + int(primary_decode_extra_bytes)
    secondary_fixed = result_slots
    primary_per_head = per_head
    secondary_per_head = per_head
    primary_preparation = None
    if sol_route:
        full_qkv_weight = heads * qkv_weight_per_head
        full_qk_chunk_scratch = heads * qk_scratch_per_head
        sol_transfer_reserve = 128 * MIB
        primary_per_head = primary_per_head - qkv_weight_per_head - range_per_head + attention_scratch_per_head
        secondary_per_head = secondary_per_head - qkv_weight_per_head - range_per_head + attention_scratch_per_head
        primary_fixed += full_qk_chunk_scratch + sol_transfer_reserve
        secondary_fixed += full_qkv_weight + full_qk_chunk_scratch + sol_transfer_reserve + query_chunk * hidden * 2 + tokens * rope_table_elements_per_token * 4
    else:
        secondary_fixed += tokens * hidden * 2 + tokens * rope_table_elements_per_token * 4
        # Exact preparation returns independent shards before allocating
        # full-sequence QKV. Require BOTH phase envelopes to fit. Keep
        # compressed-page charges, INT8 decode extra, and all reserves
        # in both envelopes; SOL genuinely overlaps these allocations.
        if not primary_qkv_preallocated:
            primary_preparation = (primary_fixed, primary_per_head - qkv_per_head)
            primary_fixed -= heads * qkv_weight_per_head
        # Acquired QKV stays live during weight preparation: retain that overlap.
    return _best_partition(heads, 1, primary, secondary, primary_fixed, secondary_fixed, primary_per_head, secondary_per_head, minimum_units=minimum_heads, primary_preparation=primary_preparation, fixed_primary_units=fixed_primary_heads)

def plan_mlp_channels(tokens: int, chunk_tokens: int, primary: DeviceMemory, secondary: DeviceMemory, *, intermediate: int=14336, hidden: int=5376, alignment: int=256, pipeline_slots: int=3, minimum_channels: int=5376) -> ParallelPlan:
    """Plan SwiGLU intermediate channels with FP8 storage and FP16 compute."""
    if tokens <= 0 or chunk_tokens <= 0:
        raise ValueError('tokens and chunk_tokens must be positive')
    if intermediate % alignment or minimum_channels % alignment:
        raise ValueError('intermediate and minimum_channels must align')
    if minimum_channels * 2 > intermediate:
        raise ValueError('minimum_channels must fit on both devices')
    pipeline_slots = max(2, int(pipeline_slots))
    expanded_per_channel = 2 * hidden * 2 + hidden * 2
    compressed_per_channel = 2 * hidden + hidden
    scratch_per_channel = chunk_tokens * (2 * 2 + 2)
    per_channel = expanded_per_channel + compressed_per_channel + scratch_per_channel
    output_slots = pipeline_slots * chunk_tokens * hidden * 4
    full_prepared_weight = max(2 * intermediate * hidden * (2 + 1), hidden * intermediate * (2 + 1))
    primary_fixed = output_slots + full_prepared_weight
    secondary_fixed = chunk_tokens * hidden * 2 + output_slots
    return _best_partition(intermediate, alignment, primary, secondary, primary_fixed, secondary_fixed, per_channel, per_channel, minimum_units=minimum_channels)

def plan_host_staging(tokens: int, query_chunk: int, host: HostMemory, attention: ParallelPlan, mlp: ParallelPlan, *, hidden: int=5376, head_dim: int=128, mlp_chunk_tokens: int=640, sol_route: bool=False, preferred_input_chunk: int=8192, minimum_input_chunk: int=1024, rope_table_elements_per_token: int=192, use_attention: bool=True, use_mlp: bool=True, attention_slot_options: tuple[int, ...]=(2,), mlp_slot_options: tuple[int, ...]=(3, 2)) -> HostStagingPlan:
    """Plan a bounded pinned pool for host-staged cross-device transfers.

    The host already owns compressed checkpoint pages.  This estimate covers
    only the temporary pinned transfer pool: chunked FP16 activation staging,
    FP32 partial results, the largest prepared dense projection/MLP shard, and
    a bounded Sol route buffer.  Buffers are reused across blocks.
    """
    if tokens <= 0 or query_chunk <= 0 or hidden <= 0 or (head_dim <= 0) or (mlp_chunk_tokens <= 0):
        raise ValueError('invalid host staging geometry')
    if minimum_input_chunk <= 0 or preferred_input_chunk < minimum_input_chunk:
        raise ValueError('invalid input chunk range')
    if rope_table_elements_per_token < 0:
        raise ValueError('invalid RoPE staging geometry')
    if use_attention and (not attention_slot_options or any((int(value) <= 0 for value in attention_slot_options))):
        raise ValueError('invalid attention slot options')
    if use_mlp and (not mlp_slot_options or any((int(value) < 2 for value in mlp_slot_options))):
        raise ValueError('invalid MLP slot options')
    if host.total_bytes <= 0 or not 0 <= host.available_bytes <= host.total_bytes:
        raise ValueError('invalid host memory snapshot')
    reserve = max(2 * 1024 ** 3, int(host.total_bytes * 0.05))
    usable = max(0, host.available_bytes - reserve)
    pinned_limit = min(int(host.total_bytes * 0.08), int(usable * 0.2))
    if not use_attention and (not use_mlp):
        return HostStagingPlan(False, 0, 0, 0, 0, reserve, pinned_limit, usable, 'no-active-phase')
    if use_attention and (not attention.enabled) or (use_mlp and (not mlp.enabled)):
        return HostStagingPlan(False, 0, 0, 0, 0, reserve, pinned_limit, usable, 'gpu-plan-disabled')
    secondary_heads = attention.secondary_units if use_attention else 0
    staged_qkv_heads = attention.primary_units + attention.secondary_units if use_attention and sol_route else secondary_heads
    secondary_channels = mlp.secondary_units if use_mlp else 0
    staged_attention = max(3 * staged_qkv_heads * head_dim * hidden * 2, hidden * secondary_heads * head_dim * 4) if use_attention else 0
    staged_mlp = max(2 * secondary_channels * hidden * 2, hidden * secondary_channels * 2) if use_mlp else 0
    staged_weight = max(staged_attention, staged_mlp)
    blocks = math.ceil(tokens / 64)
    route_buffer = 3 * blocks * blocks + blocks * head_dim * 4 if sol_route else 0
    chunks = []
    value = min(tokens, preferred_input_chunk)
    while value >= minimum_input_chunk:
        chunks.append(value)
        value //= 2
    if min(tokens, preferred_input_chunk) < minimum_input_chunk:
        chunks.append(tokens)
    chunks = tuple(dict.fromkeys((max(1, int(value)) for value in chunks)))
    attention_candidates = tuple((int(value) for value in attention_slot_options)) if use_attention else (0,)
    mlp_candidates = tuple((int(value) for value in mlp_slot_options)) if use_mlp else (0,)
    for attention_slots in attention_candidates:
        for mlp_slots in mlp_candidates:
            for input_chunk in chunks:
                activation = attention_slots * input_chunk * hidden * 2 if use_attention else 0
                partials = attention_slots * query_chunk * hidden * 4 if use_attention else 0
                mlp_stage = mlp_slots * mlp_chunk_tokens * hidden * (2 + 2) if use_mlp else 0
                rope_stage = input_chunk * rope_table_elements_per_token * 4 if use_attention else 0
                staged_raw = max(staged_weight, rope_stage)
                required = max(activation + partials, mlp_stage) + staged_raw + route_buffer
                margin = usable - required
                if margin >= 0 and required <= pinned_limit:
                    reason = 'preferred-pipeline' if attention_slots == attention_candidates[0] and mlp_slots == mlp_candidates[0] and (input_chunk == min(tokens, preferred_input_chunk)) else 'reduced-host-staging'
                    return HostStagingPlan(True, input_chunk, attention_slots, mlp_slots, required, reserve, pinned_limit, margin, reason)
    return HostStagingPlan(False, 0, 0, 0, 0, reserve, pinned_limit, usable, 'insufficient-host-staging')
