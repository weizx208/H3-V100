"""H3-aware final-output cache coordinated with Flash/Sol refresh steps."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any
import torch
STATE_KEY = 'v100_h3_easycache_state'
ACTIVE_KEY = 'v100_h3_easycache_active'
MODE_OFF = 'off'
MODE_ACTIVE = 'active'
SUPPORTED_MODES = (MODE_OFF, MODE_ACTIVE)
STABLE_VBAR_KEY = 'v100_h3_dynamic_vbar_controller'
PROFILE_QUALITY = 'quality'
PROFILE_SPEED = 'speed'
PROFILE_MANUAL = 'manual'
SUPPORTED_PROFILES = (PROFILE_QUALITY, PROFILE_SPEED, PROFILE_MANUAL)
REFERENCE_STEPS = 20
CACHE_CHUNK_ELEMENTS = 2 * 1024 * 1024
HISTORY_TENSOR_FIELDS = ('previous_x_video', 'previous_x_audio', 'previous_output_video', 'previous_output_audio', 'previous_video_norm', 'previous_video_frame_norm', 'previous_audio_norm', 'video_rate', 'video_frame_rate', 'audio_rate', 'cumulative_video', 'cumulative_video_frame', 'cumulative_audio')

def _video_per_frame_abs_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce H3 [B,C,T,H,W] video values while preserving latent time."""
    if tensor.ndim < 3:
        return tensor.abs().reshape(-1)
    reduce_dims = tuple((index for index in range(tensor.ndim) if index != 2))
    return tensor.abs().mean(dim=reduce_dims)

@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    group_size: int
    elements: int

    @property
    def storage_bytes(self) -> int:
        return self.values.numel() * self.values.element_size() + self.scales.numel() * self.scales.element_size()

    def decode(self, *, device, dtype=torch.float32) -> torch.Tensor:
        result = self.values.to(device=device, dtype=torch.float32)
        result.mul_(self.scales.to(device=device).unsqueeze(1))
        return result.reshape(-1)[:self.elements].reshape(self.shape).to(dtype)

    def add_to_(self, target: torch.Tensor) -> torch.Tensor:
        """Decode group-aligned chunks directly into the destination."""
        target_flat = target.reshape(-1)
        values = self.values.reshape(-1)
        groups_per_chunk = max(1, CACHE_CHUNK_ELEMENTS // int(self.group_size))
        group_count = int(self.scales.numel())
        for group_start in range(0, group_count, groups_per_chunk):
            group_stop = min(group_count, group_start + groups_per_chunk)
            element_start = group_start * int(self.group_size)
            element_stop = min(int(self.elements), group_stop * int(self.group_size))
            if element_start >= element_stop:
                break
            decoded = values[element_start:group_stop * int(self.group_size)].to(device=target.device, dtype=torch.float32).reshape(group_stop - group_start, int(self.group_size))
            decoded.mul_(self.scales[group_start:group_stop].to(device=target.device, dtype=torch.float32).unsqueeze(1))
            target_flat[element_start:element_stop].add_(decoded.reshape(-1)[:element_stop - element_start].to(dtype=target.dtype))
            del decoded
        return target

@dataclass(frozen=True)
class VideoCache:
    storage: str
    fp16: torch.Tensor | None = None
    int8: QuantizedTensor | None = None

    @property
    def storage_bytes(self) -> int:
        if self.fp16 is not None:
            return self.fp16.numel() * self.fp16.element_size()
        return self.int8.storage_bytes

    def decode(self, reference: torch.Tensor) -> torch.Tensor:
        if self.fp16 is not None:
            return self.fp16.to(device=reference.device, dtype=reference.dtype)
        return self.int8.decode(device=reference.device, dtype=reference.dtype)

    def add_to_(self, target: torch.Tensor) -> torch.Tensor:
        if self.fp16 is not None:
            source = self.fp16.reshape(-1)
            target_flat = target.reshape(-1)
            for start in range(0, int(target_flat.numel()), CACHE_CHUNK_ELEMENTS):
                stop = min(int(target_flat.numel()), start + CACHE_CHUNK_ELEMENTS)
                target_flat[start:stop].add_(source[start:stop].to(device=target.device, dtype=target.dtype))
            return target
        return self.int8.add_to_(target)

@dataclass(frozen=True)
class AVCacheEntry:
    video: VideoCache
    audio_fp32: torch.Tensor

    @property
    def storage_bytes(self) -> int:
        return self.video.storage_bytes + self.audio_fp32.numel() * self.audio_fp32.element_size()

def _encode_video_difference(output: torch.Tensor, input_: torch.Tensor, storage: str, group_size: int) -> tuple[VideoCache, float | None, int]:
    """Encode output-input without a complete FP32 delta or decode copy."""
    elements = int(output.numel())
    output_flat = output.detach().reshape(-1)
    input_flat = input_.detach().reshape(-1)
    transient_peak = 0
    if storage == 'fp16':
        encoded = torch.empty(elements, dtype=torch.float16, device=output.device)
        for start in range(0, elements, CACHE_CHUNK_ELEMENTS):
            stop = min(elements, start + CACHE_CHUNK_ELEMENTS)
            delta = torch.empty(stop - start, dtype=torch.float32, device=output.device)
            torch.sub(output_flat[start:stop], input_flat[start:stop], out=delta)
            quantized = delta.half()
            encoded[start:stop].copy_(quantized)
            transient_peak = max(transient_peak, int(delta.numel()) * (4 + 2 + 4))
        cache = VideoCache(storage='fp16', fp16=encoded.reshape(output.shape))
    elif storage == 'int8':
        group_size = int(group_size)
        groups = math.ceil(elements / group_size)
        values = torch.empty((groups, group_size), dtype=torch.int8, device=output.device)
        scales = torch.empty(groups, dtype=torch.float32, device=output.device)
        groups_per_chunk = max(1, CACHE_CHUNK_ELEMENTS // group_size)
        for group_start in range(0, groups, groups_per_chunk):
            group_stop = min(groups, group_start + groups_per_chunk)
            start = group_start * group_size
            stop = min(elements, group_stop * group_size)
            delta = torch.empty(stop - start, dtype=torch.float32, device=output.device)
            torch.sub(output_flat[start:stop], input_flat[start:stop], out=delta)
            valid = int(delta.numel())
            padded = (group_stop - group_start) * group_size
            if valid < padded:
                delta = torch.nn.functional.pad(delta, (0, padded - valid))
            grouped = delta.reshape(group_stop - group_start, group_size)
            local_scales = grouped.abs().amax(dim=1).div_(127.0)
            safe = torch.where(local_scales > 0, local_scales, torch.ones_like(local_scales))
            quantized = torch.round(grouped / safe.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
            values[group_start:group_stop].copy_(quantized)
            scales[group_start:group_stop].copy_(local_scales)
            transient_peak = max(transient_peak, padded * (4 + 1 + 4) + (group_stop - group_start) * 8)
            del delta, grouped, local_scales, safe, quantized
        cache = VideoCache(storage='int8', int8=QuantizedTensor(values.contiguous(), scales.contiguous(), tuple(output.shape), group_size, elements))
    else:
        raise ValueError(f'unsupported H3 EasyCache video storage: {storage}')
    return (cache, None, transient_peak)

def _extract_av(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    video, audio = value
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        return None
    return (video, audio)

def _options_from_call(args, kwargs) -> dict[str, Any] | None:
    candidate = kwargs.get('transformer_options')
    if isinstance(candidate, dict) and STATE_KEY in candidate:
        return candidate
    for value in reversed(args):
        if isinstance(value, dict) and STATE_KEY in value:
            return value
    return None

def _sigma_value(options: dict[str, Any]) -> float | None:
    try:
        return float(torch.as_tensor(options.get('sigmas')).reshape(-1)[0].item())
    except (TypeError, ValueError, IndexError, RuntimeError):
        return None

@dataclass
class H3EasyCacheState:
    mode: str
    refresh_backend: str
    video_threshold: float
    video_frame_p95_threshold: float
    audio_threshold: float
    start_percent: float
    end_percent: float
    subsample_factor: int = 8
    video_storage: str = 'int8'
    int8_group_size: int = 128
    memory_limit_mib: int = 512
    quality_profile: str = PROFILE_QUALITY
    caches: dict[str, AVCacheEntry] = field(default_factory=dict)
    first_uuid: str | None = None
    previous_x_video: torch.Tensor | None = None
    previous_x_audio: torch.Tensor | None = None
    previous_output_video: torch.Tensor | None = None
    previous_output_audio: torch.Tensor | None = None
    previous_video_norm: torch.Tensor | None = None
    previous_video_frame_norm: torch.Tensor | None = None
    previous_audio_norm: torch.Tensor | None = None
    video_rate: torch.Tensor | None = None
    video_frame_rate: torch.Tensor | None = None
    audio_rate: torch.Tensor | None = None
    cumulative_video: torch.Tensor | None = None
    cumulative_video_frame: torch.Tensor | None = None
    cumulative_audio: torch.Tensor | None = None
    force_refresh_next: bool = False
    current_sigma_key: str | None = None
    current_step_decision: str | None = None
    current_step_had_candidate: bool = False
    schedule_ordinals: dict[str, int] = field(default_factory=dict)
    step_count: int | None = None
    effective_video_threshold: float = 0.0
    effective_video_frame_p95_threshold: float = 0.0
    effective_audio_threshold: float = 0.0
    step_threshold_scale: float = 1.0
    minimum_steps: int = 3
    max_consecutive_hit_steps: int = 1
    consecutive_hit_steps: int = 0
    step_policy_reason: str | None = None
    effective_window_policy: str | None = None
    effective_window_start_step: int | None = None
    effective_window_end_step: int | None = None
    disabled_reason: str | None = None

    def __post_init__(self):
        if self.mode != MODE_ACTIVE:
            raise ValueError('H3 EasyCache mode must be active')
        if self.video_storage not in ('int8', 'fp16'):
            raise ValueError('video_storage must be int8 or fp16')
        if self.quality_profile not in SUPPORTED_PROFILES:
            raise ValueError(f'quality_profile must be one of {SUPPORTED_PROFILES!r}')
        self._configure_step_policy()

    def clone_config(self) -> 'H3EasyCacheState':
        return H3EasyCacheState(mode=self.mode, refresh_backend=self.refresh_backend, video_threshold=self.video_threshold, video_frame_p95_threshold=self.video_frame_p95_threshold, audio_threshold=self.audio_threshold, start_percent=self.start_percent, end_percent=self.end_percent, subsample_factor=self.subsample_factor, video_storage=self.video_storage, int8_group_size=self.int8_group_size, memory_limit_mib=self.memory_limit_mib, quality_profile=self.quality_profile)

    def _configure_step_policy(self) -> None:
        """Resolve cache strength from the actual denoising schedule length.

        The measured thresholds came from 20-step schedules.  They are absolute
        error guards in quality mode and therefore must not be loosened merely
        to force a hit.  The explicit speed profile compensates for the wider
        interval between short-schedule steps, capped at 2.5x (20/8), and can
        reuse more consecutive steps only when the schedule is long enough to
        retain frequent exact refreshes.
        """
        steps = self.step_count
        self.effective_window_policy = 'nearest-sample' if self.quality_profile == PROFILE_SPEED else 'strict-inward'
        self.minimum_steps = 3
        self.step_threshold_scale = 1.0
        self.max_consecutive_hit_steps = 1
        self.step_policy_reason = 'quality-absolute-gates'
        if self.quality_profile == PROFILE_SPEED:
            if steps:
                self.step_threshold_scale = max(1.0, min(2.5, REFERENCE_STEPS / max(1, int(steps))))
                if steps >= 30:
                    self.max_consecutive_hit_steps = 3
                elif steps >= 13:
                    self.max_consecutive_hit_steps = 2
            self.step_policy_reason = 'speed-step-normalized-gates'
        elif self.quality_profile == PROFILE_MANUAL:
            self.step_policy_reason = 'manual-absolute-gates'
        self._resolve_step_window()
        self._refresh_effective_thresholds()

    def _resolve_step_window(self) -> None:
        """Map percentage boundaries onto the finite denoising schedule.

        Quality mode keeps the exact percentage contract by rounding the start
        inward with ceil and the end inward with floor. Speed mode maps each
        boundary to its nearest real sample. This matters for short schedules:
        with eight steps, 0.85 lies at ordinal 5.95, so an exact float compare
        discards the penultimate step (ordinal 6 / 7 = 0.857).
        First and last steps remain unconditionally protected.
        """
        steps = self.step_count
        self.effective_window_start_step = None
        self.effective_window_end_step = None
        if steps is None or steps < self.minimum_steps:
            return
        last = max(1, int(steps) - 1)
        if self.quality_profile == PROFILE_SPEED:
            start_ordinal = math.floor(float(self.start_percent) * last + 0.5)
            end_ordinal = math.floor(float(self.end_percent) * last + 0.5)
        else:
            start_ordinal = math.ceil(float(self.start_percent) * last)
            end_ordinal = math.floor(float(self.end_percent) * last)
        start_ordinal = max(1, start_ordinal)
        end_ordinal = min(int(steps) - 2, end_ordinal)
        if start_ordinal > end_ordinal:
            return
        self.effective_window_start_step = start_ordinal + 1
        self.effective_window_end_step = end_ordinal + 1

    def _refresh_effective_thresholds(self) -> None:
        """Keep direct/programmatic gate edits compatible with older callers."""
        scale = self.step_threshold_scale
        self.effective_video_threshold = min(1.0, float(self.video_threshold) * scale)
        self.effective_video_frame_p95_threshold = min(4.0, float(self.video_frame_p95_threshold) * scale)
        self.effective_audio_threshold = min(1.0, float(self.audio_threshold) * scale)

    def prepare_run(self, sigmas=None) -> None:
        self.clear_runtime()
        from .sampling_schedule import sample_sigma_boundaries
        values = sample_sigma_boundaries(sigmas)
        self.step_count = len(values) - 1 if values else None
        for ordinal, sigma in enumerate(values[:-1]):
            self.schedule_ordinals[f'{sigma:.12g}'] = ordinal
        self._configure_step_policy()

    def clear_runtime(self) -> None:
        self._clear_prediction_tensors()
        self.first_uuid = None
        self.force_refresh_next = False
        self.current_sigma_key = None
        self.current_step_decision = None
        self.current_step_had_candidate = False
        self.schedule_ordinals.clear()
        self.step_count = None
        self.effective_window_policy = None
        self.effective_window_start_step = None
        self.effective_window_end_step = None
        self.consecutive_hit_steps = 0
        self.disabled_reason = None

    def _clear_prediction_tensors(self):
        self.caches.clear()
        for name in HISTORY_TENSOR_FIELDS:
            setattr(self, name, None)

    def _drop_uuid_caches(self, uuids):
        dropped = 0
        for key in uuids:
            entry = self.caches.pop(key, None)
            if entry is not None:
                dropped += entry.storage_bytes
        return dropped

    def _ordinal(self, sigma: float | None) -> int | None:
        if sigma is None:
            return None
        return self.schedule_ordinals.get(f'{float(sigma):.12g}')

    def _inside_window(self, ordinal: int | None) -> bool:
        if ordinal is None or self.step_count is None or self.step_count < self.minimum_steps or (self.effective_window_start_step is None) or (self.effective_window_end_step is None):
            return False
        return 0 < ordinal < self.step_count - 1 and self.effective_window_start_step <= ordinal + 1 <= self.effective_window_end_step

    @staticmethod
    def _slice_uuid(tensor: torch.Tensor, uuids: tuple[str, ...], index: int) -> torch.Tensor:
        batch = int(tensor.shape[0])
        if not uuids or batch % len(uuids):
            raise ValueError('H3 EasyCache batch does not divide by UUID count')
        width = batch // len(uuids)
        return tensor[index * width:(index + 1) * width]

    def _subsample_video(self, tensor: torch.Tensor, uuids: tuple[str, ...]) -> torch.Tensor:
        index = uuids.index(self.first_uuid)
        value = self._slice_uuid(tensor, uuids, index)
        factor = max(1, int(self.subsample_factor))
        return value[..., ::factor, ::factor].detach().clone()

    def _subsample_audio(self, tensor: torch.Tensor, uuids: tuple[str, ...]) -> torch.Tensor:
        index = uuids.index(self.first_uuid)
        value = self._slice_uuid(tensor, uuids, index)
        factor = max(1, int(self.subsample_factor))
        return value[..., ::factor].detach().clone()

    def _can_apply(self, uuids: tuple[str, ...]) -> bool:
        return bool(uuids) and all((value in self.caches for value in uuids))

    def _begin_sigma(self, sigma: float | None) -> None:
        sigma_key = None if sigma is None else f'{float(sigma):.12g}'
        if sigma_key == self.current_sigma_key:
            return
        if self.current_step_had_candidate:
            self.force_refresh_next = self.consecutive_hit_steps >= self.max_consecutive_hit_steps
        self.current_sigma_key = sigma_key
        self.current_step_decision = None
        self.current_step_had_candidate = False

    def _apply(self, video: torch.Tensor, audio: torch.Tensor, uuids: tuple[str, ...], options=None):
        video_out = video.clone()
        audio_out = audio.clone()
        decode_peak = 0
        for index, key in enumerate(uuids):
            video_slice = self._slice_uuid(video_out, uuids, index)
            audio_slice = self._slice_uuid(audio_out, uuids, index)
            entry = self.caches[key]
            entry.video.add_to_(video_slice)
            audio_slice.add_(entry.audio_fp32.to(device=audio_slice.device, dtype=audio_slice.dtype))
            video_chunk = min(int(video_slice.numel()), CACHE_CHUNK_ELEMENTS) * 4
            audio_chunk = int(audio_slice.numel()) * max(0, int(audio_slice.element_size()))
            decode_peak = max(decode_peak, video_chunk, audio_chunk)
        return [video_out, audio_out]

    def _cache_outputs(self, video_output: torch.Tensor, audio_output: torch.Tensor, video_input: torch.Tensor, audio_input: torch.Tensor, uuids: tuple[str, ...]) -> None:
        if self.disabled_reason is not None:
            return
        old_bytes = sum((value.storage_bytes for value in self.caches.values()))
        replacements = dict(((key, value.storage_bytes) for key, value in self.caches.items()))
        added_bytes = 0
        estimated_peak = old_bytes
        for index, key in enumerate(uuids):
            elements = self._slice_uuid(video_output, uuids, index).numel()
            audio_bytes = self._slice_uuid(audio_output, uuids, index).numel() * 4
            if self.video_storage == 'int8':
                groups = math.ceil(elements / self.int8_group_size)
                payload = groups * (self.int8_group_size + 4)
                chunk_groups = min(groups, max(1, CACHE_CHUNK_ELEMENTS // self.int8_group_size))
                transient = chunk_groups * (self.int8_group_size * 20 + 12)
            else:
                payload = elements * 2
                transient = min(elements, CACHE_CHUNK_ELEMENTS) * 18
            for tensor in (video_output, video_input):
                part = self._slice_uuid(tensor, uuids, index)
                if not part.is_contiguous():
                    transient += part.numel() * part.element_size()
            added_bytes += payload + audio_bytes
            replacements[key] = payload + audio_bytes
            estimated_peak = max(estimated_peak, old_bytes + added_bytes + transient)
        limit = self.memory_limit_mib * 1024 * 1024
        reason = 'cache_memory_limit' if sum(replacements.values()) > limit else 'cache_update_memory_limit' if estimated_peak > limit else None
        if reason is not None:
            self._disable_cache(reason)
            return
        updated = {}
        old_bytes = sum((value.storage_bytes for value in self.caches.values()))
        update_transient_peak = 0
        for index, key in enumerate(uuids):
            video_out = self._slice_uuid(video_output, uuids, index)
            video_in = self._slice_uuid(video_input, uuids, index)
            audio_out = self._slice_uuid(audio_output, uuids, index)
            audio_in = self._slice_uuid(audio_input, uuids, index)
            video_cache, quant_error, transient_bytes = _encode_video_difference(video_out, video_in, self.video_storage, self.int8_group_size)
            audio_diff = torch.empty(tuple(audio_out.shape), dtype=torch.float32, device=audio_out.device)
            torch.sub(audio_out.detach(), audio_in.detach(), out=audio_diff)
            updated[key] = AVCacheEntry(video_cache, audio_diff)
            update_transient_peak = max(update_transient_peak, old_bytes + sum((value.storage_bytes for value in updated.values())) + int(transient_bytes))
        prospective = dict(self.caches)
        prospective.update(updated)
        steady_bytes = sum((value.storage_bytes for value in prospective.values()))
        memory_limit_bytes = self.memory_limit_mib * 1024 * 1024
        if steady_bytes > memory_limit_bytes:
            self.disabled_reason = 'cache_memory_limit'
            self.caches.clear()
            return
        if update_transient_peak > memory_limit_bytes:
            self.disabled_reason = 'cache_update_memory_limit'
            self.caches.clear()
            return
        self.caches = prospective
        return (steady_bytes, update_transient_peak)

    def execute(self, executor, args, kwargs, options):
        if self.disabled_reason is not None:
            return self._execute_uncached(executor, args, kwargs, options)
        completed = []
        model_failed = False

        def exact(*call_args, **call_kwargs):
            nonlocal model_failed
            try:
                result = executor(*call_args, **call_kwargs)
            except BaseException:
                model_failed = True
                raise
            completed.append(result)
            return result
        try:
            return self._execute_with_cache(exact, args, kwargs, options)
        except (torch.OutOfMemoryError, MemoryError):
            if model_failed:
                raise
        self._disable_cache('cache_resource_failure')
        if completed:
            return completed[0]
        return self._execute_uncached(executor, args, kwargs, options)

    def _disable_cache(self, reason):
        self._clear_prediction_tensors()
        self.disabled_reason = reason
        self.current_step_decision = 'refresh'
        self.force_refresh_next = True

    def _execute_uncached(self, executor, args, kwargs, options):
        result = executor(*args, **kwargs)
        return result

    def _execute_with_cache(self, executor, args, kwargs, options):
        self._refresh_effective_thresholds()
        inputs = _extract_av(args[0] if args else kwargs.get('x'))
        if inputs is None:
            return executor(*args, **kwargs)
        if options.get('easycache') is not None or options.get('patches_replace', {}).get('dit', {}):
            self.disabled_reason = 'external_cache_or_block_patch'
            return executor(*args, **kwargs)
        video_input, audio_input = inputs
        raw_uuids = options.get('uuids')
        if not isinstance(raw_uuids, (list, tuple)) or not raw_uuids:
            self.disabled_reason = 'missing_uuids'
            return executor(*args, **kwargs)
        uuids = tuple((str(value) for value in raw_uuids))
        if self.first_uuid is None:
            self.first_uuid = uuids[0]
        sigma = _sigma_value(options)
        self._begin_sigma(sigma)
        ordinal = self._ordinal(sigma)
        if ordinal is not None and ordinal == self.step_count - 1:
            self._clear_prediction_tensors()
            self.current_step_decision = 'refresh'
            result = executor(*args, **kwargs)
            outputs = _extract_av(result)
            if outputs is None or outputs[0].shape != video_input.shape or outputs[1].shape != audio_input.shape:
                self.disabled_reason = 'unsupported_output_layout'
                return result
            return result
        is_lead_call = self.first_uuid in uuids
        input_video_sample = self._subsample_video(video_input, uuids) if is_lead_call else None
        input_audio_sample = self._subsample_audio(audio_input, uuids) if is_lead_call else None
        input_video_change = input_audio_change = None
        input_video_frame_change = None
        predicted_video_change = predicted_audio_change = None
        predicted_video_frame_change = None
        predicted_video_frame_p95 = None
        if input_video_sample is not None and self.previous_x_video is not None:
            input_video_change = (input_video_sample - self.previous_x_video).abs().mean()
            input_video_frame_change = _video_per_frame_abs_mean(input_video_sample - self.previous_x_video)
        if input_audio_sample is not None and self.previous_x_audio is not None:
            input_audio_change = (input_audio_sample - self.previous_x_audio).abs().mean()
        if input_video_change is not None and input_audio_change is not None and (self.video_rate is not None) and (self.audio_rate is not None) and (self.previous_video_norm is not None) and (self.previous_audio_norm is not None):
            predicted_video_change = self.video_rate * input_video_change / self.previous_video_norm
            predicted_audio_change = self.audio_rate * input_audio_change / self.previous_audio_norm
            if self.cumulative_video is None:
                self.cumulative_video = torch.zeros_like(predicted_video_change)
                self.cumulative_audio = torch.zeros_like(predicted_audio_change)
            self.cumulative_video = self.cumulative_video + predicted_video_change
            self.cumulative_audio = self.cumulative_audio + predicted_audio_change
        if input_video_frame_change is not None and self.video_frame_rate is not None and (self.previous_video_frame_norm is not None):
            predicted_video_frame_change = self.video_frame_rate * input_video_frame_change / self.previous_video_frame_norm
            if self.cumulative_video_frame is None:
                self.cumulative_video_frame = torch.zeros_like(predicted_video_frame_change)
            self.cumulative_video_frame = self.cumulative_video_frame + predicted_video_frame_change
            predicted_video_frame_p95 = torch.quantile(self.cumulative_video_frame.float(), 0.95)
        if self.current_step_decision == 'candidate':
            can_hit = self._can_apply(uuids)
        elif self.current_step_decision == 'refresh':
            can_hit = False
        else:
            can_hit = is_lead_call and self.disabled_reason is None and (not self.force_refresh_next) and self._inside_window(ordinal) and self._can_apply(uuids) and (self.cumulative_video is not None) and (self.cumulative_audio is not None) and bool((self.cumulative_video < self.effective_video_threshold).item()) and bool((self.cumulative_audio < self.effective_audio_threshold).item()) and (self.effective_video_frame_p95_threshold <= 0.0 or (predicted_video_frame_p95 is not None and bool((predicted_video_frame_p95 < self.effective_video_frame_p95_threshold).item())))
            if is_lead_call:
                self.current_step_decision = 'candidate' if can_hit else 'refresh'
        predicted = self._apply(video_input, audio_input, uuids, options) if can_hit else None
        if can_hit and self.mode == MODE_ACTIVE:
            self.current_step_had_candidate = True
            if is_lead_call:
                self.consecutive_hit_steps += 1
                self.force_refresh_next = self.consecutive_hit_steps >= self.max_consecutive_hit_steps
            return predicted
        if self.current_step_decision == 'refresh':
            dropped = self._drop_uuid_caches(uuids)
        result = executor(*args, **kwargs)
        outputs = _extract_av(result)
        if outputs is None or outputs[0].shape != video_input.shape or outputs[1].shape != audio_input.shape:
            self.disabled_reason = 'unsupported_output_layout'
            return result
        video_output, audio_output = outputs
        if is_lead_call and input_video_change is not None and (self.previous_output_video is not None):
            output_video_sample = self._subsample_video(video_output, uuids)
            output_video_change = (output_video_sample - self.previous_output_video).abs().mean()
            if float(input_video_change.item()) > 1e-12:
                self.video_rate = output_video_change / input_video_change
            if input_video_frame_change is not None:
                output_video_frame_change = _video_per_frame_abs_mean(output_video_sample - self.previous_output_video)
                fallback_rate = self.video_rate if self.video_rate is not None else torch.ones((), device=output_video_frame_change.device)
                self.video_frame_rate = torch.where(input_video_frame_change > 1e-12, output_video_frame_change / input_video_frame_change.clamp_min(1e-12), fallback_rate.expand_as(output_video_frame_change))
        elif is_lead_call:
            output_video_sample = self._subsample_video(video_output, uuids)
        else:
            output_video_sample = None
        if is_lead_call and input_audio_change is not None and (self.previous_output_audio is not None):
            output_audio_sample = self._subsample_audio(audio_output, uuids)
            output_audio_change = (output_audio_sample - self.previous_output_audio).abs().mean()
            if float(input_audio_change.item()) > 1e-12:
                self.audio_rate = output_audio_change / input_audio_change
        elif is_lead_call:
            output_audio_sample = self._subsample_audio(audio_output, uuids)
        else:
            output_audio_sample = None
        cache_result = self._cache_outputs(video_output, audio_output, video_input, audio_input, uuids)
        if self.disabled_reason is not None:
            self._disable_cache(self.disabled_reason)
            return result
        if cache_result is not None:
            steady_bytes, update_peak_bytes = cache_result
        if is_lead_call:
            self.previous_x_video = input_video_sample
            self.previous_x_audio = input_audio_sample
            self.previous_output_video = output_video_sample
            self.previous_output_audio = output_audio_sample
            self.previous_video_norm = output_video_sample.abs().mean().clamp_min(1e-08)
            self.previous_video_frame_norm = _video_per_frame_abs_mean(output_video_sample).clamp_min(1e-08)
            self.previous_audio_norm = output_audio_sample.abs().mean().clamp_min(1e-08)
            self.cumulative_video = None
            self.cumulative_video_frame = None
            self.cumulative_audio = None
            self.force_refresh_next = False
            self.consecutive_hit_steps = 0
        return result

def h3_easycache_diffusion_wrapper(executor, *args, **kwargs):
    options = _options_from_call(args, kwargs)
    state = None if options is None else options.get(STATE_KEY)
    if not isinstance(state, H3EasyCacheState):
        return executor(*args, **kwargs)
    return state.execute(executor, args, kwargs, options)

def h3_easycache_outer_sample_wrapper(executor, *args, **kwargs):
    import comfy.model_patcher
    guider = getattr(executor, 'class_obj', None)
    original_options = getattr(guider, 'model_options', None)
    if not isinstance(original_options, dict):
        return executor(*args, **kwargs)
    configured = original_options.get('transformer_options', {}).get(STATE_KEY)
    if not isinstance(configured, H3EasyCacheState):
        return executor(*args, **kwargs)
    guider.model_options = comfy.model_patcher.create_model_options_clone(original_options)
    state = configured.clone_config()
    guider.model_options.setdefault('transformer_options', {})[STATE_KEY] = state
    sigmas = kwargs.get('sigmas')
    if sigmas is None and len(args) > 3:
        sigmas = args[3]
    state.prepare_run(sigmas)
    try:
        result = executor(*args, **kwargs)
        return result
    except BaseException as error:
        raise
    finally:
        state.clear_runtime()
        guider.model_options = original_options

def patch_model_for_h3_easycache(model, *, mode=MODE_ACTIVE, refresh_backend='flash_attn', video_threshold=0.05, video_frame_p95_threshold=0.0, audio_threshold=0.05, start_percent=0.15, end_percent=0.85, video_storage='int8', int8_group_size=128, memory_limit_mib=512, quality_profile=PROFILE_QUALITY):
    import comfy.patcher_extension
    if mode != MODE_ACTIVE:
        raise ValueError('H3 EasyCache mode must be active')
    if quality_profile not in SUPPORTED_PROFILES:
        raise ValueError(f'quality_profile must be one of {SUPPORTED_PROFILES!r}')
    if not 0 <= float(start_percent) < float(end_percent) <= 1:
        raise ValueError('H3 EasyCache requires 0 <= start_percent < end_percent <= 1')
    patched = model.clone()
    patched.model_options = dict(patched.model_options)
    options = dict(patched.model_options.get('transformer_options', {}))
    patched.model_options['transformer_options'] = options
    if STABLE_VBAR_KEY not in options:
        raise RuntimeError('H3 EasyCache must be configured by H3 V100 Optimize')
    if 'easycache' in options:
        raise RuntimeError('Remove the external EasyCache/LazyCache node before enabling H3 EasyCache')
    if options.get('patches_replace', {}).get('dit', {}):
        raise RuntimeError('H3 EasyCache cannot be combined with TE or block-cache replacements')
    state = H3EasyCacheState(mode=str(mode), refresh_backend=str(refresh_backend), video_threshold=max(0.0, float(video_threshold)), video_frame_p95_threshold=max(0.0, float(video_frame_p95_threshold)), audio_threshold=max(0.0, float(audio_threshold)), start_percent=float(start_percent), end_percent=float(end_percent), video_storage=str(video_storage), int8_group_size=max(16, int(int8_group_size)), memory_limit_mib=max(64, int(memory_limit_mib)), quality_profile=str(quality_profile))
    options[STATE_KEY] = state
    options[ACTIVE_KEY] = True
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, 'v100_h3_easycache', h3_easycache_diffusion_wrapper)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, 'v100_h3_easycache', h3_easycache_outer_sample_wrapper)
    return (patched, state)
