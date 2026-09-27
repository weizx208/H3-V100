"""Bounded-output exact Flash stream for long MiniMax H3 sequences."""
from __future__ import annotations
import logging
from functools import partial
import torch
from .sol_native import load_sol_ops
from .sol_calibration import _full_audio_query_block_ranges
LOGGER = logging.getLogger('H3V100ExactFlashRange')
DEFAULT_CHUNK_TOKENS = 2048
MIN_RETRY_CHUNK_TOKENS = 64

def _is_memory_error(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return any((marker in message for marker in ('out of memory', 'memory allocation', 'cudamalloc', 'vbar_fault', 'vbar fault', 'result 2')))

class ExactFlashRangeStream:
    """Yield exact dense Flash attention without a full output allocation."""
    _h3_v100_sol_range_stream = True
    _h3_v100_exact_flash_range_stream = True

    def __init__(self, q, k, v, *, scale=None, chunk_tokens=DEFAULT_CHUNK_TOKENS, minimum_chunk_tokens=MIN_RETRY_CHUNK_TOKENS, audio_ranges=(), audio_overwrite_active=False):
        self.q = q
        self.k = k
        self.v = v
        self.scale = float(scale if scale is not None else q.shape[-1] ** (-0.5))
        self.tokens = int(q.shape[2])
        self.chunk_tokens = max(64, int(chunk_tokens) // 64 * 64)
        self.minimum_chunk_tokens = max(64, int(minimum_chunk_tokens) // 64 * 64)
        self.audio_block_ranges = _full_audio_query_block_ranges(audio_ranges, self.tokens, 64) if audio_overwrite_active else ()
        self._closed = False
        self._fallback = None
        self._runtime_failure = None
        self._timing_callback = None
        self._timing_events = []
        self._timing_finished = False

    def attach_fallback(self, fallback, runtime_failure=None):
        self._fallback = fallback
        self._runtime_failure = runtime_failure
        return self

    def attach_kernel_timing(self, callback):
        """Report summed dense-attention CUDA time after stream consumption.

        Per-range events exclude the output-projection kernels that run between
        generator yields, so Sol admission compares attention with attention
        without materializing a complete exact result.
        """
        self._timing_callback = callback
        return self

    def _finish_timing(self, *, completed, error=None):
        if self._timing_finished or self._timing_callback is None:
            return
        self._timing_finished = True
        timing_ms = None
        timing_error = error
        if completed and self._timing_events:
            try:
                self._timing_events[-1][1].synchronize()
                timing_ms = sum((float(start.elapsed_time(end)) for start, end in self._timing_events))
            except BaseException as event_error:
                timing_error = event_error
        elif completed:
            timing_ms = 0.0
        try:
            self._timing_callback(timing_ms, timing_error)
        finally:
            self._timing_callback = None
            self._timing_events.clear()
            timing_error = None

    def exact_fallback(self, error=None):
        self.close(completed=False, error=error)
        if self._runtime_failure is not None:
            self._runtime_failure(error)
        if self._fallback is None:
            raise RuntimeError('exact Flash range stream exhausted its fallback')
        return self._fallback()

    def __iter__(self):
        if self._closed:
            raise RuntimeError('exact Flash range stream was already consumed')
        ops = load_sol_ops(require_dense=True)
        start = 0
        active_chunk = self.chunk_tokens
        skip_index = 0
        failure = None
        try:
            while start < self.tokens:
                while skip_index < len(self.audio_block_ranges) and self.audio_block_ranges[skip_index][1] * 64 <= start:
                    skip_index += 1
                skip_range = self.audio_block_ranges[skip_index] if skip_index < len(self.audio_block_ranges) else None
                skip_start = skip_range[0] * 64 if skip_range else self.tokens
                skip_stop = min(skip_range[1] * 64, self.tokens) if skip_range else self.tokens
                is_audio = bool(skip_range is not None and skip_start <= start < skip_stop)
                if is_audio:
                    count = min(active_chunk, skip_stop - start)
                    batch, heads, _tokens, width = self.q.shape
                    out = torch.empty_strided((batch, heads, count, width), (heads * count * width, width, heads * width, 1), dtype=self.q.dtype, device=self.q.device).zero_()
                    lse = torch.zeros((batch, heads, count), dtype=torch.float32, device=self.q.device)
                    yield (start, out, lse)
                    start += count
                    continue
                count = min(active_chunk, skip_start - start, self.tokens - start)
                try:
                    timing_start = timing_end = None
                    if self._timing_callback is not None:
                        timing_start = torch.cuda.Event(enable_timing=True)
                        timing_end = torch.cuda.Event(enable_timing=True)
                        timing_start.record()
                    out, lse = ops.sol_dense_rect(self.q, self.k, self.v, start, count, self.tokens, self.scale)
                    if timing_end is not None:
                        timing_end.record()
                        self._timing_events.append((timing_start, timing_end))
                except Exception as error:
                    if not _is_memory_error(error) or active_chunk <= self.minimum_chunk_tokens:
                        raise
                    active_chunk = max(self.minimum_chunk_tokens, active_chunk // 2 // 64 * 64)
                    LOGGER.warning('H3 V100 exact Flash range reduced chunk after memory pressure: tokens=%d, start=%d, chunk_tokens=%d.', self.tokens, start, active_chunk)
                    torch.cuda.empty_cache()
                    continue
                yield (start, out, lse)
                start += count
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                self.close(completed=start >= self.tokens, error=failure)
            finally:
                failure = None

    def close(self, *, completed=False, error=None):
        if self._closed:
            return
        self._closed = True
        try:
            self._finish_timing(completed=bool(completed), error=error)
        finally:
            self.q = self.k = self.v = None

def make_exact_flash_range_stream(q, k, v, *, scale=None, chunk_tokens=DEFAULT_CHUNK_TOKENS, runtime_failure=None, audio_ranges=(), audio_overwrite_active=False):
    """Build an exact stream with bounded smaller-chunk projection retries."""
    active_chunk = max(64, int(chunk_tokens) // 64 * 64)
    stream = ExactFlashRangeStream(q, k, v, scale=scale, chunk_tokens=active_chunk, audio_ranges=audio_ranges, audio_overwrite_active=audio_overwrite_active)
    if active_chunk > MIN_RETRY_CHUNK_TOKENS:
        next_chunk = max(MIN_RETRY_CHUNK_TOKENS, active_chunk // 2 // 64 * 64)
        stream.attach_fallback(partial(make_exact_flash_range_stream, q, k, v, scale=scale, chunk_tokens=next_chunk, runtime_failure=runtime_failure, audio_ranges=audio_ranges, audio_overwrite_active=audio_overwrite_active), runtime_failure=runtime_failure)
    elif runtime_failure is not None:
        stream._runtime_failure = runtime_failure
    return stream
