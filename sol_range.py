"""Bounded SM70 corrected SOL attention and exact audio ranges."""
from __future__ import annotations
import logging
import math
import torch
from .sol_native import _build_route, load_sol_ops
from .sol_calibration import _full_audio_query_block_ranges
LOGGER = logging.getLogger('H3V100FusedSolActive')
STATE_KEY = 'v100_h3_fused_sol_active_state'
STABLE_VBAR_KEY = 'v100_h3_dynamic_vbar_controller'
PROTECTED_LAYERS = {0, 1, 40, 49}
CORRECTED_STREAM_CHUNK_TOKENS = 2048

class CorrectedSolRangeStream:
    """Yield centroid-corrected attention in bounded query ranges."""
    _h3_v100_sol_range_stream = True
    _h3_v100_corrected_sol_range_stream = True

    def __init__(self, q, k, v, k_centroids, v_centroids, row_ptr, offsets, route_bitmask, softmax_scale, prefix_blocks, chunk_tokens=CORRECTED_STREAM_CHUNK_TOKENS, audio_ranges=(), audio_overwrite_active=False):
        self.q = q
        self.k = k
        self.v = v
        self.k_centroids = k_centroids
        self.v_centroids = v_centroids
        self.row_ptr = row_ptr
        self.offsets = offsets
        self.route_bitmask = route_bitmask
        self.softmax_scale = float(softmax_scale)
        self.prefix_blocks = int(prefix_blocks)
        self.chunk_tokens = max(64, int(chunk_tokens) // 64 * 64)
        self.tokens = int(q.shape[2])
        self.blocks = math.ceil(self.tokens / 64)
        self.audio_block_ranges = _full_audio_query_block_ranges(audio_ranges, self.tokens, 64) if audio_overwrite_active else ()
        self.skipped_audio_query_blocks = sum((stop - start for start, stop in self.audio_block_ranges))
        self._closed = False
        self._exact_fallback = None
        self._runtime_failure = None

    def attach_fallback(self, exact_fallback, runtime_failure=None):
        self._exact_fallback = exact_fallback
        self._runtime_failure = runtime_failure
        return self

    def exact_fallback(self, error=None):
        self.close()
        if self._runtime_failure is not None:
            self._runtime_failure(error)
        if self._exact_fallback is None:
            if isinstance(error, BaseException):
                raise RuntimeError('corrected Sol range stream has no exact fallback') from error
            raise RuntimeError('corrected Sol range stream has no exact fallback')
        return self._exact_fallback()

    def __iter__(self):
        if self._closed:
            raise RuntimeError('corrected Sol range stream was already consumed or closed')
        ops = load_sol_ops(require_range=True)
        chunk_blocks = max(1, self.chunk_tokens // 64)
        skip_index = 0
        query_block_start = 0
        try:
            while query_block_start < self.blocks:
                while skip_index < len(self.audio_block_ranges) and self.audio_block_ranges[skip_index][1] <= query_block_start:
                    skip_index += 1
                skip_range = self.audio_block_ranges[skip_index] if skip_index < len(self.audio_block_ranges) else None
                is_audio = bool(skip_range is not None and skip_range[0] <= query_block_start < skip_range[1])
                if is_audio:
                    query_block_stop = min(query_block_start + chunk_blocks, skip_range[1])
                else:
                    next_audio = skip_range[0] if skip_range is not None else self.blocks
                    query_block_stop = min(query_block_start + chunk_blocks, next_audio, self.blocks)
                query_block_count = query_block_stop - query_block_start
                start_token = query_block_start * 64
                if is_audio:
                    query_tokens = min(self.tokens, query_block_stop * 64) - start_token
                    batch, heads, _tokens, width = self.q.shape
                    out = torch.empty_strided((batch, heads, query_tokens, width), (heads * query_tokens * width, width, heads * width, 1), dtype=self.q.dtype, device=self.q.device).zero_()
                    lse = torch.zeros((batch, heads, query_tokens), dtype=torch.float32, device=self.q.device)
                else:
                    out, lse = ops.sol_sparse_corrected_csr_fused_range(self.q, self.k, self.v, self.k_centroids, self.v_centroids, self.row_ptr, self.offsets, self.route_bitmask, self.softmax_scale, self.prefix_blocks, query_block_start, query_block_count)
                yield (start_token, out, lse)
                query_block_start = query_block_stop
        finally:
            self.close()

    def consume_probe(self):
        finite = torch.ones((), device=self.q.device, dtype=torch.bool)
        for _start, out, lse in self:
            finite.logical_and_(torch.isfinite(out).all())
            finite.logical_and_(torch.isfinite(lse).all())
            del out, lse
        return torch.where(finite, torch.ones((), device=finite.device, dtype=torch.float32), torch.full((), float('nan'), device=finite.device))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.q = self.k = self.v = None
        self.k_centroids = self.v_centroids = None
        self.row_ptr = self.offsets = self.route_bitmask = None

def run_fused_sol_active(q, k, v, *, tau, prefix_stop, scale, stream_output=False, stream_chunk_tokens=CORRECTED_STREAM_CHUNK_TOKENS, memory_limit_mib=1024, audio_ranges=(), audio_overwrite_active=False, route_builder=None, route_context=None):
    ops = load_sol_ops(require_range=bool(stream_output))
    softmax_scale = float(scale if scale is not None else 128 ** (-0.5))
    route_kwargs = {'tau': float(tau), 'prefix_stop': int(prefix_stop), 'memory_limit_mib': int(memory_limit_mib)}
    if callable(route_builder):
        row_ptr, offsets, bitmask, kc16, vc16, density = route_builder(ops, q, k, v, route_context=route_context, **route_kwargs)
    else:
        row_ptr, offsets, bitmask, kc16, vc16, density = _build_route(ops, q, k, v, **route_kwargs)
    prefix_blocks = min(math.ceil(int(q.shape[2]) / 64), math.ceil(max(0, int(prefix_stop)) / 64))
    if stream_output:
        return (CorrectedSolRangeStream(q, k, v, kc16, vc16, row_ptr, offsets, bitmask, softmax_scale, prefix_blocks, chunk_tokens=stream_chunk_tokens, audio_ranges=audio_ranges, audio_overwrite_active=audio_overwrite_active), float(density))
    output, lse = ops.sol_sparse_corrected_csr_fused(q, k, v, kc16, vc16, row_ptr, offsets, bitmask, softmax_scale, prefix_blocks)
    del lse, row_ptr, offsets, bitmask, kc16, vc16
    batch, heads, tokens, width = output.shape
    output = output.transpose(1, 2).reshape(batch, tokens, heads * width)
    return (output, float(density))
