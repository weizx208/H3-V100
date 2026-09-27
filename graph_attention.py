"""Graph-compatible dual attention: main-thread CUDA0 and worker-owned CUDA1.

No CUDA1 tensor crosses into the primary allocation-graph owner. Transfers
remain bounded pinned-host staging; exact/SOL/audio mathematics are unchanged.
"""
from contextlib import ExitStack, nullcontext
import torch
from .dual_attention import (
    _local_qkv, _local_qkv_chunk, _local_audio, _local_range,
    _local_sol_stream, _local_sol_next,
    _exact_audio_skip_ranges, qkv_head_rows, out_head_columns,
)


def _secondary_worker(function, *args):
    """Do not export a failing worker's device-tensor traceback to the owner."""
    failure = None
    try:
        with torch.inference_mode():
            return function(*args)
    except Exception as error:
        try:
            failure = type(error)(str(error))
        except Exception:
            failure = RuntimeError(str(error))
    # Outside the except block: the original traceback and its cuda:1 locals
    # have been released on the worker. Retain the error class/message only.
    raise failure


def release_cached_secondary(device):
    """Keep secondary allocator cleanup on its worker even after a retry."""
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

class _SecondaryActor:
    """Own every cuda:1 exact-attention tensor on one non-graph worker thread."""

    def __init__(self, device, attention, pool, heads, key_chunk):
        self.device = device
        self.attention = attention
        self.pool = pool
        self.heads = int(heads)
        self.key_chunk = int(key_chunk)
        self.items = {}
        self.qkv = None
        self.audio = None
        self.sol_stream = None
        self.sol_iterator = None

    def _device(self):
        torch.cuda.set_device(self.device)

    def put(self, name, host, shape, dtype):
        self._device()
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        tensor.copy_(host, non_blocking=True)
        torch.cuda.synchronize(self.device)
        self.items[name] = tensor

    def allocate(self, name, shape, dtype):
        self._device()
        self.items[name] = torch.empty(shape, dtype=dtype, device=self.device)

    def adopt_qkv(self, workspace):
        self._device()
        if workspace.qkv is None:
            raise RuntimeError('secondary QKV workspace was released before use')
        self.qkv = workspace.qkv

    def copy_slice(self, name, index, host):
        self._device()
        self.items[name][index].copy_(host, non_blocking=True)
        torch.cuda.synchronize(self.device)

    def compute_qkv(self):
        self._device()
        self.qkv = _local_qkv(
            self.attention, self.items['x'], self.items['qkv_weight'],
            self.items['q_norm'], self.items['k_norm'], self.items.get('rope'),
            self.heads, self.key_chunk, already_scaled=True, outputs=self.qkv,
        )
        for name in ('x', 'qkv_weight', 'q_norm', 'k_norm', 'rope'):
            self.items.pop(name, None)

    def compute_audio(self, audio_ranges, head_dim):
        self._device()
        self.audio = _local_audio(self.qkv, audio_ranges, self.heads, head_dim, self.key_chunk)

    def allocate_sol_qkv(self, tokens, head_dim, hidden):
        self._device()
        if self.qkv is None:
            self.qkv = tuple(torch.empty((1, self.heads, tokens, head_dim),
                                        dtype=torch.float16, device=self.device) for _ in range(3))
        self.items['x'] = torch.empty((min(tokens, self.key_chunk), hidden),
                                     dtype=torch.float16, device=self.device)

    def compute_sol_qkv_chunk(self, host, start, stop, total_heads, first_head):
        self._device()
        chunk_input = self.items['x'][:stop - start]
        chunk_input.copy_(host, non_blocking=True)
        rope = self.items.get('rope')
        chunk = _local_qkv_chunk(
            self.attention, chunk_input, self.items['qkv_weight'],
            self.items['q_norm'], self.items['k_norm'],
            None if rope is None else rope[:, start:stop], total_heads,
            already_scaled=True,
        )
        for destination, source in zip(self.qkv, chunk):
            destination[:, :, start:stop].copy_(source[:, first_head:])
        # Main may reuse the pinned buffer only after this transfer completes.
        torch.cuda.synchronize(self.device)

    def release_qkv_inputs(self):
        self._device()
        for name in ('x', 'qkv_weight', 'q_norm', 'k_norm', 'rope'):
            self.items.pop(name, None)

    def start_sol(self, config):
        self._device()
        self.sol_stream, _ = _local_sol_stream(self.qkv, config)
        self.sol_iterator = iter(self.sol_stream)

    def next_sol(self):
        self._device()
        start, part = _local_sol_next(self.sol_iterator, self.items['out_weight'], self.audio)
        count = int(part.shape[0])
        if count > self.pool.partial.shape[0]:
            raise RuntimeError('secondary Sol query range exceeds admitted pool')
        self.pool.partial[:count].copy_(part, non_blocking=True)
        torch.cuda.synchronize(self.device)
        return start, count

    def compute_range(self, ops, start, stop, tokens, scale, audio_skip_ranges):
        self._device()
        part = _local_range(
            ops, self.qkv, self.items['out_weight'], start, stop, tokens,
            scale, self.audio, audio_skip_ranges=audio_skip_ranges,
        )
        self.pool.partial[:stop - start].copy_(part, non_blocking=True)
        torch.cuda.synchronize(self.device)
        del part

    def reset(self):
        try:
            self._device()
            if self.sol_stream is not None:
                self.sol_stream.close()
        finally:
            self.sol_iterator = self.sol_stream = None
            self.qkv = None
            self.audio = None
            self.items.clear()
            torch.cuda.synchronize(self.device)


def run(state, attention, target, rope_freqs, transformer_options,
        primary, secondary, primary_heads, secondary_heads,
        key_chunk, pool, audio_ranges, values, transaction, sol_config=None):
    """Keep primary allocations in their graph scope and secondary tensors worker-owned."""
    from .h3_mixed_precision import _projection_weight_context
    from .sol_native import load_sol_ops
    import comfy.model_management as model_management

    heads = int(attention.heads)
    head_dim = int(attention.head_dim)
    tokens = int(target.shape[0])
    hidden = int(target.shape[1])
    actor = _SecondaryActor(secondary, attention, pool, secondary_heads, key_chunk)
    def on_secondary(function, *args):
        return state._graph_executor.submit(_secondary_worker, function, *args).result()

    def stage(name, source):
        shape = tuple(int(dim) for dim in source.shape)
        host = pool.raw_view(source.dtype, shape)
        host.copy_(source, non_blocking=True)
        torch.cuda.synchronize(primary)
        on_secondary(actor.put, name, host, shape, source.dtype)

    pending = None
    primary_stream = None
    is_sol = isinstance(sol_config, dict)
    weight_owner = ExitStack()
    try:
        workspace = getattr(values.get('primary_qkv'), 'secondary', None)
        if workspace is not None:
            on_secondary(actor.adopt_qkv, workspace)

        ranges = ((0, primary_heads), (primary_heads, heads))
        projection_context = _projection_weight_context(attention.qkv_proj, primary, torch.float16, transformer_options)
        # SOL retains the prepared full-width primary weight until all token
        # chunks finish, matching the original cast/pin ownership contract.
        context = nullcontext(weight_owner.enter_context(projection_context)) if is_sol else projection_context
        with context as prepared:
            weight, bias = prepared
            if bias is not None:
                raise RuntimeError('dual H3 attention requires bias-free QKV')
            qkv0 = weight if is_sol else qkv_head_rows(weight, *ranges[0], heads, head_dim)
            source1 = weight if is_sol else qkv_head_rows(weight, *ranges[1], heads, head_dim)
            stage('qkv_weight', source1)
            del source1
        del prepared, weight, bias
        del context, projection_context

        with _projection_weight_context(attention.out_proj, primary, torch.float32, transformer_options) as prepared:
            weight, bias = prepared
            if bias is not None:
                raise RuntimeError('dual H3 attention requires bias-free output projection')
            out0 = out_head_columns(weight, *ranges[0], heads, head_dim)
            source1 = out_head_columns(weight, *ranges[1], heads, head_dim)
            stage('out_weight', source1)
            del source1
        del prepared, weight, bias

        q_weight0 = model_management.cast_to(attention.q_norm.weight, dtype=torch.float32, device=primary).contiguous()
        k_weight0 = model_management.cast_to(attention.k_norm.weight, dtype=torch.float32, device=primary).contiguous()
        stage('q_norm', q_weight0)
        stage('k_norm', k_weight0)


        if not is_sol:
            on_secondary(actor.allocate, 'x', (tokens, hidden), torch.float16)
            for start in range(0, tokens, pool.input_chunk):
                stop = min(tokens, start + pool.input_chunk)
                scaled = target[start:stop].mul(1.0 / 16.0).half()
                host = pool.input[:stop - start]
                host.copy_(scaled, non_blocking=True)
                torch.cuda.synchronize(primary)
                on_secondary(actor.copy_slice, 'x', slice(start, stop), host)
                del scaled

        if rope_freqs is not None:
            on_secondary(actor.allocate, 'rope', tuple(int(dim) for dim in rope_freqs.shape), rope_freqs.dtype)
            for start in range(0, int(rope_freqs.shape[1]), pool.input_chunk):
                stop = min(int(rope_freqs.shape[1]), start + pool.input_chunk)
                source = rope_freqs[:, start:stop].contiguous()
                host = pool.raw_view(source.dtype, tuple(int(dim) for dim in source.shape))
                host.copy_(source, non_blocking=True)
                torch.cuda.synchronize(primary)
                on_secondary(actor.copy_slice, 'rope', (slice(None), slice(start, stop)), host)
                del source


        if is_sol:
            primary_qkv = values.get('primary_qkv')
            if primary_qkv is None:
                primary_qkv = tuple(torch.empty((1, primary_heads, tokens, head_dim),
                                           dtype=torch.float16, device=primary) for _ in range(3))
            on_secondary(actor.allocate_sol_qkv, tokens, head_dim, hidden)
            for start in range(0, tokens, key_chunk):
                stop = min(tokens, start + key_chunk)
                scaled = target[start:stop].mul(1.0 / 16.0).half()
                host = pool.input[:stop - start]
                host.copy_(scaled, non_blocking=True)
                torch.cuda.synchronize(primary)
                pending = state._graph_executor.submit(
                    _secondary_worker, actor.compute_sol_qkv_chunk, host,
                    start, stop, heads, primary_heads,
                )
                chunk = _local_qkv_chunk(
                    attention, scaled, qkv0, q_weight0, k_weight0,
                    None if rope_freqs is None else rope_freqs[:, start:stop],
                    heads, already_scaled=True,
                )
                for destination, source in zip(primary_qkv, chunk):
                    destination[:, :, start:stop].copy_(source[:, :primary_heads])
                pending.result()
                del chunk, source, destination, scaled
            qkv0 = primary_qkv
            del primary_qkv
            on_secondary(actor.release_qkv_inputs)
            weight_owner.close()
        else:
            pending = state._graph_executor.submit(_secondary_worker, actor.compute_qkv)
            qkv0 = _local_qkv(attention, target, qkv0, q_weight0, k_weight0,
                              rope_freqs, primary_heads, key_chunk, already_scaled=False,
                              outputs=values.get('primary_qkv'))
            pending.result()
        del q_weight0, k_weight0

        pending = state._graph_executor.submit(_secondary_worker, actor.compute_audio, audio_ranges, head_dim)
        audio0 = _local_audio(qkv0, audio_ranges, primary_heads, head_dim, key_chunk)
        pending.result()

        ops = load_sol_ops(require_dense=True)
        audio_skip_ranges = _exact_audio_skip_ranges(audio0, tokens)
        scale = head_dim ** (-0.5)
        if is_sol:
            config = dict(sol_config, scale=scale)
            pending = state._graph_executor.submit(_secondary_worker, actor.start_sol, config)
            primary_stream, _ = _local_sol_stream(qkv0, config)
            primary_iterator = iter(primary_stream)
            pending.result()
            cursor = 0
            while cursor < tokens:
                pending = state._graph_executor.submit(_secondary_worker, actor.next_sol)
                start, primary_part = _local_sol_next(primary_iterator, out0, audio0)
                secondary_start, secondary_count = pending.result()
                count = int(primary_part.shape[0])
                if start != cursor or secondary_start != cursor or count != secondary_count:
                    raise RuntimeError('dual Sol head streams returned different query ranges')
                if count > pool.partial.shape[0] or cursor + count > tokens:
                    raise RuntimeError('dual Sol query range exceeds admitted pool')
                returned = values['returned'][:count]
                returned.copy_(pool.partial[:count], non_blocking=True)
                torch.cuda.synchronize(primary)
                primary_part.add_(returned).mul_(16.0)
                target[cursor:cursor + count].copy_(primary_part)
                transaction.mark_partial_target_write()
                del primary_part
                cursor += count
        for start in range(0, 0 if is_sol else tokens, state.query_chunk):
            stop = min(tokens, start + state.query_chunk)
            pending = state._graph_executor.submit(
                _secondary_worker, actor.compute_range, ops, start, stop,
                tokens, scale, audio_skip_ranges,
            )
            primary_part = _local_range(
                ops, qkv0, out0, start, stop, tokens, scale, audio0,
                audio_skip_ranges=audio_skip_ranges,
            )
            pending.result()
            returned = values['returned'][:stop - start]
            returned.copy_(pool.partial[:stop - start], non_blocking=True)
            torch.cuda.synchronize(primary)
            primary_part.add_(returned).mul_(16.0)
            target[start:stop].copy_(primary_part)
            transaction.mark_partial_target_write()
            del primary_part

    finally:
        if pending is not None:
            try:
                pending.result()
            except Exception:
                pass
        try:
            if primary_stream is not None:
                primary_stream.close()
        finally:
            try:
                weight_owner.close()
            finally:
                on_secondary(actor.reset)
