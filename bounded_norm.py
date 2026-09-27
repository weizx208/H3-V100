"""Row-wise scheduling of the original RMSNorm, with bounded scratch space."""
import logging
from contextvars import ContextVar
import torch
LOGGER = logging.getLogger(__name__)
MARKER = '_h3_v100_bounded_norm'
CHUNK_ROWS = 1024
CURRENT_OPTIONS = ContextVar('h3_norm_options', default=None)

def plan_rows(tokens, width, free_bytes, reserve_bytes):
    """Use full RMSNorm when affordable; otherwise bound its two tile outputs.

    Inactive allocator totals are deliberately excluded. A real allocation
    can still fail; the execution loop owns bounded recovery for that case.
    """
    row_bytes = int(width) * 4
    slack = 2 * 1024 ** 2 + int(tokens) * 8
    available = max(0, int(free_bytes) - int(reserve_bytes) - slack)
    output = int(tokens) * row_bytes
    if available >= 2 * output:
        return int(tokens)
    affordable = max(0, (available - output) // (2 * row_bytes))
    if affordable >= 256:
        affordable = affordable // 256 * 256
    return min(int(tokens), int(affordable))

def execute_with_recovery(original_forward, x, rows, recover):
    """Release failed frame/output before retrying, and never mutate residual."""
    while True:
        failure = None
        try:
            return original_forward(x) if rows >= x.shape[0] else run_chunked_norm(original_forward, x, rows)
        except torch.cuda.OutOfMemoryError as error:
            failure = type(error).__name__
        if rows <= 1:
            raise torch.cuda.OutOfMemoryError('H3 norm2 minimum workspace failed; full output still required: ' + failure) from None
        rows = max(1, rows // 2)
        recover(rows, failure)

def run_chunked_norm(original_forward, x, chunk_rows=CHUNK_ROWS, out=None):
    if out is None:
        out = torch.empty_like(x)
    for start in range(0, x.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x.shape[0])
        part = original_forward(x[start:stop])
        out[start:stop].copy_(part)
        del part
    return out

def try_cached_norm(original_forward, x, reserve_bytes, free_bytes, cached_bytes, *, acquire_from_graph=False):
    """Try actual output storage without crediting aggregate cache as capacity.

    Both the pre-check and the fresh check after allocation retain the original
    driver reserve plus two tile temporaries. A fragmented cache or operator
    OOM returns to the validated bounded execution path after partial tensors are gone.
    """
    tokens, width = x.shape
    row_bytes = width * 4
    output_bytes = tokens * row_bytes
    slack = 2 * 1024 ** 2 + tokens * 8
    minimum_rows = min(tokens, CHUNK_ROWS)
    minimum_free = reserve_bytes + slack + 2 * minimum_rows * row_bytes
    if (cached_bytes < output_bytes and not acquire_from_graph) or free_bytes < minimum_free:
        return (None, 0, 'ineligible')
    try:
        out = torch.empty_like(x)
        actual_free, _ = torch.cuda.mem_get_info(x.device)
        if actual_free < minimum_free:
            return (None, 0, 'reserve_declined')
        rows = min(tokens, (actual_free - reserve_bytes - slack) // (2 * row_bytes))
        if rows < tokens and rows >= 256:
            rows = rows // 256 * 256
        return (run_chunked_norm(original_forward, x, rows, out=out), rows, 'success')
    except torch.cuda.OutOfMemoryError:
        return (None, 0, 'resource_declined')

def make_bounded_norm(original_forward, stage='norm2'):

    def forward(self, x):
        if x.device.type != 'cuda' or x.dtype != torch.float32 or x.ndim != 2 or torch.is_grad_enabled():
            return original_forward(x)
        from .runtime_memory import request_cuda_headroom
        options = CURRENT_OPTIONS.get()
        if options is None:
            options = {}
        free, total = torch.cuda.mem_get_info(x.device)
        reserve = max(512 * 1024 ** 2, int(total * 0.04))
        tokens, width = x.shape
        rows = plan_rows(tokens, width, free, reserve)
        cached_result = None
        if rows == 0:
            from .graph_capacity import active_primary_graph
            cached_bytes = max(0, torch.cuda.memory_reserved(x.device) - torch.cuda.memory_allocated(x.device))
            cached_result, cached_rows, outcome = try_cached_norm(original_forward, x, reserve, free, cached_bytes,
                acquire_from_graph=active_primary_graph(x.device) is not None)
            if cached_result is not None:
                rows = cached_rows
        if rows == 0:
            minimum = x.numel() * x.element_size() + 2 * min(tokens, CHUNK_ROWS) * width * 4
            demand = minimum + reserve + 2 * 1024 ** 2 + tokens * 8
            result = request_cuda_headroom(x.device, options, reason=f'{stage}-plan', required_free_bytes=demand, minimum_reclaimable_mib=0, allow_vbar_release=True)
            rows = plan_rows(tokens, width, result['after']['free_bytes'], reserve)
            if rows == 0:
                raise torch.cuda.OutOfMemoryError(f"H3 {stage} no admissible plan: tokens={tokens}, driver_free={result['after']['free_bytes']}, target={demand}, shortfall={max(0, demand - result['after']['free_bytes'])}")
        if cached_result is not None:
            return cached_result

        def recover(next_rows, failure):
            LOGGER.warning('H3 %s resource retry: tokens=%d chunk_rows=%d reason=%s', stage, tokens, next_rows, failure)
            torch.cuda.empty_cache()
        return execute_with_recovery(original_forward, x, rows, recover)
    setattr(forward, MARKER, True)
    return forward
