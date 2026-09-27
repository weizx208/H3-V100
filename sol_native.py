"""Native corrected Sol route construction and operator loading."""
from __future__ import annotations
import logging
import math
from pathlib import Path
import torch
LOGGER = logging.getLogger('H3V100FusedSolShadow')
STABLE_VBAR_KEY = 'v100_h3_dynamic_vbar_controller'

def load_sol_ops(require_range=False, require_dense=False, require_dual_centroid=False):
    namespace = 'h3_v100_research_flash_bm128_cuda'
    ops = getattr(torch.ops, namespace)
    required = ['sol_prepare', 'sol_build_route_csr_bitmask', 'sol_build_route_csr_bitmask_limited', 'sol_sparse_corrected_csr_fused']
    if require_range:
        required.append('sol_sparse_corrected_csr_fused_range')
    if require_dense:
        required.append('sol_dense_rect')
    if require_dual_centroid:
        required.append('sol_kv_centroids_fp16')
    if any((not hasattr(ops, name) for name in required)):
        native_dir = Path(__file__).resolve().parent
        library = native_dir / 'h3_v100_sol_cuda.cp312-win_amd64.pyd'
        if not library.is_file():
            raise RuntimeError(f'Missing H3 Sol native library: {library}')
        torch.ops.load_library(str(library))
    missing = [name for name in required if not hasattr(ops, name)]
    if missing:
        raise RuntimeError(f'H3 Sol native library misses {missing}')
    return ops

def _build_route(ops, q, k, v, *, tau, prefix_stop, memory_limit_mib=1024):
    block_size = 64
    batch, heads, tokens, width = (int(value) for value in q.shape)
    blocks = math.ceil(tokens / block_size)
    prefix_blocks = min(blocks, math.ceil(max(0, int(prefix_stop)) / block_size))
    qc, kc, vc, threshold = ops.sol_prepare(q, k, v, float(tau), block_size)
    words = math.ceil(blocks / 32)
    rows = batch * heads * blocks
    fixed_persistent_bytes = batch * heads * blocks * words * 4 + (rows + 1) * 4 + batch * heads * blocks * width * 2 * 2
    configured_bytes = max(128, int(memory_limit_mib)) * 2 ** 20
    driver_free = int(torch.cuda.mem_get_info(q.device)[0])
    bitmask_rowptr_bytes = batch * heads * blocks * words * 4 + (rows + 1) * 4
    allocation_reserve = (320 + 256) * 2 ** 20
    offset_budget_bytes = min(configured_bytes - fixed_persistent_bytes, driver_free - allocation_reserve - bitmask_rowptr_bytes)
    if offset_budget_bytes < 0:
        raise RuntimeError(f'corrected Sol route fixed metadata needs {fixed_persistent_bytes / 2 ** 20:.1f} MiB with reserve, but only {driver_free / 2 ** 20:.1f} MiB is driver-free')
    row_ptr, offsets, bitmask = ops.sol_build_route_csr_bitmask_limited(qc, kc, threshold, block_size, prefix_blocks, 1, int(offset_budget_bytes // 4))
    density = int(offsets.numel()) / max(1, batch * heads * blocks * blocks)
    del qc, threshold
    kc16 = kc.half()
    del kc
    vc16 = vc.half()
    del vc
    post_route_free = int(torch.cuda.mem_get_info(q.device)[0])
    if post_route_free < 320 * 2 ** 20:
        raise RuntimeError(f'corrected Sol route leaves only {post_route_free / 2 ** 20:.1f} MiB for streamed projection; 320 MiB is required')
    return (row_ptr, offsets, bitmask, kc16, vc16, density)
