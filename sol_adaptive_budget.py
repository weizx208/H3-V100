"""Memory-bounded, global equal-NNZ Sol route allocation.

Only a bounded head tile owns logits and int64 sorting scratch. Across tiles
we retain FP32 cumulative probabilities and int16 column indices, never full
scores, probabilities, int64 indices and prior masks together. The shared
Top-p search still spans ALL heads; it is not a per-head budget approximation.
"""
from __future__ import annotations
import math
import torch
TILE_PAIRS = 4 * 1024 * 1024

def planner_memory_bytes(batch, heads, blocks, prefix_blocks=0):
    """Conservative incremental peak, excluding already resident QKV/history.

    Include centroid prepare/conversion, sorted storage, tile intermediates,
    final boolean route and worst-case CSR. A separate 576 MiB downstream
    reserve is required by the caller. Never credit allocator fragmentation.
    """
    bh = int(batch) * int(heads)
    n = int(blocks)
    tail = max(0, n - int(prefix_blocks))
    tile = min(bh, max(1, TILE_PAIRS // max(1, tail * tail)))
    sorted_storage = bh * tail * tail * 6
    scratch = tile * tail * tail * 80
    centroids = bh * n * (128 * 4 * 3 + 8)
    route = bh * n * n * 5 + bh * n * math.ceil(n / 32) * 4
    return max(sorted_storage + scratch, route + sorted_storage // 3) + centroids + 64 * 2 ** 20

def _finite(*values):
    if not bool(torch.stack([torch.isfinite(value).all() for value in values]).all().item()):
        raise ValueError('adaptive planner received non-finite centroids/scores')

def _unpack_tail(bitmask, prefix, blocks):
    columns = torch.arange(prefix, blocks, device=bitmask.device)
    return (bitmask[..., prefix:blocks, columns // 32] >> columns % 32 & 1).bool()

def bounded_equal_budget_route(qc, kc, threshold, prefix_blocks, *, risk=None, prior_bitmask=None, temperature=2.0, variance_alpha=0.15, minimum_optional=8, minimum_optional_ratio=0.07, maximum_optional_ratio=0.5, prior_logit_bonus=0.02):
    """Return a route and budget receipt using the reference global Top-p rule.

    CDF binary search uses per-row searchsorted rather than allocating a
    full [heads, blocks, blocks] comparison on each of twenty iterations.
    Final count correction uses adjacent CDF differences (probabilities).
    """
    batch, heads, blocks, width = map(int, qc.shape)
    if qc.shape != kc.shape or width != 128 or threshold.shape != qc.shape[:-1]:
        raise ValueError('adaptive planner requires matching B,H,N,128 centroids')
    if blocks > 32767:
        raise ValueError('adaptive planner column index exceeds int16 capacity')
    _finite(qc, kc, threshold)
    if risk is not None:
        if risk.shape != threshold.shape:
            raise ValueError('adaptive risk shape mismatch')
        _finite(risk)
    prefix = min(blocks, max(0, int(prefix_blocks)))
    bh, tail = (batch * heads, blocks - prefix)
    if tail * tail > TILE_PAIRS:
        raise ValueError('adaptive planner row tile exceeds validated workspace')
    if not tail:
        route = torch.ones((batch, heads, blocks, blocks), device=qc.device, dtype=torch.bool)
        return (route, {'p': 1.0, 'base_nnz': route.numel(), 'same_global_nnz': True, 'optional_nnz': 0, 'row_count_min': 0, 'row_count_max': 0, 'applied_optional_floor': 0})
    rows = bh * tail
    qflat, kflat = (qc.reshape(bh, blocks, width), kc.reshape(bh, blocks, width))
    limits = threshold.reshape(bh, blocks)
    riskflat = None if risk is None else risk.reshape(bh, blocks)
    prior = None
    if prior_bitmask is not None:
        if tuple(prior_bitmask.shape) != (batch, heads, blocks, math.ceil(blocks / 32)):
            raise ValueError('adaptive history shape mismatch')
        prior = prior_bitmask.reshape(bh, blocks, -1)
    cdf = torch.empty((rows, tail), device=qc.device, dtype=torch.float32)
    order16 = torch.empty((rows, tail), device=qc.device, dtype=torch.int16)
    index = torch.arange(tail, device=qc.device)
    optional = (index[:, None] - index[None, :]).abs() > 1
    available_row = optional.sum(-1, dtype=torch.int64)
    row_denominator = available_row.clamp_min(1).unsqueeze(-1)
    available = available_row.repeat(bh)
    base_optional = torch.zeros((), device=qc.device, dtype=torch.int64)
    tile_heads = min(bh, max(1, TILE_PAIRS // (tail * tail)))
    tile_finite = []
    for first in range(0, bh, tile_heads):
        last = min(bh, first + tile_heads)
        scores = torch.matmul(qflat[first:last, prefix:], kflat[first:last, prefix:].transpose(-2, -1))
        tile_finite.append(torch.isfinite(scores).all())
        base_optional += ((scores > limits[first:last, prefix:, None]) & optional).sum()
        if riskflat is not None and float(variance_alpha) > 0:
            score_mean = scores.mean(-1, keepdim=True)
            score_std = scores.var(-1, keepdim=True, unbiased=False).clamp_min_(1e-08).sqrt_()
            scores = (scores - score_mean) / score_std
            r = riskflat[first:last, prefix:]
            r = (r - r.mean(-1, keepdim=True)) / r.var(-1, keepdim=True, unbiased=False).clamp_min(1e-08).sqrt()
            scores.add_(r.unsqueeze(-2), alpha=float(variance_alpha))
        masked = scores.masked_fill(~optional, 0.0)
        mean = masked.sum(-1, keepdim=True) / row_denominator
        centered = (scores - mean).masked_fill(~optional, 0.0)
        variance = centered.square().sum(-1, keepdim=True) / row_denominator
        logits = centered / variance.clamp_min_(1e-08).sqrt_()
        logits.div_(max(0.05, float(temperature)))
        logits.masked_fill_(~optional, -torch.inf)
        if prior is not None and float(prior_logit_bonus) > 0:
            logits.add_(_unpack_tail(prior[first:last], prefix, blocks) & optional, alpha=float(prior_logit_bonus))
        logits.masked_fill_((available_row == 0).view(1, tail, 1), 0.0)
        probabilities = torch.softmax(logits, dim=-1)
        sorted_probabilities, order = probabilities.sort(-1, descending=True)
        destination = slice(first * tail, last * tail)
        cdf[destination].copy_(sorted_probabilities.cumsum(-1).reshape(-1, tail))
        order16[destination].copy_(order.reshape(-1, tail))
        del scores, masked, mean, centered, variance, logits, probabilities, sorted_probabilities, order
    scores_finite = torch.stack(tile_finite).all()
    finite_value, target = torch.stack((scores_finite.to(torch.int64), base_optional)).tolist()
    if not finite_value:
        raise ValueError('adaptive planner received non-finite centroids/scores')
    requested_floor = math.ceil(tail * max(0.0, min(1.0, float(minimum_optional_ratio))))
    floor = max(int(minimum_optional), min(requested_floor, target // rows))
    minimum = available.clamp_max(floor)
    ceiling = max(int(minimum_optional), math.ceil(tail * max(0.0, min(1.0, float(maximum_optional_ratio)))))
    maximum = available.clamp_max(ceiling)
    minimum_sum, maximum_sum = torch.stack((minimum.sum(), maximum.sum())).tolist()
    relaxed = target < minimum_sum or target > maximum_sum
    if relaxed:
        minimum, maximum, floor = (torch.zeros_like(available), available.clone(), 0)
    low, high = (torch.zeros((), device=qc.device), torch.ones((), device=qc.device))
    for _ in range(20):
        p = (low + high) * 0.5
        counts = torch.searchsorted(cdf, p.expand(rows, 1).contiguous()).flatten() + 1
        counts = counts.maximum(minimum).minimum(maximum)
        below = counts.sum() < base_optional
        low, high = (torch.where(below, p, low), torch.where(below, high, p))
    counts = torch.searchsorted(cdf, high.expand(rows, 1).contiguous()).flatten() + 1
    counts = counts.maximum(minimum).minimum(maximum)
    row = torch.arange(rows, device=qc.device)
    for _ in range(tail + 1):
        current = int(counts.sum().item())
        if current == target:
            break
        adding = current < target
        eligible = counts < maximum if adding else counts > minimum
        take = min(abs(target - current), int(eligible.sum().item()))
        if not take:
            raise RuntimeError('adaptive route cannot satisfy global NNZ')
        pos = (counts if adding else counts - 1).clamp(0, tail - 1)
        marginal = cdf[row, pos] - torch.where(pos > 0, cdf[row, (pos - 1).clamp_min(0)], 0.0)
        marginal.masked_fill_(~eligible, -torch.inf if adding else torch.inf)
        chosen = torch.topk(marginal, take, largest=adding).indices
        counts[chosen] += 1 if adding else -1
    else:
        raise RuntimeError('adaptive global count correction did not converge')
    del cdf
    route = torch.ones((bh, blocks, blocks), device=qc.device, dtype=torch.bool)
    selected_nnz = torch.zeros((), device=qc.device, dtype=torch.int64)
    rank = torch.arange(tail, device=qc.device).view(1, 1, tail)
    for first in range(0, bh, tile_heads):
        last = min(bh, first + tile_heads)
        selection = torch.zeros((last - first, tail, tail), device=qc.device, dtype=torch.bool)
        order = order16[first * tail:last * tail].view(last - first, tail, tail).long()
        selection.scatter_(-1, order, rank < counts[first * tail:last * tail].view(last - first, tail, 1))
        selection &= optional
        selected_nnz += selection.sum(dtype=torch.int32)
        route[first:last, prefix:, prefix:] = selection | ~optional
        del selection, order
    forced_count, selected_count = torch.stack(((~optional).sum(), selected_nnz)).tolist()
    base_nnz = bh * (blocks * blocks - tail * tail + forced_count) + target
    if selected_count != target:
        raise RuntimeError('adaptive route changed the base Sol NNZ')
    return (route.view(batch, heads, blocks, blocks), {'base_nnz': base_nnz})
