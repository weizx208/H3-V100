"""Production Sol adaptive route policy with transactional base-Sol fallback."""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import math
import torch
from .sol_adaptive_budget import bounded_equal_budget_route, planner_memory_bytes
from .sol_native import _build_route
from .dual_runtime import _fatal_device_failure
LOGGER = logging.getLogger('H3V100AdaptiveBudget')
POLICY_KEY = 'v100_h3_sol_adaptive_budget_policy'
POLICY_TAG = 'equal-nnz-t2-risk-a15-floor-r07-hyst-b02-bounded-v1'
MAX_TOKENS = 112853
RESERVE_BYTES = 576 * 2 ** 20

@dataclass
class AdaptiveBudgetState:
    min_tokens: int = 16000
    max_tokens: int = MAX_TOKENS
    workspace_limit_mib: int = 1536
    history_limit_mib: int = 128
    temperature: float = 2.0
    variance_alpha: float = 0.15
    minimum_optional: int = 8
    minimum_optional_ratio: float = 0.07
    maximum_optional_ratio: float = 0.5
    prior_logit_bonus: float = 0.02
    route_history_bytes: int = 0
    route_history: dict = field(default_factory=dict, repr=False)
    disabled_signatures: set = field(default_factory=set, repr=False)
    run_signature: tuple | None = field(default=None, repr=False)

    def release_history(self):
        self.route_history.clear()
        self.route_history_bytes = 0

    def reset(self):
        self.release_history()
        self.disabled_signatures.clear()
        self.run_signature = None

    def _base(self, reason, ops, q, k, v, common):
        return _build_route(ops, q, k, v, **common)

    def build_route(self, ops, q, k, v, *, tau, prefix_stop, memory_limit_mib=1024, route_context=None):
        common = dict(tau=tau, prefix_stop=prefix_stop, memory_limit_mib=memory_limit_mib)
        tokens = int(q.shape[2])
        if not self.min_tokens <= tokens <= self.max_tokens:
            self.release_history()
            return self._base('token-range', ops, q, k, v, common)
        signature = (str(q.device), str(q.dtype), tuple(q.shape), float(tau), int(prefix_stop), self.temperature, self.variance_alpha, self.minimum_optional, self.minimum_optional_ratio, self.maximum_optional_ratio, self.prior_logit_bonus)
        if signature != self.run_signature:
            self.release_history()
            self.run_signature = signature
        if signature in self.disabled_signatures:
            return self._base('failed-earlier-this-run', ops, q, k, v, common)
        if not hasattr(ops, 'sol_prepare_with_risk'):
            return self._base('native-risk-unavailable', ops, q, k, v, common)
        blocks = math.ceil(tokens / 64)
        prefix = min(blocks, math.ceil(max(0, int(prefix_stop)) / 64))
        estimated = planner_memory_bytes(q.shape[0], q.shape[1], blocks, prefix)
        if estimated > self.workspace_limit_mib * 2 ** 20:
            self.release_history()
            return self._base('workspace-cap', ops, q, k, v, common)
        if int(torch.cuda.mem_get_info(q.device)[0]) < estimated + RESERVE_BYTES:
            self.release_history()
            return self._base('driver-headroom', ops, q, k, v, common)
        history_key = None
        history_step = None
        prior = None
        bitmask_bytes = int(q.shape[0]) * int(q.shape[1]) * blocks * math.ceil(blocks / 32) * 4
        history_allowed = 50 * bitmask_bytes <= self.history_limit_mib * 2 ** 20
        if not history_allowed:
            self.release_history()
        elif isinstance(route_context, dict):
            try:
                history_step = int(route_context['step_index'])
                history_key = (signature, int(route_context['block_index']))
                if not 0 <= history_key[1] < 50:
                    history_key = None
            except (KeyError, TypeError, ValueError):
                history_key = None
        if history_key is not None:
            previous = self.route_history.get(history_key)
            if previous is not None and previous[0] == history_step - 1:
                prior = previous[1]
        failure = None
        try:
            route, info = self._build_adaptive(ops, q, k, v, common, prefix, prior)
        except Exception as error:
            if _fatal_device_failure(error):
                raise
            failure = f'{type(error).__name__}: {error}'
        if failure is not None:
            prior = previous = None
            self.release_history()
            self.disabled_signatures.add(signature)
            if q.device.type == 'cuda':
                torch.cuda.empty_cache()
            LOGGER.warning('H3 adaptive planner failed; retrying original Sol: %s', failure)
            return self._base('planner-error', ops, q, k, v, common)
        if history_key is not None:
            previous = self.route_history.get(history_key)
            previous_bytes = previous[1].numel() * previous[1].element_size() if previous else 0
            self.route_history[history_key] = (history_step, route[2])
            self.route_history_bytes += bitmask_bytes - previous_bytes
        return route

    def _build_adaptive(self, ops, q, k, v, common, prefix, prior):
        qc, kc, vc, threshold, risk = ops.sol_prepare_with_risk(q, k, v, float(common['tau']), 64)
        if not bool(torch.isfinite(vc).all().item()):
            raise ValueError('adaptive V centroids are non-finite')
        selected, info = bounded_equal_budget_route(qc, kc, threshold, prefix, risk=risk, prior_bitmask=prior, temperature=self.temperature, variance_alpha=self.variance_alpha, minimum_optional=self.minimum_optional, minimum_optional_ratio=self.minimum_optional_ratio, maximum_optional_ratio=self.maximum_optional_ratio, prior_logit_bonus=self.prior_logit_bonus)
        blocks = selected.shape[-1]
        rows = selected.numel() // blocks
        metadata_bytes = (rows + 1) * 4 + int(info['base_nnz']) * 4 + rows * math.ceil(blocks / 32) * 4
        centroids_bytes = kc.numel() * 4
        if metadata_bytes + centroids_bytes > max(128, int(common['memory_limit_mib'])) * 2 ** 20:
            raise RuntimeError('adaptive route exceeds Sol metadata cap')
        del qc, threshold, risk
        row_ptr, offsets = ops.sol_pack_route_csr(selected, 64, 0)
        if offsets.numel() != info['base_nnz']:
            raise RuntimeError('adaptive packed CSR changed the base Sol NNZ')
        bitmask = ops.sol_pack_route_bitmask(selected)
        del selected
        kc16 = kc.half()
        del kc
        vc16 = vc.half()
        del vc
        if int(torch.cuda.mem_get_info(q.device)[0]) < 320 * 2 ** 20:
            raise RuntimeError('adaptive route leaves insufficient projection headroom')
        density = offsets.numel() / max(1, rows * blocks)
        return ((row_ptr, offsets, bitmask, kc16, vc16, density), info)

def install_adaptive_budget(options, *, min_tokens=16384):
    """Install per-model state; caller is the public Sol branch only."""
    state = AdaptiveBudgetState(min_tokens=max(16000, int(min_tokens)))
    options[POLICY_KEY] = {'enabled': True, 'owner': 'main', 'tag': POLICY_TAG, 'min_tokens': state.min_tokens, 'max_tokens': state.max_tokens, 'route_builder': state.build_route, 'state': state}
    return state
