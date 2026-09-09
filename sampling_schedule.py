"""Read sampler boundaries for local Sol/EasyCache policy; never alter sigmas."""
import math
import torch

def sample_sigma_boundaries(sigmas):
    """Return an unambiguous descending schedule, including its final boundary.

    Euler evaluates the start of each interval, including when a split ends at
    nonzero sigma. Duplicate/nonfinite/non-descending schedules cannot safely
    map sigma to an ordinal: leave approximation policy unarmed for those.
    This runs at the sample boundary only, using the existing FP32 convention.
    """
    try:
        values = tuple(torch.as_tensor(sigmas).detach().float().cpu().reshape(-1).tolist())
    except (TypeError, ValueError, RuntimeError):
        return ()
    if len(values) < 2 or not all((math.isfinite(v) and v >= 0 for v in values)):
        return ()
    if any((a <= b for a, b in zip(values, values[1:]))):
        return ()
    return values
