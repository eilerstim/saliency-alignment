"""Per-token attention alignment metrics.

Each metric compares a single ``(H, W)`` saliency map against a single
``(H, W)`` boolean ground-truth mask and returns a scalar tensor on the
same device as the inputs. ``NaN`` is returned when the metric is
undefined (empty mask, zero-mass saliency, constant saliency).

Aggregation:
    Per-token list -> per-image score: mean across tokens (NaNs ignored).
    Per-image list -> per-model score: mean and median across images.

The functions are torch-native to avoid CPU<->GPU transfers in the
validation loop and assume float-typed saliency input (cast once at the
call site).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def _nan(device: torch.device) -> Tensor:
    return torch.tensor(float("nan"), device=device)


def coverage(
    sal: Float[Tensor, "H W"], mask: Bool[Tensor, "H W"]
) -> Float[Tensor, ""]:
    """Fraction of total saliency mass falling inside ``mask``.

    ``coverage = sum(sal[mask]) / sum(sal)``. Returns NaN when total
    saliency is non-positive.
    """
    total = sal.sum()
    if total <= 0:
        return _nan(sal.device)
    return sal[mask].sum() / total


def amr_normalised(
    sal: Float[Tensor, "H W"], mask: Bool[Tensor, "H W"]
) -> Float[Tensor, ""]:
    """Chance-normalised Attention Mass Ratio.

    ``amr = coverage(sal, mask) / (mask.sum() / mask.numel())``

    1.0 = uniform attention (chance level)
    > 1.0 = attention preferentially concentrated inside mask
    < 1.0 = attention preferentially outside mask
    """
    chance = mask.float().mean()
    if chance == 0:
        return _nan(sal.device)
    return coverage(sal, mask) / chance


def average_precision(
    scores: Float[Tensor, "H W"], y_true: Bool[Tensor, "H W"]
) -> Float[Tensor, ""]:
    """Average precision treating saliency as a pixel-level ranker.

    Equivalent to ``sklearn.metrics.average_precision_score``: the mean
    precision at each positive's rank. Stable sort gives deterministic
    tie-breaking.
    """
    s = scores.flatten()
    y = y_true.flatten().to(s.dtype)
    n_pos = y.sum()
    if n_pos == 0:
        return _nan(s.device)
    order = torch.argsort(s, descending=True, stable=True)
    y_sorted = y[order]
    cum_tp = torch.cumsum(y_sorted, dim=0)
    ranks = torch.arange(1, y_sorted.numel() + 1, device=s.device, dtype=s.dtype)
    return (cum_tp / ranks * y_sorted).sum() / n_pos


def nss(
    sal: Float[Tensor, "H W"], mask: Bool[Tensor, "H W"]
) -> Float[Tensor, ""]:
    """Normalised Scanpath Saliency at fixation locations.

    Z-normalises ``sal`` to mean=0, std=1 and returns the mean over pixels
    where ``mask`` is True. Returns NaN for an empty mask or constant saliency.
    """
    if not mask.any():
        return _nan(sal.device)
    std = sal.std(unbiased=False)
    if std == 0:
        return _nan(sal.device)
    return ((sal - sal.mean()) / std)[mask].mean()


MetricFn = Callable[
    [Float[Tensor, "H W"], Bool[Tensor, "H W"]], Float[Tensor, ""]
]

# Mapping of metric name -> callable. Used by the validation loop to
# compute and aggregate metrics in a single pass.
ALIGNMENT_METRICS: dict[str, MetricFn] = {
    "amr_norm": amr_normalised,
    "ap": average_precision,
    "nss": nss,
}
