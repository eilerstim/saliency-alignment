"""Per-token attention alignment metrics.

Each metric compares a single ``(H, W)`` saliency map against a single
``(H, W)`` boolean ground-truth mask and returns a scalar tensor on the
same device as the inputs. ``NaN`` is returned when the metric is
undefined (empty mask, zero-mass saliency, constant saliency).

Aggregation:
    Per-token tensor list -> per-image score: mean across tokens, ignoring NaNs.
    Per-image list -> per-model score: mean and median across images.

The functions are torch-native to avoid CPU<->GPU transfers in the
training/validation loop.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def _nan_like(reference: Tensor) -> Tensor:
    return torch.tensor(float("nan"), device=reference.device, dtype=torch.float32)


def coverage(
    sal: Float[Tensor, "H W"], mask: Bool[Tensor, "H W"]
) -> Float[Tensor, ""]:
    """Fraction of total saliency mass falling inside ``mask``.

    ``coverage = sum_{(i,j) in mask} sal_{i,j} / sum_{(i,j)} sal_{i,j}``

    Returns NaN when total saliency is non-positive.
    """
    sal = sal.float()
    total = sal.sum()
    if total <= 0:
        return _nan_like(sal)
    return sal[mask].sum() / total


def amr_normalised(
    sal: Float[Tensor, "H W"], mask: Bool[Tensor, "H W"]
) -> Float[Tensor, ""]:
    """Chance-normalised Attention Mass Ratio.

    ``amr = coverage(sal, mask) / (mask.sum() / mask.numel())``

    1.0 = uniform attention (chance level)
    > 1.0 = attention preferentially concentrated inside mask
    < 1.0 = attention preferentially outside mask

    Returns NaN when the mask is empty or saliency has no mass.
    """
    sal = sal.float()
    chance = mask.float().mean()
    if chance == 0:
        return _nan_like(sal)
    return coverage(sal, mask) / chance


def average_precision(
    scores: Float[Tensor, "H W"], y_true: Bool[Tensor, "H W"]
) -> Float[Tensor, ""]:
    """Average precision treating saliency as a pixel-level ranker.

    Equivalent to ``sklearn.metrics.average_precision_score``: the mean
    precision evaluated at each positive's rank position. Stable sort
    is used so ties are resolved deterministically.

    Returns NaN when there are no positives.
    """
    s = scores.float().flatten()
    y = y_true.flatten().to(s.dtype)
    n_pos = y.sum()
    if n_pos == 0:
        return _nan_like(s)
    order = torch.argsort(s, descending=True, stable=True)
    y_sorted = y[order]
    cum_tp = torch.cumsum(y_sorted, dim=0)
    ranks = torch.arange(1, y_sorted.numel() + 1, device=s.device, dtype=s.dtype)
    prec = cum_tp / ranks
    return (prec * y_sorted).sum() / n_pos


def nss(
    sal: Float[Tensor, "H W"], mask: Bool[Tensor, "H W"]
) -> Float[Tensor, ""]:
    """Normalised Scanpath Saliency at fixation locations.

    Z-normalises ``sal`` to mean=0, std=1 and returns the mean over pixels
    where ``mask`` is True. Invariant to monotonic affine scaling.

    Returns NaN when the mask is empty or the saliency is constant.
    """
    sal = sal.float()
    if not mask.any():
        return _nan_like(sal)
    std = sal.std(unbiased=False)
    if std == 0:
        return _nan_like(sal)
    z = (sal - sal.mean()) / std
    return z[mask].mean()


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
