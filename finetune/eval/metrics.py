"""Per-token attention-alignment metrics.

Each metric accepts batched saliency / mask inputs of shape ``(..., H, W)``,
reduces over the last two dims and returns a tensor of shape ``(...)``.
``NaN`` is returned for entries that are undefined (empty mask, zero-mass
saliency, constant saliency) so they can be safely ignored at aggregation
time without dragging the average towards zero.

The functions are torch-native, vectorise across the leading dimensions
and operate on the device of the inputs — pass the GPU tensors straight
out of the model and skip any host round-trip.

Aggregation:
    per-token tensor   -> per-image score: ``torch.nanmean`` across tokens.
    per-image scores   -> per-model score: NaN-ignoring mean / median /
                          std over images (computed in a final reduce).
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def _nan_like(reference: Tensor) -> Tensor:
    return torch.full_like(reference, float("nan"))


def coverage(
    sal: Float[Tensor, "... H W"], mask: Bool[Tensor, "... H W"]
) -> Float[Tensor, "..."]:
    """Fraction of total saliency mass falling inside ``mask``::

        coverage = sum(sal * mask) / sum(sal)

    NaN where the saliency mass is non-positive (an attention map that
    has been clipped to zero — should not happen after softmax).
    """
    sal = sal.float()
    total = sal.sum(dim=(-2, -1))
    mass_in = (sal * mask).sum(dim=(-2, -1))
    cov = mass_in / total.clamp(min=torch.finfo(sal.dtype).tiny)
    return torch.where(total > 0, cov, _nan_like(cov))


def amr_normalised(
    sal: Float[Tensor, "... H W"], mask: Bool[Tensor, "... H W"]
) -> Float[Tensor, "..."]:
    """Chance-normalised Attention Mass Ratio::

        amr = coverage(sal, mask) / (mask.sum() / mask.numel())

    * ``1.0``  → uniform attention (chance level).
    * ``> 1.0`` → attention preferentially concentrated inside the mask.
    * ``< 1.0`` → attention preferentially placed outside.

    NaN for an empty mask or zero-mass saliency. A full-coverage mask is
    fine: chance is ``1`` and AMR collapses to ``1`` by construction.
    """
    cov = coverage(sal, mask)
    chance = mask.float().mean(dim=(-2, -1))
    amr = cov / chance.clamp(min=torch.finfo(cov.dtype).tiny)
    return torch.where(chance > 0, amr, _nan_like(amr))


def average_precision(
    scores: Float[Tensor, "... H W"], y_true: Bool[Tensor, "... H W"]
) -> Float[Tensor, "..."]:
    """Pixel-level Average Precision.

    Sorts pixels by saliency (descending) and computes the standard

        AP = (1 / N_pos) * sum_{i: y_i = 1} P(i)

    matching ``sklearn.metrics.average_precision_score`` for a binary
    problem. Stable sort gives deterministic tie-breaking. NaN where the
    mask contains no positives (AP is undefined; returning ``0`` would
    bias the per-image average downwards).
    """
    s = scores.float().flatten(-2)
    y = y_true.flatten(-2).to(s.dtype)

    order = torch.argsort(s, dim=-1, descending=True, stable=True)
    y_sorted = torch.gather(y, dim=-1, index=order)

    cum_tp = torch.cumsum(y_sorted, dim=-1)
    ranks = torch.arange(1, s.shape[-1] + 1, device=s.device, dtype=s.dtype)

    n_pos = y.sum(dim=-1)
    ap = (cum_tp / ranks * y_sorted).sum(dim=-1) / n_pos.clamp(min=1)
    return torch.where(n_pos > 0, ap, _nan_like(ap))


def nss(
    sal: Float[Tensor, "... H W"], mask: Bool[Tensor, "... H W"]
) -> Float[Tensor, "..."]:
    """Normalised Scanpath Saliency.

    Z-scores ``sal`` to mean 0 / std 1 (population std) and returns the
    average z-score over pixels selected by ``mask``. NaN for an empty
    mask or a constant saliency map (z-score undefined).
    """
    sal_flat = sal.float().flatten(-2)
    mask_flat = mask.flatten(-2).to(sal_flat.dtype)

    mean = sal_flat.mean(dim=-1, keepdim=True)
    std = sal_flat.std(dim=-1, unbiased=False, keepdim=True)
    # Replace zero std with 1 so the z-score stays finite; the result is
    # masked out below via ``torch.where``.
    z = (sal_flat - mean) / torch.where(std > 0, std, torch.ones_like(std))

    n = mask_flat.sum(dim=-1)
    score = (z * mask_flat).sum(dim=-1) / n.clamp(min=1)
    valid = (std.squeeze(-1) > 0) & (n > 0)
    return torch.where(valid, score, _nan_like(score))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def per_image_aggregate(
    per_token: Float[Tensor, "T M"]
) -> Float[Tensor, " M"]:
    """NaN-ignoring mean across the token dimension. ``M`` is the number of
    metrics. All-NaN columns return NaN (the image contributes nothing)."""
    if per_token.numel() == 0:
        return torch.full(
            (per_token.shape[-1],), float("nan"),
            device=per_token.device, dtype=per_token.dtype,
        )
    return torch.nanmean(per_token, dim=0)


def per_model_aggregate(
    per_image: Float[Tensor, "N M"]
) -> dict[str, Float[Tensor, " M"]]:
    """Reduce a stack of per-image scores into mean / median / std plus the
    per-metric count of finite images. NaN images are excluded so they
    don't drag the aggregates."""
    finite = torch.isfinite(per_image)
    n = finite.sum(dim=0)

    # nanmean / nanmedian preserve NaN columns when all entries are NaN.
    mean = torch.nanmean(per_image, dim=0)
    median = torch.nanmedian(per_image, dim=0).values

    # Population std with NaNs ignored: var = E[x^2] - E[x]^2 over finite values.
    safe = torch.where(finite, per_image, torch.zeros_like(per_image))
    sq_mean = (safe**2).sum(dim=0) / n.clamp(min=1)
    var = sq_mean - mean**2
    std = var.clamp(min=0).sqrt()
    std = torch.where(n > 0, std, torch.full_like(std, float("nan")))

    return {"mean": mean, "median": median, "std": std, "n_images": n}
