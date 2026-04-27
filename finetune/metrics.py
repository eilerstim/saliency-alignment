"""Per-token attention alignment metrics.

Each metric accepts batched saliency / mask inputs of shape ``(..., H, W)``,
reduces over the last two dims, and returns a tensor of shape ``(...)``.
``NaN`` is returned for entries that are undefined (empty mask, zero-mass
saliency, constant saliency).

Aggregation:
    Per-token tensor -> per-image score: NaN-ignoring mean across tokens.
    Per-image list -> per-model score: mean and median across images.

The functions are torch-native and operate on the device of the inputs.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def _nan_like(reference: Tensor) -> Tensor:
    return torch.full_like(reference, float("nan"))


def coverage(
    sal: Float[Tensor, "... H W"], mask: Bool[Tensor, "... H W"]
) -> Float[Tensor, "..."]:
    """Fraction of total saliency mass falling inside ``mask``.

    ``coverage = sum(sal[mask]) / sum(sal)``. NaN where total saliency
    is non-positive.
    """
    sal = sal.float()
    total = sal.sum(dim=(-2, -1))
    mass_in = (sal * mask).sum(dim=(-2, -1))
    cov = mass_in / total.clamp(min=torch.finfo(sal.dtype).tiny)
    return torch.where(total > 0, cov, _nan_like(cov))


def amr_normalised(
    sal: Float[Tensor, "... H W"], mask: Bool[Tensor, "... H W"]
) -> Float[Tensor, "..."]:
    """Chance-normalised Attention Mass Ratio.

    ``amr = coverage(sal, mask) / (mask.sum() / mask.numel())``

    1.0 = uniform attention (chance level)
    > 1.0 = attention preferentially concentrated inside mask
    < 1.0 = attention preferentially outside mask

    NaN where the mask is empty or saliency has no mass.
    """
    cov = coverage(sal, mask)
    chance = mask.float().mean(dim=(-2, -1))
    amr = cov / chance.clamp(min=torch.finfo(cov.dtype).tiny)
    return torch.where(chance > 0, amr, _nan_like(amr))


def average_precision(
    scores: Float[Tensor, "... H W"], y_true: Bool[Tensor, "... H W"]
) -> Float[Tensor, "..."]:
    """Average precision treating saliency as a pixel-level ranker.

    Equivalent to ``sklearn.metrics.average_precision_score``: the mean
    precision at each positive's rank. Stable sort gives deterministic
    tie-breaking.

    NaN where there are no positives.
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
    """Normalised Scanpath Saliency at fixation locations.

    Z-normalises ``sal`` to mean=0, std=1 and returns the mean over pixels
    where ``mask`` is True. NaN for an empty mask or constant saliency.
    """
    sal_flat = sal.float().flatten(-2)
    mask_flat = mask.flatten(-2).to(sal_flat.dtype)

    mean = sal_flat.mean(dim=-1, keepdim=True)
    std = sal_flat.std(dim=-1, unbiased=False, keepdim=True)
    # Replace zero std with 1 so the z-score is finite; the result is
    # masked out below via ``torch.where``.
    z = (sal_flat - mean) / torch.where(std > 0, std, torch.ones_like(std))

    n = mask_flat.sum(dim=-1)
    score = (z * mask_flat).sum(dim=-1) / n.clamp(min=1)
    valid = (std.squeeze(-1) > 0) & (n > 0)
    return torch.where(valid, score, _nan_like(score))


MetricFn = Callable[
    [Float[Tensor, "... H W"], Bool[Tensor, "... H W"]], Float[Tensor, "..."]
]

# Mapping of metric name -> callable. Used by the validation loop to
# compute and aggregate metrics in a single pass.
ALIGNMENT_METRICS: dict[str, MetricFn] = {
    "amr_norm": amr_normalised,
    "ap": average_precision,
    "nss": nss,
}


def summarize_alignment(
    scores: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Compute mean / median / n per metric from per-image score lists.

    Empty entries are dropped (the caller can decide how to render them).
    """
    summary: dict[str, dict[str, float]] = {}
    for name, values in scores.items():
        if not values:
            continue
        arr = torch.tensor(values, dtype=torch.float64)
        summary[name] = {
            "mean": arr.mean().item(),
            "median": arr.median().item(),
            "n": float(len(values)),
        }
    return summary


def format_alignment_table(summary: dict[str, dict[str, float]]) -> str:
    """Render a per-model alignment metric summary as an ASCII table.

    Iterates over ``ALIGNMENT_METRICS`` (not ``summary``) so the column
    order is stable and missing metrics are rendered as ``—``.
    """
    header = ("Metric", "Mean", "Median", "N images")
    rows: list[tuple[str, ...]] = [header]
    for name in ALIGNMENT_METRICS:
        stats = summary.get(name)
        if stats is None:
            rows.append((name, "—", "—", "0"))
        else:
            rows.append(
                (
                    name,
                    f"{stats['mean']:.4f}",
                    f"{stats['median']:.4f}",
                    f"{int(stats['n'])}",
                )
            )

    widths = [max(len(r[c]) for r in rows) for c in range(len(header))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [sep]
    for i, row in enumerate(rows):
        lines.append(
            "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
        )
        if i == 0:
            lines.append(sep)
    lines.append(sep)
    return "\n".join(lines)
