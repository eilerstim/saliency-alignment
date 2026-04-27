"""Saliency alignment evaluation metrics.

All metrics operate on a *single* (saliency, mask) pair where:

* ``saliency`` is a tensor of attention scores at the resolution of ``mask``.
  Higher values mean the model attends more to that pixel.
* ``mask`` is a boolean tensor of the same shape, ``True`` inside the
  reference region for the current token, ``False`` elsewhere.

The metrics themselves are intentionally agnostic to whether the saliency
input has been normalised to a probability distribution. ``amr_normalised``
divides out the total saliency mass, so any non-negative scaling works;
``average_precision`` is rank-based; ``nss`` z-scores its input. We still
recommend feeding probability-distribution-like saliency (post-softmax) to
keep numerical behaviour consistent with the training criterion.

All functions return ``float("nan")`` for ill-defined inputs (e.g. empty
mask, all-zero saliency, mask covering the entire image) instead of raising,
so callers can safely aggregate across many tokens with ``nanmean``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    """Move a tensor to CPU and convert to a contiguous fp64 NumPy array.

    fp64 keeps cumulative sums and z-scoring numerically stable when the
    saliency mass is concentrated on a few pixels (very small post-softmax
    values otherwise lose precision under fp32).
    """
    if isinstance(x, Tensor):
        x = x.detach().to(dtype=torch.float64, device="cpu").contiguous().numpy()
    return np.asarray(x, dtype=np.float64)


def _to_bool_numpy(mask: Tensor | np.ndarray) -> np.ndarray:
    """Convert a mask (tensor or array, any dtype) to a bool NumPy array."""
    if isinstance(mask, Tensor):
        return mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    return np.asarray(mask, dtype=bool)


def amr_normalised(
    saliency: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
) -> float:
    """Chance-normalised Attention Mass Ratio.

    Defined as the fraction of total saliency mass falling inside ``mask``,
    divided by the fraction of pixels covered by ``mask``::

        AMR = (sum(sal * mask) / sum(sal)) / (mask.sum() / mask.size)

    Interpretation:

    * ``AMR == 1.0`` → the model places attention proportionally to area
      (chance level).
    * ``AMR > 1.0``  → attention is preferentially concentrated *inside*
      the reference mask.
    * ``AMR < 1.0``  → attention is preferentially placed *outside*.

    Returns ``nan`` if the mask is empty or covers the full image (chance
    level undefined / trivial), or if the saliency mass is zero.
    """
    sal = _to_numpy(saliency)
    msk = _to_bool_numpy(mask)

    if sal.shape != msk.shape:
        raise ValueError(
            f"saliency shape {sal.shape} does not match mask shape {msk.shape}"
        )

    n_in = int(msk.sum())
    n_total = msk.size
    if n_in == 0 or n_in == n_total:
        return float("nan")

    total_mass = float(sal.sum())
    if total_mass <= 0.0 or not math.isfinite(total_mass):
        return float("nan")

    coverage = float(sal[msk].sum()) / total_mass
    chance = n_in / n_total
    return coverage / chance


def average_precision(
    saliency: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
) -> float:
    """Average Precision treating saliency as a binary-detection score map.

    Pixels are sorted by saliency (descending); each mask pixel is then a
    true positive. The metric equals the area under the precision-recall
    curve, computed as the standard

        AP = (1 / N_pos) * sum_{i: y_i = 1} P(i)

    where P(i) is precision at the i-th ranked pixel. This matches
    sklearn's ``average_precision_score`` for a binary problem.

    Returns ``nan`` if the mask contains no positives. AP is undefined
    in that case, and returning ``0`` (as in the reference snippet) would
    bias the per-image / per-model average downwards.

    Notes:
        * Ties in saliency are broken by ``np.argsort``'s stable ordering
          (NumPy default is non-stable; we explicitly use ``kind='stable'``
          to make results reproducible).
        * The ``+ 1e-8`` smoothing in the inspirational snippet is removed:
          when ``y.sum() > 0`` the denominator is already strictly positive,
          and the epsilon would otherwise systematically bias AP downwards.
    """
    sal = _to_numpy(saliency).ravel()
    msk = _to_bool_numpy(mask).ravel()

    if sal.shape != msk.shape:
        raise ValueError(
            f"saliency size {sal.size} does not match mask size {msk.size}"
        )

    n_pos = int(msk.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-sal, kind="stable")
    y_sorted = msk[order].astype(np.float64)

    cum_tp = np.cumsum(y_sorted)
    precision = cum_tp / np.arange(1, y_sorted.size + 1, dtype=np.float64)

    return float(precision[y_sorted == 1.0].sum() / n_pos)


def nss(
    saliency: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
) -> float:
    """Normalised Scanpath Saliency.

    The saliency map is z-scored over *all* pixels (population mean and
    standard deviation), and the metric is the average z-score at the
    pixels selected by ``mask``::

        z = (sal - sal.mean()) / sal.std(ddof=0)
        NSS = z[mask].mean()

    Interpretation:

    * ``NSS > 0`` → the model assigns above-average saliency to the
      reference region.
    * ``NSS == 0`` → no preference relative to image-wide average.

    Returns ``nan`` if the mask is empty or the saliency map has zero
    variance (z-score undefined).
    """
    sal = _to_numpy(saliency)
    msk = _to_bool_numpy(mask)

    if sal.shape != msk.shape:
        raise ValueError(
            f"saliency shape {sal.shape} does not match mask shape {msk.shape}"
        )

    if not msk.any():
        return float("nan")

    std = float(sal.std(ddof=0))
    if std <= 0.0 or not math.isfinite(std):
        return float("nan")

    mean = float(sal.mean())
    return float(((sal[msk] - mean) / std).mean())


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_per_image(token_scores: Sequence[float]) -> float:
    """Average per-token scores into a single per-image score.

    NaNs (skipped tokens) are ignored. Returns ``nan`` if all tokens were
    skipped (so the image contributes nothing to the per-model average).
    """
    if len(token_scores) == 0:
        return float("nan")
    arr = np.asarray(token_scores, dtype=np.float64)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def aggregate_per_model(image_scores: Sequence[float]) -> dict[str, float]:
    """Reduce per-image scores into the headline per-model statistics.

    Returns ``mean``, ``median``, ``std`` (population, ddof=0) and the count
    of images that contributed a finite score. NaN images are excluded so
    they don't drag the mean towards zero.
    """
    arr = np.asarray(image_scores, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "n_images": 0,
        }
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std(ddof=0)),
        "n_images": n,
    }
