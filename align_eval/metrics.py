"""Per-token attention-alignment metrics.

Each metric takes batched saliency / mask inputs of shape ``(..., H, W)``,
reduces over the last two dims and returns a tensor of shape ``(...)``.
``NaN`` is returned for entries that are undefined (empty mask, zero-mass
saliency, constant saliency) so they can be ignored at aggregation time
without dragging the average down.

Functions are torch-native and run on the device of the inputs.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from vl_saliency.maps import SaliencyGrid

METRIC_NAMES: tuple[str, ...] = ("AMR", "AP", "NSS")


def _nan_like(reference: Tensor) -> Tensor:
    return torch.full_like(reference, float("nan"))


def amr(
    sal: Float[Tensor, "*B H W"], mask: Bool[Tensor, "*B H W"]
) -> Float[Tensor, "*B"]:
    """Chance-normalised Attention Mass Ratio.

    ``amr = (sum(sal[mask]) / sum(sal)) / (mask.sum() / mask.numel())``

    1.0 = uniform attention (chance level); >1 concentrates inside mask;
    <1 outside. NaN for empty mask or zero-mass saliency.
    """
    sal = sal.float()
    total = sal.sum(dim=(-2, -1))
    mass_in = (sal * mask).sum(dim=(-2, -1))
    chance = mask.float().mean(dim=(-2, -1))
    score = mass_in / total.clamp(min=torch.finfo(sal.dtype).tiny)
    score = score / chance.clamp(min=torch.finfo(sal.dtype).tiny)
    return torch.where((total > 0) & (chance > 0), score, _nan_like(score))


def average_precision(
    sal: Float[Tensor, "*B H W"], mask: Bool[Tensor, "*B H W"]
) -> Float[Tensor, "*B"]:
    """Pixel-level Average Precision (matches sklearn's binary AP).

    Stable sort gives deterministic tie-breaking. NaN where the mask
    contains no positives.
    """
    s = sal.float().flatten(-2)
    y = mask.flatten(-2).to(s.dtype)
    order = torch.argsort(s, dim=-1, descending=True, stable=True)
    y_sorted = torch.gather(y, dim=-1, index=order)
    cum_tp = torch.cumsum(y_sorted, dim=-1)
    ranks = torch.arange(1, s.shape[-1] + 1, device=s.device, dtype=s.dtype)
    n_pos = y.sum(dim=-1)
    ap = (cum_tp / ranks * y_sorted).sum(dim=-1) / n_pos.clamp(min=1)
    return torch.where(n_pos > 0, ap, _nan_like(ap))


def nss(
    sal: Float[Tensor, "*B H W"], mask: Bool[Tensor, "*B H W"]
) -> Float[Tensor, "*B"]:
    """Normalised Scanpath Saliency: mean z-score inside the mask.

    NaN for an empty mask or constant saliency.
    """
    sal_flat = sal.float().flatten(-2)
    mask_flat = mask.flatten(-2).to(sal_flat.dtype)

    mean = sal_flat.mean(dim=-1, keepdim=True)
    std = sal_flat.std(dim=-1, unbiased=False, keepdim=True)
    z = (sal_flat - mean) / torch.where(std > 0, std, torch.ones_like(std))

    n = mask_flat.sum(dim=-1)
    score = (z * mask_flat).sum(dim=-1) / n.clamp(min=1)
    valid = (std.squeeze(-1) > 0) & (n > 0)
    return torch.where(valid, score, _nan_like(score))


def per_image_scores(
    saliency: SaliencyGrid,
    masks: list[Tensor],
    segment_ids: Int[Tensor, "B S M"],
    labels: Int[Tensor, "B S"],
    image_ids: Tensor | None = None,
) -> tuple[
    Float[Tensor, "B 3"],
    dict[str, int],
    tuple[int | None, list[int], list[int]] | None,
]:
    """Compute per-image (AMR, AP, NSS) by NaN-mean over supervised tokens.

    Mirrors :class:`finetune.criterion.base.Criterion` for the attention /
    token-mask construction so what is evaluated matches what was trained
    against. Rows for images with no supervised tokens stay NaN.

    Returns the score tensor, per-stage drop counts, and a sample of
    ``(image_id, caption_seg_ids, mask_seg_ids)`` from the first
    ``no_mask_overlap`` image in the batch (or ``None``) for diagnostic
    logging. ``image_id`` is ``None`` if ``image_ids`` was not provided.
    """
    device = labels.device
    batch_size = saliency.batch_size
    scores = torch.full((batch_size, len(METRIC_NAMES)), float("nan"), device=device)
    drops = {"no_supervised_tokens": 0, "no_segments": 0, "no_mask_overlap": 0}
    sample: tuple[int | None, list[int], list[int]] | None = None

    for b in range(batch_size):
        mask = masks[b].to(device)
        attn = saliency.maps_for_image(batch_idx=b, image_idx=0)

        seg_ids = segment_ids[b][labels[b] != -100]
        if seg_ids.shape[0] == 0:
            drops["no_supervised_tokens"] += 1
            continue
        attn = attn[-seg_ids.shape[0]:]

        has_segments = (seg_ids != -1).any(dim=1)
        if not has_segments.any():
            drops["no_segments"] += 1
            continue
        seg_ids = seg_ids[has_segments]
        attn = attn[has_segments]

        # Bilinear-upsample to annotation resolution, then softmax over the
        # spatial dim so AMR sees a proper probability distribution. Raw
        # ``saliency`` is logits (the training criterion softmaxes before
        # KL); without this, ``sum(sal)`` can be <= 0 and AMR collapses to
        # NaN. AP is rank-based and NSS is affine-invariant, so they're
        # unaffected by the transform.
        attn = F.interpolate(
            attn.unsqueeze(1).float(),
            size=tuple(mask.shape),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        attn = torch.softmax(attn.flatten(1), dim=1).view(-1, *mask.shape)

        valid = seg_ids != -1
        token_mask = (
            (mask[None, None] == seg_ids[:, :, None, None]) & valid[:, :, None, None]
        ).any(dim=1)

        # Caption-referenced ids may not exist in the panoptic mask (data
        # inconsistency); without overlap every per-token metric is NaN
        # and the image's row would stay NaN. Count it as its own bucket
        # so it doesn't get conflated with "kept".
        if not token_mask.any():
            drops["no_mask_overlap"] += 1
            if sample is None:
                img_id = int(image_ids[b]) if image_ids is not None else None
                sample = (
                    img_id,
                    seg_ids[valid].unique().cpu().tolist(),
                    mask.unique().cpu().tolist(),
                )
            continue

        per_token = torch.stack(
            [amr(attn, token_mask), average_precision(attn, token_mask),
             nss(attn, token_mask)], dim=-1
        )
        scores[b] = torch.nanmean(per_token, dim=0)

    return scores, drops, sample


def summarise(per_image: Float[Tensor, "N 3"]) -> dict[str, dict[str, float]]:
    """Reduce per-image scores into mean / median / std / count, NaN-aware."""
    finite = torch.isfinite(per_image)
    n = finite.sum(dim=0)
    mean = torch.nanmean(per_image, dim=0)
    median = torch.nanmedian(per_image, dim=0).values

    safe = torch.where(finite, per_image, torch.zeros_like(per_image))
    var = (safe**2).sum(dim=0) / n.clamp(min=1) - mean**2
    std = torch.where(n > 0, var.clamp(min=0).sqrt(), _nan_like(mean))

    return {
        name: {
            "mean": float(mean[i]),
            "median": float(median[i]),
            "std": float(std[i]),
            "n_images": int(n[i]),
        }
        for i, name in enumerate(METRIC_NAMES)
    }


def format_table(summary: dict[str, dict[str, float]]) -> str:
    """Render a fixed-width metric summary table."""
    header = f"| {'metric':<6} | {'mean':>9} | {'median':>9} | {'std':>9} | {'n':>6} |"
    sep = "|" + "-" * (len(header) - 2) + "|"
    lines = [sep, header, sep]
    for name in METRIC_NAMES:
        s = summary[name]
        lines.append(
            f"| {name:<6} | {s['mean']:>9.4f} | {s['median']:>9.4f} | "
            f"{s['std']:>9.4f} | {s['n_images']:>6d} |"
        )
    lines.append(sep)
    return "\n".join(lines)
