"""Evaluate the saliency alignment of a saved fine-tuned model.

The runner replays the same per-token mask-building logic used by
``finetune.criterion.Criterion`` so the reported metrics correspond
exactly to the supervision signal the model was trained against.

Aggregation
-----------
The pipeline reports per-model scores, computed bottom-up:

    per-token score   →  per-image score (mean over the image's tokens)
                      →  per-model score (mean over images)

NaNs (tokens / images that were skipped because the metric is undefined,
e.g. empty masks for AP) are excluded from each level so they don't bias
the aggregate towards zero.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, ProcessorMixin

from vl_saliency import Saliency
from vl_saliency.maps import SaliencyGrid

from .metrics import (
    aggregate_per_image,
    aggregate_per_model,
    amr_normalised,
    average_precision,
    nss,
)

logger = logging.getLogger(__name__)


METRIC_NAMES: tuple[str, ...] = ("AMR", "AP", "NSS")


@dataclass(frozen=True)
class TokenSample:
    """One supervised generated token: a normalised attention map plus its
    binary reference region. Both tensors live at the same resolution
    (the panoptic mask's), as in the training criterion."""

    saliency: torch.Tensor  # (H, W) probability-like
    mask: torch.Tensor      # (H, W) bool


# ---------------------------------------------------------------------------
# Per-batch token extraction
# ---------------------------------------------------------------------------


def _iter_tokens_from_batch(
    batch: dict,
    saliency: SaliencyGrid,
    *,
    device: torch.device,
) -> Iterator[tuple[int, TokenSample]]:
    """Yield ``(image_index_in_batch, TokenSample)`` for every supervised
    token in ``batch``.

    Mirrors :meth:`finetune.criterion.base.Criterion.__call__`:
        * select tokens whose label is ``!= -100`` (i.e. caption tokens),
        * align them with the trailing ``gen_len`` rows of the saliency grid,
        * keep only tokens that reference at least one segment id,
        * upsample patch-level saliency to mask resolution and softmax it.
    """
    labels: torch.Tensor = batch["labels"]
    segment_ids: torch.Tensor = batch["segment_ids"]
    masks: list[torch.Tensor] = batch["masks"]

    for b in range(saliency.batch_size):
        mask = masks[b].to(device)
        attn = saliency.maps_for_image(batch_idx=b, image_idx=0)  # (T, h, w)

        gen_ids = labels[b] != -100
        seg_ids = segment_ids[b][gen_ids]  # (gen_len, max_segments)
        gen_len = seg_ids.shape[0]
        if gen_len == 0:
            continue

        attn = attn[-gen_len:]

        has_segments = (seg_ids != -1).any(dim=1)
        if not has_segments.any():
            continue

        seg_ids = seg_ids[has_segments].to(device)
        attn = attn[has_segments]

        # Upsample patch-level attention to mask resolution.
        attn = F.interpolate(
            attn.unsqueeze(1).float(),
            size=tuple(mask.shape),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Convert each token's attention map to a probability distribution
        # over pixels, matching the criterion's normalisation.
        n_tokens = attn.shape[0]
        attn = torch.softmax(attn.flatten(1), dim=1).view(n_tokens, *mask.shape)

        # Build per-token boolean mask: True for any pixel whose panoptic id
        # matches one of the token's referenced segments.
        valid_seg = seg_ids != -1  # (n_tokens, max_segments)
        seg_match = (mask[None, None] == seg_ids[:, :, None, None]) & valid_seg[
            :, :, None, None
        ]
        token_mask = seg_match.any(dim=1)  # (n_tokens, H, W)

        for t in range(n_tokens):
            yield b, TokenSample(saliency=attn[t], mask=token_mask[t])


# ---------------------------------------------------------------------------
# Top-level evaluation loop
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Per-metric aggregate plus the underlying per-image score arrays
    (handy for downstream plotting / significance testing)."""

    per_image_scores: dict[str, list[float]]
    per_model: dict[str, dict[str, float]]
    n_tokens: int


def evaluate(
    model: PreTrainedModel,
    dataloader: Iterable[dict],
    *,
    device: torch.device,
    saliency_kwargs: dict | None = None,
) -> EvaluationResult:
    """Run the full evaluation on ``dataloader`` and aggregate to per-model
    scores. ``model`` should already be in eval mode and on ``device``."""
    saliency_kwargs = {"backend": "torch_eager", **(saliency_kwargs or {})}

    per_image: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    n_tokens_total = 0

    # ``no_grad`` (rather than ``inference_mode``) so the tensors stored
    # inside the saliency accumulator remain regular autograd-aware tensors.
    # Inference-mode tensors can't be reused outside the context, which the
    # saliency grid relies on after the forward pass returns.
    with torch.no_grad(), Saliency(model, **saliency_kwargs):
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:  # collator returned no valid examples
                continue

            tensor_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(**tensor_batch, return_dict=True)
            saliency: SaliencyGrid = outputs.saliency

            # Group per-token scores by their image-in-batch so we can
            # average per image immediately and keep memory bounded.
            buckets: dict[int, dict[str, list[float]]] = {}
            for b, sample in _iter_tokens_from_batch(
                tensor_batch, saliency, device=device
            ):
                bucket = buckets.setdefault(
                    b, {m: [] for m in METRIC_NAMES}
                )
                bucket["AMR"].append(amr_normalised(sample.saliency, sample.mask))
                bucket["AP"].append(average_precision(sample.saliency, sample.mask))
                bucket["NSS"].append(nss(sample.saliency, sample.mask))
                n_tokens_total += 1

            for bucket in buckets.values():
                for metric in METRIC_NAMES:
                    per_image[metric].append(aggregate_per_image(bucket[metric]))

            if (batch_idx + 1) % 25 == 0:
                logger.info(
                    "Evaluated %d batches  (%d images, %d supervised tokens so far)",
                    batch_idx + 1,
                    len(per_image["AMR"]),
                    n_tokens_total,
                )

    per_model = {m: aggregate_per_model(per_image[m]) for m in METRIC_NAMES}
    return EvaluationResult(
        per_image_scores=per_image,
        per_model=per_model,
        n_tokens=n_tokens_total,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_results_table(result: EvaluationResult) -> str:
    """Render a fixed-width table summarising per-model metrics.

    Columns: metric name, mean, median, std (population), and the number
    of images that contributed a finite score.
    """
    header = (
        f"| {'metric':<6} | {'mean':>9} | {'median':>9} | "
        f"{'std':>9} | {'n_images':>8} |"
    )
    sep = "|" + "-" * (len(header) - 2) + "|"
    lines = [sep, header, sep]
    for metric in METRIC_NAMES:
        stats = result.per_model[metric]
        lines.append(
            f"| {metric:<6} | "
            f"{stats['mean']:>9.4f} | "
            f"{stats['median']:>9.4f} | "
            f"{stats['std']:>9.4f} | "
            f"{stats['n_images']:>8d} |"
        )
    lines.append(sep)
    lines.append(f"Tokens evaluated: {result.n_tokens}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataloader plumbing for stand-alone eval
# ---------------------------------------------------------------------------


def build_eval_dataloader(cfg: DictConfig, processor: ProcessorMixin) -> DataLoader:
    """Construct the validation dataloader the same way ``FineTuner`` does,
    but always with ``shuffle=False`` and ``drop_last=False`` so every image
    in the validation slice is evaluated exactly once."""
    dataset = instantiate(cfg.data.dataset, cfg.data, split="validation")
    collate_fn = instantiate(cfg.data.eval_collator, processor=processor)

    dl_kwargs = OmegaConf.to_container(
        getattr(cfg, "dataloader", {}), resolve=True
    )
    assert isinstance(dl_kwargs, dict)
    dl_kwargs["shuffle"] = False
    dl_kwargs["drop_last"] = False

    return DataLoader(dataset, collate_fn=collate_fn, **dl_kwargs)
