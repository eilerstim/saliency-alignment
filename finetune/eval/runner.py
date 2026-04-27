"""Multi-GPU saliency-alignment evaluator.

Each rank loads its own copy of the checkpoint, processes a disjoint
shard of the validation dataloader (via ``DistributedSampler``) and
produces a per-image score tensor. The shards are all-gathered at the
end so rank 0 can print the per-model summary.

Metrics are computed batched on the GPU, vectorised across all
supervised tokens within an image (one ``argsort`` for AP, etc.). The
mask-building logic mirrors :class:`finetune.criterion.base.Criterion`
exactly so what we evaluate is what we trained against.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import lightning as L
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from transformers import ProcessorMixin

from vl_saliency.maps import SaliencyGrid

from .metrics import (
    amr_normalised,
    average_precision,
    nss,
    per_image_aggregate,
    per_model_aggregate,
)

logger = logging.getLogger(__name__)


METRIC_NAMES: tuple[str, ...] = ("AMR", "AP", "NSS")
N_METRICS = len(METRIC_NAMES)


# ---------------------------------------------------------------------------
# Per-image extraction (batched across the image's supervised tokens)
# ---------------------------------------------------------------------------


def _per_image_tokens(
    saliency: SaliencyGrid,
    labels: torch.Tensor,
    segment_ids: torch.Tensor,
    mask: torch.Tensor,
    b: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build ``(attn, token_mask)`` of shape ``(n_tokens, H, W)`` for image
    ``b`` in the batch, or ``None`` if it has no supervised tokens.

    Mirrors :meth:`Criterion.__call__`: trailing-``gen_len`` slice of the
    saliency grid, ``has_segments`` filter, bilinear upsample and softmax.
    """
    attn = saliency.maps_for_image(batch_idx=b, image_idx=0)  # (T, h, w)

    gen_ids = labels[b] != -100
    seg_ids = segment_ids[b][gen_ids]
    gen_len = seg_ids.shape[0]
    if gen_len == 0:
        return None

    attn = attn[-gen_len:]

    has_segments = (seg_ids != -1).any(dim=1)
    if not has_segments.any():
        return None

    seg_ids = seg_ids[has_segments].to(mask.device)
    attn = attn[has_segments]

    attn = F.interpolate(
        attn.unsqueeze(1).float(),
        size=tuple(mask.shape),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    n_tokens = attn.shape[0]
    attn = torch.softmax(attn.flatten(1), dim=1).view(n_tokens, *mask.shape)

    valid_seg = seg_ids != -1
    seg_match = (mask[None, None] == seg_ids[:, :, None, None]) & valid_seg[
        :, :, None, None
    ]
    token_mask = seg_match.any(dim=1)  # (n_tokens, H, W)

    return attn, token_mask


def _evaluate_batch(
    batch: dict, saliency: SaliencyGrid, *, device: torch.device
) -> tuple[torch.Tensor, int]:
    """Compute per-image scores for one batch.

    Returns ``(scores, n_tokens)`` where ``scores`` has shape
    ``(batch_size, N_METRICS)`` (NaN rows for images that contributed no
    supervised tokens) and ``n_tokens`` is the number of supervised
    tokens that fed the metrics.
    """
    labels = batch["labels"]
    segment_ids = batch["segment_ids"]
    masks: list[torch.Tensor] = batch["masks"]

    batch_size = saliency.batch_size
    scores = torch.full(
        (batch_size, N_METRICS), float("nan"), device=device, dtype=torch.float32
    )

    n_tokens = 0
    for b in range(batch_size):
        out = _per_image_tokens(
            saliency, labels, segment_ids, masks[b].to(device), b
        )
        if out is None:
            continue
        attn, token_mask = out
        n_tokens += attn.shape[0]

        # Stack per-token scores once; columns are (AMR, AP, NSS).
        per_token = torch.stack(
            [
                amr_normalised(attn, token_mask),
                average_precision(attn, token_mask),
                nss(attn, token_mask),
            ],
            dim=-1,
        )  # (n_tokens, N_METRICS)
        scores[b] = per_image_aggregate(per_token)

    return scores, n_tokens


# ---------------------------------------------------------------------------
# Distributed evaluation loop
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Per-image score matrix plus per-model aggregates.

    ``per_image_scores`` is a CPU tensor of shape ``(N_images, N_METRICS)``
    in metric order (``AMR, AP, NSS``). It already contains NaN entries
    for images that the loop skipped (no supervised tokens, or DDP
    padding) so callers can ``torch.nanmean`` it freely.
    """

    per_image_scores: torch.Tensor
    per_model: dict[str, dict[str, float]]
    n_tokens: int


def evaluate(
    model: torch.nn.Module,
    dataloader: Iterable[dict],
    *,
    fabric: L.Fabric,
    log_every: int = 25,
) -> EvaluationResult:
    """Run evaluation across all ranks and aggregate to per-model scores.

    The caller is responsible for wrapping ``model`` in a ``Saliency``
    context *before* DDP wrapping (otherwise the forward-method patch
    lands on the outer wrapper while DDP still calls the unpatched inner
    forward). ``model`` should already be set up via ``fabric.setup_module``
    and ``dataloader`` produced by :func:`build_eval_dataloader`.
    """
    device = fabric.device

    local_scores: list[torch.Tensor] = []
    n_tokens_local = 0

    # ``no_grad`` rather than ``inference_mode`` so the QK accumulator's
    # cached tensors remain usable after the forward pass returns.
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Even for batches the collator dropped (returned ``None``),
            # emit a NaN row per expected sample so per-rank tensor lengths
            # stay in lock-step for the post-loop ``all_gather``.
            if batch is None:
                bs = dataloader.batch_size or 0
                scores = torch.full(
                    (bs, N_METRICS), float("nan"),
                    device=device, dtype=torch.float32,
                )
                local_scores.append(scores)
                continue

            tensor_batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            outputs = model(**tensor_batch, return_dict=True)
            saliency: SaliencyGrid = outputs.saliency

            scores, n_t = _evaluate_batch(
                tensor_batch, saliency, device=device
            )
            local_scores.append(scores)
            n_tokens_local += n_t

            if fabric.global_rank == 0 and (batch_idx + 1) % log_every == 0:
                logger.info(
                    "Rank 0: evaluated %d batches (%d local tokens)",
                    batch_idx + 1,
                    n_tokens_local,
                )

    local_tensor = (
        torch.cat(local_scores, dim=0)
        if local_scores
        else torch.empty((0, N_METRICS), device=device)
    )

    # Sum-reduce token counts; gather per-image score rows across ranks.
    n_tokens_total = int(
        fabric.all_reduce(
            torch.tensor(n_tokens_local, device=device), reduce_op="sum"
        ).item()
    )

    # ``DistributedSampler(drop_last=True)`` makes per-rank counts equal,
    # so a plain stack-and-flatten is sufficient.
    gathered = fabric.all_gather(local_tensor)
    if gathered.dim() == 3:  # (world_size, N_local, M)
        per_image_scores = gathered.flatten(0, 1)
    else:  # single-rank fallback: (N_local, M)
        per_image_scores = gathered

    per_image_cpu = per_image_scores.detach().to("cpu")
    aggregates = per_model_aggregate(per_image_cpu)

    per_model = {
        name: {
            "mean": float(aggregates["mean"][i]),
            "median": float(aggregates["median"][i]),
            "std": float(aggregates["std"][i]),
            "n_images": int(aggregates["n_images"][i]),
        }
        for i, name in enumerate(METRIC_NAMES)
    }

    return EvaluationResult(
        per_image_scores=per_image_cpu,
        per_model=per_model,
        n_tokens=n_tokens_total,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_results_table(result: EvaluationResult) -> str:
    """Fixed-width per-metric table with mean / median / std / image count."""
    header = (
        f"| {'metric':<6} | {'mean':>9} | {'median':>9} | "
        f"{'std':>9} | {'n_images':>8} |"
    )
    sep = "|" + "-" * (len(header) - 2) + "|"
    lines = [sep, header, sep]
    for metric in METRIC_NAMES:
        s = result.per_model[metric]
        lines.append(
            f"| {metric:<6} | "
            f"{s['mean']:>9.4f} | {s['median']:>9.4f} | "
            f"{s['std']:>9.4f} | {s['n_images']:>8d} |"
        )
    lines.append(sep)
    lines.append(f"Tokens evaluated: {result.n_tokens}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distributed-friendly dataloader
# ---------------------------------------------------------------------------


def build_eval_dataloader(
    cfg: DictConfig, processor: ProcessorMixin, *, world_size: int, rank: int
) -> DataLoader:
    """Validation dataloader sharded across ranks.

    ``DistributedSampler(drop_last=True)`` keeps per-rank lengths equal so
    the post-loop ``all_gather`` doesn't need padding. At most
    ``world_size - 1`` images are dropped from the tail of the validation
    set (≪ 0.1 % of a 10k-image split — negligible) instead of being
    duplicated and double-counted.
    """
    dataset = instantiate(cfg.data.dataset, cfg.data, split="validation")
    collate_fn = instantiate(cfg.data.eval_collator, processor=processor)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )

    dl_kwargs = OmegaConf.to_container(
        getattr(cfg, "dataloader", {}), resolve=True
    )
    assert isinstance(dl_kwargs, dict)
    dl_kwargs.pop("shuffle", None)
    # Keep ``drop_last=True`` (the training default) so every batch is
    # full-sized: the post-loop ``all_gather`` then doesn't have to worry
    # about a ragged partial last batch. At most ``batch_size - 1`` images
    # per rank are dropped — << 1 % of the validation slice.
    dl_kwargs["drop_last"] = True

    return DataLoader(dataset, sampler=sampler, collate_fn=collate_fn, **dl_kwargs)
