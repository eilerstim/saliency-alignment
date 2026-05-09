"""Standalone attention-alignment evaluation.

Loads a saved checkpoint, runs the validation set under the same
``Saliency`` context as training, computes per-image alignment metrics
across all ranks (DDP via Lightning Fabric), and writes a summary table
plus per-image scores to the output directory.

Splitting this out of the training pipeline avoids re-entering the
FSDP-wrapped model after fit (Lightning leaves inner units sharded),
and lets metrics be re-computed against any saved checkpoint.

Reuses ``configs/config.yaml`` so checkpoint location, model dtype,
dataset paths and dataloader kwargs all stay defined in one place.
The same env vars as training resolve ``run_id`` to the matching
checkpoint directory.

Usage (multi-GPU via SLURM srun, one rank per GPU):

    srun python -m align_eval.eval                # ${checkpoint_dir}/${run_id}
    srun python -m align_eval.eval model_path=... # explicit override
"""

import json
import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForImageTextToText, AutoProcessor

from vl_saliency import Saliency

from .metrics import METRIC_NAMES, format_table, per_image_scores, summarise

logger = logging.getLogger(__name__)


def _build_dataloader(
    cfg: DictConfig, processor, *, world_size: int, rank: int
) -> DataLoader:
    """Validation dataloader sharded across ranks.

    ``DistributedSampler(drop_last=True)`` keeps per-rank lengths equal so
    the post-loop ``all_gather`` doesn't need padding. At most
    ``world_size - 1`` images are dropped from the tail (negligible).
    """
    dataset = instantiate(cfg.data.dataset, cfg.data, split="validation")
    collate_fn = instantiate(cfg.data.eval_collator, processor=processor)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
    )

    dl_kwargs = OmegaConf.to_container(cfg.dataloader, resolve=True)
    assert isinstance(dl_kwargs, dict)
    dl_kwargs.pop("shuffle", None)
    dl_kwargs["drop_last"] = True

    return DataLoader(dataset, sampler=sampler, collate_fn=collate_fn, **dl_kwargs)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    fabric = L.Fabric(
        accelerator="auto",
        strategy="ddp",
        precision="bf16-mixed",
        plugins=[SLURMEnvironment()],
    )
    fabric.launch()

    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    model_path = cfg.get("model_path") or f"{cfg.checkpoint_dir}/{cfg.run_id}"
    hydra_wd = HydraConfig.get().runtime.output_dir
    if fabric.global_rank == 0:
        logger.info(
            "Loading checkpoint: %s (world_size=%d)",
            model_path, fabric.world_size,
        )

    # Eager attention is required for vl_saliency's hook-based extractor.
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=cfg.model.dtype, attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    model.requires_grad_(False)

    dataloader = _build_dataloader(
        cfg, processor, world_size=fabric.world_size, rank=fabric.global_rank
    )

    # Patch the bare model BEFORE Fabric wraps it: ``Saliency`` rebinds
    # ``model.forward``, and DDP only sees the patched version if the
    # patch happens before wrapping.
    with Saliency(model, backend="torch_eager"), torch.no_grad():
        model = fabric.setup_module(model, move_to_device=True)

        # Pad per-batch scores to ``batch_size`` so per-rank totals stay in
        # lock-step for ``all_gather``. Rows are NaN when the collator
        # dropped an example (empty caption) or the whole batch.
        bs = dataloader.batch_size or 0
        nan_pad = lambda n: torch.full(  # noqa: E731
            (n, len(METRIC_NAMES)), float("nan"), device=fabric.device
        )

        local: list[torch.Tensor] = []
        for i, batch in enumerate(dataloader):
            if batch is None:
                local.append(nan_pad(bs))
                continue
            batch = {k: v.to(fabric.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(**batch, return_dict=True)
            scores = per_image_scores(
                outputs.saliency, batch["masks"],
                batch["segment_ids"], batch["labels"],
            )
            if scores.shape[0] < bs:
                scores = torch.cat([scores, nan_pad(bs - scores.shape[0])], dim=0)
            local.append(scores)
            if fabric.global_rank == 0 and (i + 1) % 25 == 0:
                logger.info("Evaluated %d batches", i + 1)

    local_tensor = torch.cat(local, dim=0) if local else torch.empty(
        (0, len(METRIC_NAMES)), device=fabric.device
    )
    gathered = fabric.all_gather(local_tensor)
    per_image = gathered.flatten(0, 1) if gathered.dim() == 3 else gathered

    if fabric.global_rank == 0:
        summary = summarise(per_image.cpu())
        logger.info("\n=== Attention alignment metrics ===\n%s", format_table(summary))

        out_dir = Path(hydra_wd)
        (out_dir / "alignment_summary.json").write_text(json.dumps(
            {"checkpoint": model_path, "summary": summary}, indent=2
        ))
        torch.save(per_image.cpu(), out_dir / "alignment_per_image.pt")
        logger.info("Wrote alignment metrics to %s", out_dir)


if __name__ == "__main__":
    evaluate()
