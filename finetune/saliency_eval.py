"""Stand-alone entry point for saliency-alignment evaluation.

Reuses the training Hydra config (``configs/config.yaml``) so the
checkpoint location, model dtype, dataset paths and dataloader kwargs
all stay defined in a single place. Set the same ``SLURM_JOB_NAME`` /
``MODEL`` / ``CRITERION`` / ``LAMBDA`` / ``SLURM_JOB_ID`` env vars the
launch script uses for training and ``run_id`` resolves to the same
checkpoint directory.

Usage (multi-GPU via SLURM srun):

    srun python -m finetune.saliency_eval                # ${run_id}
    srun python -m finetune.saliency_eval model_path=... # explicit override

Each rank loads the checkpoint on its local GPU, processes a disjoint
shard of the validation set (DDP), and rank 0 logs the per-model AMR /
AP / NSS table plus a JSON summary alongside the Hydra outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from omegaconf import DictConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

from vl_saliency import Saliency

from .eval import build_eval_dataloader, evaluate, format_results_table

logger = logging.getLogger(__name__)


def _resolve_model_path(cfg: DictConfig) -> str:
    """Allow ``model_path=...`` to override the default checkpoint
    location, otherwise fall back to ``${checkpoint_dir}/${run_id}``."""
    model_path = cfg.get("model_path", None)
    if model_path:
        return str(model_path)
    return f"{cfg.checkpoint_dir}/{cfg.run_id}"


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    fabric = L.Fabric(
        accelerator="auto",
        strategy="ddp",
        precision="bf16-mixed",
        plugins=[SLURMEnvironment()],
    )
    fabric.launch()

    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    hydra_wd = HydraConfig.get().runtime.output_dir
    model_path = _resolve_model_path(cfg)
    if fabric.global_rank == 0:
        logger.info("Hydra working directory: %s", hydra_wd)
        logger.info(
            "Loading checkpoint from %s on %d ranks",
            model_path,
            fabric.world_size,
        )

    # Eager attention is required for vl_saliency's hook-based extractor.
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=cfg.model.dtype,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    model.eval()
    model.requires_grad_(False)

    dataloader = build_eval_dataloader(
        cfg, processor, world_size=fabric.world_size, rank=fabric.global_rank
    )

    if fabric.global_rank == 0:
        logger.info(
            "Validation set: %d images total, %d per rank, batch_size=%d",
            len(dataloader.dataset),  # type: ignore[arg-type]
            len(dataloader.sampler),  # type: ignore[arg-type]
            dataloader.batch_size,
        )

    # Patch the bare model BEFORE Fabric wraps it: ``Saliency`` rebinds
    # ``model.forward``, and DDP only sees the patched version if the
    # patch happens before wrapping (otherwise DDP keeps calling the
    # unpatched inner forward).
    with Saliency(model, backend="torch_eager"):
        model = fabric.setup_module(model, move_to_device=True)
        result = evaluate(model, dataloader, fabric=fabric)

    if fabric.global_rank == 0:
        table = format_results_table(result)
        logger.info("\n=== Saliency alignment metrics ===\n%s", table)

        out_path = Path(hydra_wd) / "saliency_eval.json"
        out_path.write_text(
            json.dumps(
                {
                    "model_path": model_path,
                    "n_tokens": result.n_tokens,
                    "per_model": result.per_model,
                },
                indent=2,
            )
        )
        logger.info("Wrote per-model summary to %s", out_path)


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
