"""Stand-alone entry point for saliency-alignment evaluation.

Reuses the training Hydra config (``configs/config.yaml``) so the
checkpoint location, model dtype, dataset paths and dataloader kwargs
all stay defined in a single place. Set the same ``SLURM_JOB_NAME`` /
``MODEL`` / ``CRITERION`` / ``LAMBDA`` / ``SLURM_JOB_ID`` environment
variables the launch script uses for training and ``run_id`` will
resolve to the same checkpoint directory.

Usage:

    uv run -m finetune.saliency_eval                    # uses ${run_id}
    uv run -m finetune.saliency_eval model_path=...     # explicit override

The script loads the saved model, runs inference on the deterministic
validation slice (``[n_train:]`` of the COCONut images sorted by id)
and prints a table of per-model AMR / AP / NSS scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

from .eval import build_eval_dataloader, evaluate, format_results_table

logger = logging.getLogger(__name__)


def _resolve_model_path(cfg: DictConfig) -> str:
    """Allow ``model_path=...`` to override the default checkpoint
    location, otherwise fall back to ``${checkpoint_dir}/${run_id}``
    (where ``finetune/finetune.py`` saves)."""
    model_path = cfg.get("model_path", None)
    if model_path:
        return str(model_path)
    return f"{cfg.checkpoint_dir}/{cfg.run_id}"


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    hydra_wd = HydraConfig.get().runtime.output_dir
    logger.info("Hydra working directory: %s", hydra_wd)

    model_path = _resolve_model_path(cfg)
    logger.info("Loading checkpoint from %s", model_path)

    # Eager attention is required for vl_saliency's hook-based extractor.
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=cfg.model.dtype,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.requires_grad_(False)

    dataloader = build_eval_dataloader(cfg, processor)
    logger.info(
        "Evaluating on %d validation images (batch_size=%d)",
        len(dataloader.dataset),  # type: ignore[arg-type]
        dataloader.batch_size,
    )

    result = evaluate(model, dataloader, device=device)

    table = format_results_table(result)
    logger.info("\n=== Saliency alignment metrics ===\n%s", table)

    # Persist machine-readable results next to the Hydra log so multiple
    # checkpoints can be compared post-hoc without re-parsing the table.
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
