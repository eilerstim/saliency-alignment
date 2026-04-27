"""Stand-alone entry point for saliency-alignment evaluation.

Usage:

    uv run -m finetune.saliency_eval model_path=models/<run_id>

The script loads a model previously saved by ``finetune/finetune.py``,
runs inference on the deterministic validation slice (``n_val`` images
following the first ``n_train`` training images), and prints a table of
per-model AMR / AP / NSS scores.
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
    """Allow either an explicit ``model_path=...`` override or fall back
    to ``${checkpoint_dir}/${run_id}`` (the default training output)."""
    if getattr(cfg, "model_path", None):
        return str(cfg.model_path)
    return f"{cfg.checkpoint_dir}/{cfg.run_id}"


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="saliency_eval"
)
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
