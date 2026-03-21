import logging
import os

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForImageTextToText, AutoProcessor

from vl_saliency import Saliency

from .lightning import FineTuner
from .strategy import load_lt_state, load_strategy

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def finetune(cfg: DictConfig):
    # Get local rank for distributed training
    rank = int(os.environ["SLURM_LOCALID"])

    # Log Hydra working directory
    hydra_wd = HydraConfig.get().runtime.output_dir
    if rank == 0:
        logger.info(f"Hydra working directory: {hydra_wd}")

    # Set seed for reproducibility
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    # Instantiate model and processor
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.name, dtype=torch.float32
    )
    processor = AutoProcessor.from_pretrained(cfg.model.name)

    # Freeze entire model except language model head
    model.requires_grad_(False)
    model.model.multi_modal_projector.requires_grad_(True)
    model.model.multi_modal_projector.to(torch.float32)

    # Prepare model for training
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Prepare loggers
    loggers = [
        CSVLogger(save_dir=f"{hydra_wd}/logs", name="training_logs"),
        WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            save_dir=hydra_wd,
            config=OmegaConf.to_container(cfg, resolve=True),
        ),
    ]

    # Instantiate fine-tuner and trainer
    fine_tuner = FineTuner(cfg, model, processor)
    trainer = L.Trainer(
        default_root_dir=hydra_wd,
        logger=loggers,
        strategy=load_strategy(cfg.strategy),
        plugins=[SLURMEnvironment()],
        **cfg.trainer,
    )

    # Fine-tuning
    with Saliency(model, backend="torch_eager"):
        trainer.fit(fine_tuner)

    # Gather and save model state dict on rank 0
    state = load_lt_state(cfg.strategy, trainer, model)

    # Save model and processor
    if rank == 0:
        save_dir = f"{cfg.checkpoint_dir}/{cfg.run_id}"
        model.save_pretrained(save_dir, state_dict=state)
        processor.save_pretrained(save_dir)
        logger.info(f"Model weights saved to {save_dir}")
