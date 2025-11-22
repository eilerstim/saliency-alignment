import logging

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from transformers import (  # LlavaForConditionalGeneration
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .lightning import FineTuner

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def finetune(cfg: DictConfig):
    # Log Hydra working directory
    hydra_wd = HydraConfig.get().runtime.output_dir
    logger.info(f"Hydra working directory: {hydra_wd}")

    # Set seed for reproducibility
    L.seed_everything(cfg.seed)

    # Instantiate model and processor
    model = AutoModelForImageTextToText.from_pretrained(cfg.model.name)
    processor = AutoProcessor.from_pretrained(cfg.model.name)

    # Prepare callbacks
    callbacks = [
        # Checkpointing
        ModelCheckpoint(
            dirpath=hydra_wd + "/checkpoints",
            filename="{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            every_n_epochs=cfg.checkpoint_every_n_epochs,
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval="step"),
        # Monitor GPU/CPU stats
        DeviceStatsMonitor(cpu_stats=True),
    ]

    # Prepare loggers
    loggers = [
        CSVLogger(save_dir=f"{hydra_wd}/logs", name="training_logs"),
        WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            save_dir=hydra_wd,
            config=OmegaConf.to_container(cfg, resolve=True),
        ),
    ]

    # Instantiate fine-tuner and trainer
    fine_tuner = FineTuner(cfg, model, processor)
    trainer = L.Trainer(
        default_root_dir=hydra_wd, logger=loggers, callbacks=callbacks, **cfg.trainer
    )

    # Fine-tuning
    trainer.fit(fine_tuner)
