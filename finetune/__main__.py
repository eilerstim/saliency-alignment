import logging

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from transformers import AutoModelForCausalLM

from .data import load_dataloader
from .lightning import FineTuner

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Log Hydra working directory
    hydra_wd = HydraConfig.get().runtime.output_dir
    logger.info(f"Hydra working directory: {hydra_wd}")

    # Set seed for reproducibility
    L.seed_everything(cfg.seed)

    # Instantiate model
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

    # Load dataset
    train_loader = load_dataloader(cfg.data, split="train")
    val_loader = load_dataloader(cfg.data, split="validation")

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
    fine_tuner = FineTuner(cfg, model)
    trainer = L.Trainer(
        default_root_dir=hydra_wd, logger=loggers, callbacks=callbacks, **cfg.trainer
    )

    # Fine-tuning
    trainer.fit(fine_tuner, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()  # type: ignore
