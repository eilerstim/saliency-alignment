import hydra
import lightning.pytorch as pl
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from omegaconf import DictConfig

from finetune.model import save_model
from finetune.module import FineTuner
from finetune.strategy import load_strategy
from vl_saliency import Saliency


def logger(cfg: DictConfig, output_dir: str) -> Logger:
    """Return a logger based on the configuration (either WandbLogger or CSVLogger)"""
    return (
        WandbLogger(save_dir=output_dir, **cfg.wandb, config=cfg)
        if cfg.wandb is not None
        else CSVLogger(save_dir=output_dir)
    )


def callbacks(cfg: DictConfig, output_dir: str) -> list[pl.Callback]:
    """Return a list of callbacks based on the configuration"""
    cb: list[pl.Callback] = [
        ModelCheckpoint(dirpath=output_dir, **cfg.callbacks.model_checkpoint)
    ]
    if cfg.wandb is None:  # wandb already tracks these metrics
        cb.extend([DeviceStatsMonitor(), LearningRateMonitor("step")])
    return cb


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def finetune(cfg: DictConfig):
    """Train the model based on the provided configuration."""

    pl.seed_everything(cfg.get("seed", 42))
    torch.set_float32_matmul_precision("high")

    hydra_wd = HydraConfig.get().runtime.output_dir

    module = FineTuner(cfg)
    trainer = pl.Trainer(
        logger=logger(cfg, hydra_wd),
        callbacks=callbacks(cfg, hydra_wd),
        strategy=load_strategy(cfg.strategy),
        plugins=[SLURMEnvironment()],
        **cfg.trainer,
    )

    with Saliency(module.model, backend="torch_eager"):
        trainer.fit(module)

    save_model(cfg, trainer, module)
