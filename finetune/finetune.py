import logging
import os
from functools import partial

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import (  # LlavaForConditionalGeneration
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .lightning import FineTuner

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def finetune(cfg: DictConfig):
    # Get local rank for distributed training
    rank = int(os.environ["SLURM_LOCALID"])
    logger.info(f"Local rank: {rank}")

    # Log Hydra working directory
    hydra_wd = HydraConfig.get().runtime.output_dir
    if rank == 0:
        logger.info(f"Hydra working directory: {hydra_wd}")

    # Set seed for reproducibility
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    # Instantiate model and processor
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.name, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(cfg.model.name)

    # Prepare model for training
    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

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

    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=int(2e7),
    )

    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,  # wrap large modules (Transformer blocks)
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        cpu_offload=False,  # do NOT offload, A100s don't need it
        activation_checkpointing_policy=auto_wrap_policy,  # use same policy as auto_wrap
        limit_all_gathers=True,
    )

    # Instantiate fine-tuner and trainer
    fine_tuner = FineTuner(cfg, model, processor)
    trainer = L.Trainer(
        default_root_dir=hydra_wd,
        logger=loggers,
        strategy=strategy,
        plugins=[SLURMEnvironment()],
        **cfg.trainer,
    )

    # Fine-tuning
    trainer.fit(fine_tuner)

    if rank == 0:
        # Save the final checkpoint
        model.save_pretrained(f"{hydra_wd}/weights")
