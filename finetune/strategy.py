from functools import partial

import lightning as L
import torch
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import PreTrainedModel


def load_strategy(strategy: str):
    if strategy == "fsdp":
        # Wrap large modules (Transformer blocks)
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(1e6),
        )

        # FSDP strategy with mixed precision
        return FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            use_orig_params=True,
            sync_module_states=True,
            cpu_offload=False,
            limit_all_gathers=True,
        )
    else:
        return strategy


def load_lt_state(strategy: str, trainer: L.Trainer, model: PreTrainedModel):
    if strategy == "fsdp":
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            lt_state = trainer.strategy.model.state_dict()  # type: ignore

    else:
        lt_state = trainer.strategy.model.state_dict()  # type: ignore

    # strip the Lightning prefix
    hf_state = {
        k.removeprefix("model."): v
        for k, v in lt_state.items()
        if k.startswith("model.")
    }

    return hf_state
