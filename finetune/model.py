from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
)


def build_model(
    cfg: DictConfig,
    lora_cfg: DictConfig,
) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Instantiate model and processor, optionally wrapped with LoRA adapters."""

    model = AutoModelForImageTextToText.from_pretrained(cfg.name, dtype=cfg.dtype)
    processor = AutoProcessor.from_pretrained(cfg.name)

    model.train()

    if lora_cfg.enabled:
        lora_kwargs = OmegaConf.to_container(lora_cfg, resolve=True)
        lora_kwargs.pop("enabled")
        model = get_peft_model(model, LoraConfig(**lora_kwargs))

        # PEFT inits LoRA in fp32; FSDP needs uniform dtype per flat param.
        dtype = getattr(torch, cfg.dtype)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(dtype)

        if cfg.gradient_checkpointing:
            # Gradient checkpointing requires the embedding output to track
            # grad; otherwise gradients never reach the LoRA adapters.
            model.enable_input_require_grads()
    else:
        if "all" in cfg.freeze:
            model.requires_grad_(False)
        else:
            for module in cfg.freeze:
                getattr(model.model, module).requires_grad_(False)

        for module in cfg.unfreeze:
            getattr(model.model, module).requires_grad_(True)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, processor


def load_pretrained(
    model_path: str,
    *,
    model_cls: Any = AutoModelForImageTextToText,
    **kwargs,
) -> PreTrainedModel:
    """Load a checkpoint that may be either a full HF model or a LoRA adapter.

    LoRA checkpoints (those containing ``adapter_config.json``) trigger
    loading the referenced base model, attaching the adapter, and merging
    it back so the returned model has the same module structure as a
    full-fine-tuned checkpoint. Extra kwargs are forwarded to the base
    model's ``from_pretrained``.
    """
    if (Path(model_path) / "adapter_config.json").is_file():
        peft_config = PeftConfig.from_pretrained(model_path)
        base = model_cls.from_pretrained(
            peft_config.base_model_name_or_path, **kwargs
        )
        return PeftModel.from_pretrained(base, model_path).merge_and_unload()
    return model_cls.from_pretrained(model_path, **kwargs)
