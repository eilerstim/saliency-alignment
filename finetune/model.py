import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
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

    # Eager attention is required for vl_saliency's hook-based extractor (sdpa /
    # flash do not expose per-head attention weights). Models that default to
    # sdpa (e.g. Qwen2.5-VL, Gemma-3) would otherwise yield an empty saliency
    # map and silently collapse the alignment loss to zero.
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.name, dtype=cfg.dtype, attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(cfg.name)

    # Some models (e.g. Qwen2.5-VL) leave pad_token_id unset on the top-level
    # config; vl_saliency needs it to mask padding tokens. Backfill it from the
    # tokenizer (falling back to eos) so the saliency hook can infer it.
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = (
            processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        )

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
