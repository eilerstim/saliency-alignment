import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    CLIPVisionModel,
    LlavaConfig,
    LlavaForConditionalGeneration,
    PreTrainedModel,
    ProcessorMixin,
)

from finetune.strategy import load_lt_state

if TYPE_CHECKING:
    from finetune.module import FineTuner


def build_model(
    cfg: DictConfig,
    lora_cfg: DictConfig,
) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Instantiate model and processor, optionally wrapped with LoRA adapters."""
    model: PreTrainedModel

    if hasattr(cfg, "instruction_tuned") and not cfg.instruction_tuned:
        config = LlavaConfig.from_pretrained(cfg.name)
        model = LlavaForConditionalGeneration(config)
        processor = AutoProcessor.from_pretrained(cfg.base_arch)

        llm = AutoModelForCausalLM.from_pretrained(
            cfg.base_lm,
            dtype=cfg.dtype,
        )

        llm.resize_token_embeddings(len(processor.tokenizer))
        model.model.language_model.resize_token_embeddings(len(processor.tokenizer))
        model.resize_token_embeddings(len(processor.tokenizer))

        model.model.language_model.load_state_dict(
            llm.model.state_dict(),
            strict=False,
        )

        model.lm_head.load_state_dict(
            llm.lm_head.state_dict(),
            strict=False,
        )

        model.model.vision_tower = CLIPVisionModel.from_pretrained(
            cfg.base_clip,
            dtype=cfg.dtype,
        )
        projector_path = hf_hub_download(
            repo_id=cfg.name,
            filename="mm_projector.bin",
        )

        projector_state = torch.load(projector_path, map_location="cpu")

        rename = {
            "model.mm_projector.0.weight": "linear_1.weight",
            "model.mm_projector.0.bias": "linear_1.bias",
            "model.mm_projector.2.weight": "linear_2.weight",
            "model.mm_projector.2.bias": "linear_2.bias",
        }

        clean_state = {rename[k]: v for k, v in projector_state.items() if k in rename}

        model.model.multi_modal_projector.load_state_dict(clean_state)

        model.to(dtype=torch.bfloat16)

        model.tie_weights()

    else:
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


def save_model(cfg: DictConfig, trainer: pl.Trainer, module: "FineTuner"):
    """Save the model and processor to the specified directory."""
    # Gather and save model state dict on rank 0
    state = load_lt_state(cfg.strategy, trainer, module.model)

    rank = int(os.environ["SLURM_LOCALID"])
    if rank == 0:
        save_dir = f"{cfg.checkpoint_dir}/{cfg.run_id}"
        module.model.save_pretrained(save_dir, state_dict=state)
        module.processor.save_pretrained(save_dir)

        # Fix tokenizer_class for vLLM compatibility
        tok_config_path = Path(save_dir) / "tokenizer_config.json"
        tok_config = json.loads(tok_config_path.read_text())
        tok_config["tokenizer_class"] = "LlamaTokenizer"
        tok_config_path.write_text(json.dumps(tok_config, indent=2))
