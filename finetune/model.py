from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
)


def build_model(
    cfg: DictConfig,
    lora_cfg: DictConfig | None = None,
) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Instantiate model and processor, optionally wrapped with LoRA adapters."""

    model = AutoModelForImageTextToText.from_pretrained(cfg.name, dtype=cfg.dtype)
    processor = AutoProcessor.from_pretrained(cfg.name)

    model.train()

    if lora_cfg is not None and lora_cfg.get("enabled", False):
        from peft import LoraConfig, get_peft_model

        kwargs = OmegaConf.to_container(lora_cfg, resolve=True)
        assert isinstance(kwargs, dict)
        kwargs.pop("enabled", None)
        model = get_peft_model(model, LoraConfig(**kwargs))
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
