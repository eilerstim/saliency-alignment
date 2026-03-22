from omegaconf import DictConfig
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
)


def build_model(cfg: DictConfig) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Instantiate model and processor."""

    model = AutoModelForImageTextToText.from_pretrained(cfg.name, dtype=cfg.dtype)
    processor = AutoProcessor.from_pretrained(cfg.name)

    model.train()

    if "all" in cfg.freeze:
        model.requires_grad_(False)
    for module in cfg.freeze:
        getattr(model.model, module).requires_grad_(False)
    for module in cfg.unfreeze:
        getattr(model.model, module).requires_grad_(True)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, processor
