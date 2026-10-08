"""Write the gathered (FSDP full) state dict as a checkpoint.

Full fine-tunes go through ``PreTrainedModel.save_pretrained`` with the
gathered state dict. LoRA runs cannot: PEFT's ``save_pretrained`` selects the
adapter tensors by matching the given state dict against the model's module
tree, and FSDP rewrites that tree (every wrapped module gains an
``_fsdp_wrapped_module`` level), so after training PEFT finds no adapter
tensors and silently writes an empty ``adapter_model.safetensors``. The keys
of the gathered state dict are clean, so the adapter tensors are selected by
name here and written in PEFT's file layout (``adapter_config.json`` plus
``adapter_model.safetensors`` with the adapter name stripped from the keys),
which ``PeftModel.from_pretrained`` and ``finetune.merge`` load as usual.
"""

import logging
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def peft_adapter_state(
    hf_state: dict[str, torch.Tensor], adapter_name: str = "default"
) -> dict[str, torch.Tensor]:
    """LoRA tensors of ``adapter_name`` from a full state dict, in PEFT's
    on-disk naming (``...q_proj.lora_A.default.weight`` ->
    ``...q_proj.lora_A.weight``)."""
    tag = f".{adapter_name}"
    return {
        k.replace(tag, ""): v
        for k, v in hf_state.items()
        if "lora_" in k and tag in k
    }


def save_model(model, hf_state: dict[str, torch.Tensor], save_dir: str) -> None:
    """Save ``hf_state`` as a HF checkpoint, or as a PEFT adapter for LoRA runs."""
    if not isinstance(model, PeftModel):
        model.save_pretrained(save_dir, state_dict=hf_state)
        return

    adapter_name = model.active_adapter
    adapter = peft_adapter_state(hf_state, adapter_name)
    if not adapter:
        raise RuntimeError(
            "No LoRA tensors in the gathered state dict; refusing to write an "
            "empty adapter."
        )
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = model.peft_config[adapter_name]
    inference_mode = config.inference_mode
    config.inference_mode = True  # as PEFT's save_pretrained does
    try:
        config.save_pretrained(str(out))
    finally:
        config.inference_mode = inference_mode

    save_file(
        {k: v.detach().cpu().contiguous() for k, v in adapter.items()},
        str(out / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    logger.info(f"Saved {len(adapter)} LoRA tensors to {out}")
