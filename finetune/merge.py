"""Merge a LoRA adapter into its base model and save as a full HF checkpoint.

Useful for downstream tools that don't speak PEFT (e.g. vLLM-backed
benchmarks), which need a regular HF model directory.

Usage:
    python -m finetune.merge <adapter_dir> [--output <dir>] [--dtype bfloat16]
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    CLIPVisionModel,
    LlavaConfig,
    LlavaForConditionalGeneration,
)


def create_instruction_tuned_model():
    config = LlavaConfig.from_pretrained(
        "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"
    )
    model = LlavaForConditionalGeneration(config)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    llm = AutoModelForCausalLM.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        dtype=torch.bfloat16,
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
        "openai/clip-vit-large-patch14-336",
        dtype=torch.bfloat16,
    )
    projector_path = hf_hub_download(
        repo_id="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
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

    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("adapter_dir", help="Path to LoRA adapter checkpoint.")
    ap.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <adapter_dir>-merged).",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        help="dtype to load the base model with (default: bfloat16).",
    )
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    out_dir = (
        Path(args.output)
        if args.output
        else adapter_dir.with_name(adapter_dir.name + "-merged")
    )

    # peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    # print(f"Loading base model: {peft_config.base_model_name_or_path}")
    # base = AutoModelForImageTextToText.from_pretrained(
    #     peft_config.base_model_name_or_path, dtype=args.dtype
    # )

    print("Loading base model for instruction tuning...")
    base = create_instruction_tuned_model()

    print(f"Attaching adapter from {adapter_dir} and merging...")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

    print(f"Saving merged model to {out_dir}")
    merged.save_pretrained(str(out_dir))
    AutoProcessor.from_pretrained(str(adapter_dir)).save_pretrained(str(out_dir))

    # Mirror finetune.py's vLLM-compatibility tokenizer_class fix.
    tok_config_path = out_dir / "tokenizer_config.json"
    tok_config = json.loads(tok_config_path.read_text())
    tok_config["tokenizer_class"] = "LlamaTokenizer"
    tok_config_path.write_text(json.dumps(tok_config, indent=2))

    print("Done.")


if __name__ == "__main__":
    main()
