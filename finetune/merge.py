"""Merge a LoRA adapter into its base model and save as a full HF checkpoint.

Useful for downstream tools that don't speak PEFT (e.g. vLLM-backed
benchmarks), which need a regular HF model directory.

Usage:
    python -m finetune.merge <adapter_dir> [--output <dir>] [--dtype bfloat16]
"""

import argparse
import json
from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


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
    out_dir = Path(args.output) if args.output else adapter_dir.with_name(
        adapter_dir.name + "-merged"
    )

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    print(f"Loading base model: {peft_config.base_model_name_or_path}")
    base = AutoModelForImageTextToText.from_pretrained(
        peft_config.base_model_name_or_path, dtype=args.dtype
    )

    print(f"Attaching adapter from {adapter_dir} and merging...")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

    print(f"Saving merged model to {out_dir}")
    merged.save_pretrained(str(out_dir))
    AutoProcessor.from_pretrained(str(adapter_dir)).save_pretrained(str(out_dir))

    # Mirror finetune.py's vLLM-compatibility fix on the merged dir's
    # tokenizer config. Idempotent in modern transformers but guards
    # against any auto-class or version drift that would leave a value
    # vLLM doesn't recognise.
    tok_config_path = out_dir / "tokenizer_config.json"
    tok_config = json.loads(tok_config_path.read_text())
    tok_config["tokenizer_class"] = "LlamaTokenizer"
    tok_config_path.write_text(json.dumps(tok_config, indent=2))

    print("Done.")


if __name__ == "__main__":
    main()
