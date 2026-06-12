from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Hugging Face model and processor from a Lightning checkpoint "
            "created by finetune.lightning.FineTuner."
        )
    )
    parser.add_argument(
        "checkpoint_or_run",
        type=Path,
        help=(
            "Path to a .ckpt file, an outputs/<run_id> directory, or a directory "
            "containing Lightning checkpoints."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Hydra config to use. Defaults to <run_dir>/.hydra/config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for exported Hugging Face artifacts. Defaults to "
            "models/<run_id>-from-lightning-ckpt. The layout matches "
            "finetune.py: model/adapter and processor files are saved together."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require every checkpoint tensor to match the rebuilt model.",
    )
    return parser.parse_args()


def latest_checkpoint(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got {path}")
        return path

    checkpoints = sorted(
        path.rglob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found below {path}")
    return checkpoints[0]


def infer_run_dir(path: Path) -> Path:
    resolved = path.resolve()
    parts = resolved.parts
    if "outputs" in parts:
        outputs_idx = len(parts) - 1 - parts[::-1].index("outputs")
        if outputs_idx + 1 < len(parts):
            return Path(*parts[: outputs_idx + 2])

    if path.is_dir() and (path / ".hydra" / "config.yaml").exists():
        return path

    raise ValueError(
        "Could not infer the output run directory. Pass --config and --output-dir."
    )


def load_lightning_state(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    logger.info("Loading Lightning checkpoint: %s", checkpoint_path)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state_dict is not a dict: {type(state)}")
    return state


def strip_lightning_prefix(state: dict[str, Any]) -> dict[str, Any]:
    prefixes = (
        "model.",
        "_forward_module.model.",
        "module.model.",
    )
    hf_state: dict[str, Any] = {}

    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                hf_state[key.removeprefix(prefix)] = value
                break

    if not hf_state:
        examples = ", ".join(list(state.keys())[:10])
        raise KeyError(
            "No FineTuner model weights found in checkpoint state_dict. "
            f"First keys: {examples}"
        )

    return hf_state


def save_tokenizer_compat_config(processor_dir: Path) -> None:
    tok_config_path = processor_dir / "tokenizer_config.json"
    if not tok_config_path.exists():
        return

    tok_config = json.loads(tok_config_path.read_text())
    tok_config["tokenizer_class"] = "LlamaTokenizer"
    tok_config_path.write_text(json.dumps(tok_config, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    checkpoint_path = latest_checkpoint(args.checkpoint_or_run)
    if args.config is None:
        run_dir = infer_run_dir(
            checkpoint_path if checkpoint_path.is_file() else args.checkpoint_or_run
        )
        config_path = run_dir / ".hydra" / "config.yaml"
    else:
        config_path = args.config

    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    output_dir = (
        args.output_dir or PROJECT_ROOT / "models" / f"{cfg.run_id}-from-lightning-ckpt"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Hydra config: %s", config_path)
    from finetune.model import build_model

    model, processor = build_model(cfg.model, cfg.lora)

    lt_state = load_lightning_state(checkpoint_path)
    hf_state = strip_lightning_prefix(lt_state)
    incompatible = model.load_state_dict(hf_state, strict=args.strict)

    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing:
        logger.warning("Missing keys while loading model: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys while loading model: %d", len(unexpected))

    logger.info("Saving model to %s", output_dir)
    model.save_pretrained(output_dir, state_dict=hf_state)

    logger.info("Saving processor to %s", output_dir)
    processor.save_pretrained(output_dir)
    save_tokenizer_compat_config(output_dir)

    logger.info("Export complete")


if __name__ == "__main__":
    main()
