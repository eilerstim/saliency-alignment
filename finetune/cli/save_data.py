import itertools
import logging
from pathlib import Path
from typing import Literal

import datasets
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def save_data(
    data_cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
    split: Literal["train", "validation"],
):
    """
    Load, process, and save dataset split for fine-tuning.

    Args:
        data_cfg (DictConfig): Data configuration.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        split (Literal["train", "validation"]): Dataset split to process.
    """
    # Load dataset
    ds = datasets.load_dataset(data_cfg.name, split=data_cfg[split].split)
    logger.info(f"Loaded {split} split with {len(ds)} samples.")

    def tokenize(example):
        input_ids = tokenizer(
            example["text"], truncation=False, return_attention_mask=False
        )["input_ids"]
        return {"input_ids": input_ids + [tokenizer.eos_token_id]}

    # Apply tokenization
    ds = ds.map(tokenize, remove_columns=ds.column_names, batched=True)

    block_size = data_cfg.block_size

    def pack_samples(samples):
        # concatenate then chunk
        ids = list(itertools.chain.from_iterable(samples["input_ids"]))

        # Trim to a multiple of block_size
        total_len = (len(ids) // block_size) * block_size
        ids = ids[:total_len]

        # Create chunks
        chunks = [ids[i : i + block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": chunks, "labels": chunks}

    # Pack samples into fixed-size blocks
    ds = ds.map(pack_samples, batched=True, batch_size=1000)

    # Save processed dataset
    output_dir = Path(data_cfg[split].output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each shard as a separate file
    shards = data_cfg[split].num_shards
    for shard_id in range(shards):
        shard = ds.shard(num_shards=shards, index=shard_id, contiguous=True)
        shard.save_to_disk(output_dir / f"shard_{shard_id}")

    logger.info(f"Saved {shards} shards for {split} to {output_dir}")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        logger.warning("Added missing EOS token to tokenizer.")

    # Save processed data for train and validation splits
    save_data(cfg.data, tokenizer, split="train")
    save_data(cfg.data, tokenizer, split="validation")


if __name__ == "__main__":
    main()  # type: ignore
