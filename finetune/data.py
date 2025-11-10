from pathlib import Path
from typing import Literal

from datasets import concatenate_datasets, load_from_disk
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import default_data_collator


def collate_fn(batch):
    return default_data_collator(batch)


def load_dataloader(data_cfg: DictConfig, split: Literal["train", "validation"]):
    """
    Load a PyTorch DataLoader for a specific data split.

    Args:
        data_cfg (DictConfig): Configuration dictionary (cfg.data).
        split (Literal["train", "validation"]): The data split to load.

    Returns:
        DataLoader: A PyTorch DataLoader for the specified data split.
    """
    path = data_cfg[split].save_dir
    all_paths = sorted(Path(path).glob("shard_*.arrow"))

    # Load all shards
    datasets = [load_from_disk(str(p)) for p in all_paths]
    ds = concatenate_datasets(datasets)

    # Create dataloader
    dataloader = DataLoader(
        ds.with_format("torch", columns=["input_ids", "labels"]),
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        **data_cfg.dataloader_kwargs,
    )
    return dataloader
