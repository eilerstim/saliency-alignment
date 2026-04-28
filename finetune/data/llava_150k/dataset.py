from typing import Literal

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Subset

from finetune.data.coconut import COCONutPanCapDataset


def llava_150k_instruct_dataset(
    data_cfg: DictConfig, split: Literal["train", "validation"]
):
    """Creates a combined dataset for the LLaVA 150k instruction tuning, consisting of:
    1. The original LLaVA 150k dataset (complex reasoning and conversation subsets).
    2. The COCONut panoptic segmentation dataset with captions.

    Args:
        data_cfg: Configuration containing paths for the COCONut dataset.
        split: Dataset split to load, either "train" or "validation".
    Returns:
        A concatenated dataset combining LLaVA 150k and COCONut.
    """

    ds1 = load_dataset(
        "liuhaotian/LLaVA-Instruct-150K",
        split="train",  # Only has a train split
        data_files=["complex_reasoning_77k.json", "conversation_58k.json"],
    )
    ds2 = COCONutPanCapDataset(data_cfg=data_cfg, split=split)

    if hasattr(data_cfg, "coconut_samples"):
        indices = torch.randperm(len(ds2))[: data_cfg.coconut_samples]
        ds2 = Subset(ds2, indices)

    return ConcatDataset([ds1, ds2])
