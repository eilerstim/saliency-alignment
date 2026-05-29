import copy
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin

from finetune.data.coconut.collator import _compute_suffix_tokens
from finetune.data.coconut.tokenization import (
    parse_annotated_caption,
    tokenize_from_parsed,
)
from finetune.data.utils import find_sequence


def make_collate_fn(
    processor: ProcessorMixin,
    masks_dir: str,
    images_dir: str,
) -> Callable[
    [list[dict]], tuple[dict, list[torch.Tensor | None], list[torch.Tensor | None]]
]:
    tokenizer = processor.tokenizer
    suffix_tokens = _compute_suffix_tokens(processor)

    def collate_fn(batch: list[dict]):
        texts = []
        images = []
        masks = []
        parsed_segments_batch = []

        for ex in batch:
            text = ex["conversations"]
            image = Image.open(f"{images_dir}/{ex['id']}.jpg").convert("RGB")

            if ex["segments"]:
                mask = torch.from_numpy(np.load(f"{masks_dir}/{ex['id']}.npy")).long()

                assistant_text = text[-1]["content"][-1]["text"]
                parsed_segments = parse_annotated_caption(assistant_text)
                cleaned = "".join(t for _, t in parsed_segments)

                # avoid mutating HF dataset-owned nested object
                text = copy.deepcopy(text)
                text[-1]["content"][-1]["text"] = cleaned
            else:
                mask = None
                parsed_segments = None

            prompt = processor.apply_chat_template(
                text,
                tokenize=False,
                add_generation_prompt=False,
            )

            texts.append(prompt)
            images.append(image)
            masks.append(mask)
            parsed_segments_batch.append(parsed_segments)

        processed = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = processed["input_ids"]
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        segment_ids = []

        for i, parsed_segments in enumerate(parsed_segments_batch):
            caption_start = find_sequence(input_ids[i], suffix_tokens)

            if caption_start == -1:
                labels[i, :] = -100
                segment_ids.append(None)
                continue

            caption_start += len(suffix_tokens)
            labels[i, :caption_start] = -100

            if parsed_segments is None:
                segment_ids.append(None)
                continue

            _, cap_ann_ids = tokenize_from_parsed(
                parsed_segments,
                tokenizer,
                add_special_tokens=False,
            )

            seq_len = input_ids.shape[1]
            caption_len = min(len(cap_ann_ids), seq_len - caption_start)
            cap_ann_ids = cap_ann_ids[:caption_len]

            max_segments = max((len(ids) for ids in cap_ann_ids), default=1)
            ids_tensor = torch.full(
                (seq_len, max_segments),
                -1,
                dtype=torch.long,
            )

            for j, ann_ids in enumerate(cap_ann_ids, start=caption_start):
                for k, ann_id in enumerate(ann_ids):
                    ids_tensor[j, k] = ann_id

            segment_ids.append(ids_tensor)

        processed["labels"] = labels
        return processed, masks, segment_ids

    return collate_fn


def llava_150k_instruct_dataset(
    data_cfg: DictConfig, split: Literal["train", "validation"]
) -> torch.utils.data.Dataset:
    """Creates a combined dataset for the LLaVA 150k instruction tuning, consisting of:
    1. The original LLaVA 150k dataset (complex reasoning and conversation subsets).
    2. The COCONut panoptic segmentation dataset with captions.

    Args:
        data_cfg: Configuration containing paths for the COCONut dataset.
    Returns:
        A concatenated dataset combining LLaVA 150k and COCONut.
    """
    data_files = {split: f"{data_cfg.llava_150k_dir}/{split}/data-00000-of-00001.arrow"}
    ds = load_dataset("arrow", data_files=data_files, split=split)
    return ds


if __name__ == "__main__":
    from omegaconf import DictConfig

    dict_cfg = {
        "split": "train",
        "data_dir": "data/llava_coconut_pancap_150k",
        "processor": "llava-hf/llava-1.5-7b-hf",
        "coconut": {
            "images_dir": "data/coco/images/train2017",
            "masks_dir": "data/coco/panoptic_train2017_masks",
            "ann_file": "data/coco/annotations/panoptic_train2017.json",
        },
    }
    cfg = DictConfig(dict_cfg)
    dataset = llava_150k_instruct_dataset(cfg)

    # now, get a batch from a dataloade
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=make_collate_fn(AutoProcessor.from_pretrained(cfg.processor)),
    )
    batch, maps, segment_ids = next(iter(dataloader))
    print(batch.keys())
    for k, v in batch.items():
        print(f"{k}: {v.shape}")
