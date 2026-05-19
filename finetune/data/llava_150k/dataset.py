from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, ProcessorMixin

from finetune.data.coconut.collator import _compute_suffix_tokens
from finetune.data.coconut.tokenization import (
    parse_annotated_caption,
    tokenize_from_parsed,
)
from finetune.data.utils import find_sequence


def make_collate_fn(
    processor: ProcessorMixin,
) -> Callable[[list[dict]], tuple[dict, list[torch.Tensor], list[torch.Tensor]]]:
    tokenizer = processor.tokenizer

    def collate_fn(batch):
        masks = [x.pop("mask") for x in batch]
        segment_ids = [x.pop("segment_ids") for x in batch]
        text_batch = [
            {
                "input_ids": torch.tensor(x["input_ids"]),
                "attention_mask": torch.tensor(x["attention_mask"]),
            }
            for x in batch
        ]

        padded = tokenizer.pad(
            text_batch,
            padding=True,
            return_tensors="pt",
        )

        padded["labels"] = pad_sequence(
            [torch.tensor(x["labels"]) for x in batch],
            batch_first=True,
            padding_value=-100,
        )

        padded["pixel_values"] = torch.cat([x["pixel_values"] for x in batch])
        return padded, masks, segment_ids

    return collate_fn


def build_transform(
    processor: ProcessorMixin,
    mask_dir: Path,
    image_dir: Path,
    suffix_tokens: list[int],
):
    tokenize_fn = build_tokenize_fn(processor, mask_dir, suffix_tokens)

    def transform(batch):
        examples = [
            {key: batch[key][i] for key in batch}
            for i in range(len(batch["id"]))
        ]
        tokenized = [tokenize_fn(ex) for ex in examples]

        images = [
            Image.open(image_dir / f"{id_}.jpg").convert("RGB") for id_ in batch["id"]
        ]
        image_inputs = processor.image_processor(
            images=images,
            return_tensors="pt",
        )

        out = {key: [t[key] for t in tokenized] for key in tokenized[0]}
        out["pixel_values"] = image_inputs["pixel_values"]
        return out

    return transform


def build_tokenize_fn(
    processor: ProcessorMixin,
    mask_dir: Path,
    suffix_tokens: list[int],
):
    def tokenize_fn(example):
        text = example["conversations"]

        if example["segments"]:
            mask = torch.from_numpy(np.load(mask_dir / f"{example['id']}.npy")).long()
            assistant_text = text[-1]["content"][-1]["text"]
            parsed_segments = parse_annotated_caption(assistant_text)
            cleaned = "".join(text for _, text in parsed_segments)
            text[-1]["content"][-1]["text"] = cleaned
        else:
            mask = None

        prompt = processor.apply_chat_template(
            text, tokenize=False, add_generation_prompt=False
        )

        processed = processor.tokenizer(
            prompt,
            padding=False,
            return_tensors="pt",
        )
        processed = {k: v.squeeze(0) for k, v in processed.items()}

        input_ids = processed["input_ids"]

        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        caption_start = find_sequence(input_ids, suffix_tokens) + len(suffix_tokens)
        labels[:caption_start] = -100

        if example["segments"]:
            _, cap_ann_ids = tokenize_from_parsed(
                parsed_segments, processor.tokenizer, add_special_tokens=False
            )
            seq_len = processed["input_ids"].shape[0]

            caption_len = min(len(cap_ann_ids), seq_len - caption_start)
            cap_ann_ids = cap_ann_ids[:caption_len]

            max_segments = max((len(ids) for ids in cap_ann_ids), default=1)
            segment_ids = torch.full((seq_len, max_segments), -1, dtype=torch.long)

            for j, ann_ids in enumerate(cap_ann_ids, start=caption_start):
                for k, ann_id in enumerate(ann_ids):
                    segment_ids[j, k] = ann_id
        else:
            segment_ids = None

        return {
            **processed,
            "labels": labels,
            "mask": mask,
            "segment_ids": segment_ids,
        }

    return tokenize_fn


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
    processor = AutoProcessor.from_pretrained(data_cfg.processor)
    mask_dir = Path(data_cfg.coconut.masks_dir)
    image_dir = Path(data_cfg.images_dir)
    suffix_tokens = _compute_suffix_tokens(processor)

    ds.set_transform(build_transform(processor, mask_dir, image_dir, suffix_tokens))
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
