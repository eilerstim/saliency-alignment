import itertools
import json
import logging
from pathlib import Path
from typing import Literal

import datasets
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from finetune.cli.download import download_coco, download_coconut

logger = logging.getLogger(__name__)


class COCODataset(Dataset):
    """PyTorch Dataset for COCO segmentation with captions.

    Loads COCO images with corresponding segmentation masks and captions. Masks are
    created by merging all object instances, with each pixel labeled by its category ID.

    Attributes:
        data_cfg: Data configuration containing paths and settings.
        split: Dataset split, either "train" or "validation".
        root_dir: Path to the directory containing images.
        coco: COCO API instance for segmentation annotations.
        captions: COCO API instance for caption annotations.
        ids: List of image IDs in the dataset.
        transform: Optional transform to apply to images.
        target_size: Target size (height, width) for resizing images and masks.
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        split: Literal["train", "validation"],
        transform=None,
        target_size=(1024, 1024),
    ):
        """Initialize the COCO segmentation dataset.

        Args:
            data_cfg: Configuration containing data directory and annotation file paths.
            split: Dataset split to load, either "train" or "validation".
            transform: Optional callable transform to apply to images. Defaults to None.
            target_size: Tuple of (height, width) to resize images and masks. Defaults to (1024, 1024).
        """
        download_coco(data_cfg)

        self.data_cfg = data_cfg
        self.split = split

        # Build paths based on split
        ann_file = data_cfg.annotations.ann_file.format(data_cfg[split].name)
        captions_file = data_cfg.annotations.captions_file.format(data_cfg[split].name)

        self.root_dir = Path(data_cfg.data_dir) / data_cfg[split].name
        self.coco = COCO(ann_file)
        self.captions = COCO(captions_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        """Return the total number of images in the dataset.

        Returns:
            Number of images in the dataset.
        """
        return len(self.ids)

    def __getitem__(self, idx):
        """Load and return a single sample from the dataset.

        Loads an image, creates a merged segmentation mask with category IDs, and retrieves
        all associated captions. Images and masks are resized to the target size.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing:
                - image: RGB image tensor (or PIL Image if no transform).
                - mask: Long tensor of shape (H, W) with category IDs for each pixel.
                - caption_texts: List of caption strings for this image.
        """
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = self.root_dir / img_info["file_name"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get segmentation mask for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info["height"], img_info["width"]))

        # Merge all object masks
        # Each object gets its own category id in the mask
        for ann in anns:
            if "segmentation" in ann:
                current_mask = self.coco.annToMask(ann)
                mask = np.maximum(mask, current_mask * ann["category_id"])

        # Resize image and mask (to the same size)
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = Image.fromarray(mask.astype(np.uint8)).resize(
            self.target_size, Image.NEAREST
        )
        mask = np.array(mask)

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()

        # Get all captions
        captions_ids = self.captions.getAnnIds(imgIds=img_id)
        captions_anns = self.captions.loadAnns(captions_ids)
        caption_texts = [ann["caption"] for ann in captions_anns]

        return {
            "image": image,
            "mask": mask,
            "captions": caption_texts,
        }


class COCONutPanCapDataset(Dataset):
    """PyTorch Dataset for COCONut panoptic segmentation with captions.

    Loads COCONut images with corresponding panoptic segmentation masks.
    COCONut provides high-quality panoptic annotations from the COCO dataset.

    Attributes:
        data_cfg: Data configuration containing paths and settings.
        split: Dataset split, either "train" or "validation".
        root_dir: Path to the directory containing images.
        mask_dir: Path to the directory containing panoptic masks.
        captions_dir: Path to the directory containing captions.
        annotations: Dictionary containing parsed annotation data.
        images: List of image information dictionaries.
        img_id_to_info: Mapping from image ID to image information.
        annotations_list: List of annotation dictionaries with segments_info.
        file_name_to_annotation: Mapping from mask file name to segments_info.
    """

    def __init__(self, data_cfg: DictConfig, split: Literal["train", "validation"]):
        """Initialize the COCONut panoptic segmentation dataset.

        Args:
            data_cfg: Configuration containing data directory and annotation file paths.
            split: Dataset split to load, either "train" or "validation".
        """
        download_coco(data_cfg)
        download_coconut(data_cfg)

        self.data_cfg = data_cfg
        self.split = split

        # Build paths - COCONut uses train2017 images from COCO
        self.root_dir = Path(data_cfg.data_dir) / "train2017"
        self.mask_dir = Path(data_cfg.coconut.masks_dir)
        self.captions_dir = Path(data_cfg.coconut.captions_dir)

        # Load panoptic annotations from JSON file
        ann_file = data_cfg.coconut.ann_file
        with open(ann_file) as f:
            self.annotations = json.load(f)

        # Extract image information
        self.images = self.annotations["images"]
        self.img_id_to_info = {img["id"]: img for img in self.images}

        # Create mapping from file_name to annotation (segments_info)
        self.annotations_list = self.annotations["annotations"]
        self.file_name_to_annotation = {
            ann["file_name"]: ann for ann in self.annotations_list
        }

        # Filter images to only those with captions available
        # The length is determined by available caption files
        caption_files = list(self.captions_dir.glob("*.txt"))
        caption_basenames = {f.stem for f in caption_files}

        # Filter images to only include those with captions
        self.images = [
            img
            for img in self.images
            if img["file_name"].split(".")[0] in caption_basenames
        ]

    def __len__(self):
        """Return the total number of images in the dataset.

        Returns:
            Number of images with available captions.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """Load and return a single sample from the dataset.

        Loads an image, its panoptic segmentation mask, associated captions, and segment info.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing:
                - image: RGB PIL Image.
                - mask: Long tensor of shape (H, W) with panoptic segmentation IDs.
                - caption_text: Caption string for this image.
                - segments_info: List of tuples (id, category_id) for each segment in the mask.
        """
        img_info = self.images[idx]
        image_path = self.root_dir / img_info["file_name"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load panoptic mask
        # COCONut stores masks as PNG files with the same image ID
        mask_filename = f"{img_info['file_name'].split('.')[0]}.png"
        mask_path = self.mask_dir / mask_filename
        mask = Image.open(mask_path)
        mask = torch.from_numpy(np.array(mask)).long()

        # Load captions from TXT file
        caption_filename = f"{img_info['file_name'].split('.')[0]}.txt"
        caption_path = self.captions_dir / caption_filename

        with open(caption_path) as f:
            caption_text = f.read().strip()

        # Get segments_info for this image
        # The annotation file_name includes .png extension for panoptic masks
        annotation_key = mask_filename
        annotation = self.file_name_to_annotation.get(annotation_key, {})

        # Extract (id, category_id) tuples from segments_info
        segments_info = []
        if "segments_info" in annotation:
            segments_info = [
                (seg["id"], seg["category_id"]) for seg in annotation["segments_info"]
            ]

        # Return a dict for clarity
        return {
            "image": image,
            "mask": mask,
            "caption": caption_text,
            "segments_info": segments_info,
        }


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
