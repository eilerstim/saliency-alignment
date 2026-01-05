"""Dataset for PNG-COCO panoptic segmentation with grounded captions."""

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PNGCOCODataset(Dataset):
    """PyTorch Dataset for PNG-COCO panoptic segmentation with grounded captions.

    Loads COCO images with corresponding panoptic segmentation masks and captions
    from the PNG-COCO dataset, which provides phrase-level grounding annotations
    mapping caption phrases to panoptic segment IDs.

    The PNG-COCO annotation format:
    - image_id: COCO image ID (as string)
    - annotator_id: ID of the annotator
    - caption: Full caption text
    - segments: List of phrase segments, each with:
        - utterance: The phrase text
        - segment_ids: List of panoptic segment IDs this phrase refers to
        - plural: Whether the noun is plural
        - noun: Whether this is a noun phrase

    Attributes:
        data_cfg: Data configuration containing paths and settings.
        split: Dataset split, either "train" or "validation".
        root_dir: Path to the directory containing images.
        panoptic_dir: Path to the directory containing panoptic segmentation masks.
        annotations: List of PNG-COCO annotations.
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        split: Literal["train", "validation"],
    ):
        """Initialize the PNG-COCO dataset.

        Args:
            data_cfg: Configuration containing data directory and annotation file paths.
                Expected structure (from configs/data/png.yaml inheriting coco.yaml):
                - images_dir: Base images directory
                - panoptic.segmentation_dir: Panoptic segmentation masks directory
                - png.ann_file_train: PNG-COCO train annotations JSON
                - png.ann_file_val: PNG-COCO val annotations JSON
            split: Dataset split to load, either "train" or "validation".
        """
        self.data_cfg = data_cfg
        self.split = split

        # Map split name to COCO naming convention
        coco_split = "train" if split == "train" else "val"

        # Build paths using config structure from coconut.yaml
        self.root_dir = Path(data_cfg.images_dir) / f"{coco_split}2017"
        self.panoptic_dir = (
            Path(data_cfg.panoptic.segmentation_dir) / f"{coco_split}2017"
        )

        # Load PNG-COCO annotations from config paths
        png_ann_file = (
            data_cfg.png.ann_file_train if split == "train" else data_cfg.png.ann_file_val
        )
        with open(png_ann_file) as f:
            self.annotations = json.load(f)

        logger.info(
            f"Loaded {len(self.annotations)} annotations for split '{split}'"
        )

    def __len__(self):
        """Return the total number of annotations in the dataset.

        Returns:
            Number of caption annotations.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """Load and return a single sample from the dataset.

        Loads an image, its panoptic segmentation mask, the grounded caption
        with segment IDs per phrase.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - image: RGB PIL Image.
                - panoptic_mask: Long tensor of shape (H, W) with panoptic segment IDs.
                - caption: Full caption string.
                - segments: List of segment dicts with utterance and segment_ids.
        """
        ann = self.annotations[idx]
        image_id = int(ann["image_id"])

        # Construct filenames from image_id (COCO uses zero-padded 12-digit IDs)
        image_filename = f"{image_id:012d}.jpg"
        mask_filename = f"{image_id:012d}.png"

        image_path = self.root_dir / image_filename
        mask_path = self.panoptic_dir / mask_filename

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load panoptic mask
        mask = np.array(Image.open(mask_path))

        # COCO panoptic masks are stored as RGB images where
        # id = R + G * 256 + B * 256^2
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask = (
                mask[:, :, 0].astype(np.int32)
                + mask[:, :, 1].astype(np.int32) * 256
                + mask[:, :, 2].astype(np.int32) * 256 * 256
            )

        panoptic_mask = torch.from_numpy(mask).long()

        # Extract caption and segments from PNG-COCO annotation
        caption = ann["caption"]
        segments = ann["segments"]

        return {
            "image": image,
            "panoptic_mask": panoptic_mask,
            "caption": caption,
            "segments": segments,
        }
