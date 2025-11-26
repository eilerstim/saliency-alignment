import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset

from finetune.data.grand.download import download_grand

logger = logging.getLogger(__name__)


class GranDDataset(Dataset):
    """PyTorch Dataset for GranD (Grounded and Dense Captions) dataset.

    Loads GranD images with corresponding segmentation masks from dense captions.
    GranD provides detailed scene descriptions with grounded object references.

    Attributes:
        data_cfg: Data configuration containing paths and settings.
        split: Dataset split, either "train" or "validation".
        images_dir: Path to the directory containing images.
        annotations_dir: Path to the directory containing JSON annotation files.
        image_files: List of image file paths.
        annotation_files: List of corresponding annotation file paths.
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        split: Literal["train", "validation"],
    ):
        """Initialize the GranD dataset.

        Args:
            data_cfg: Configuration containing data directory and paths.
            split: Dataset split to load, either "train" or "validation".
        """
        download_grand(data_cfg)

        self.data_cfg = data_cfg
        self.split = split

        # Build paths
        self.images_dir = Path(data_cfg.grand.images_dir)
        self.annotations_dir = Path(data_cfg.grand.repo_dir)

        # Find all annotation files and corresponding images
        # Iterate through JSON files since there are more images than annotations
        annotation_files = sorted(self.annotations_dir.glob("*.json"))

        self.image_files = []
        self.annotation_files = []

        for json_file in annotation_files:
            # Construct expected image file name
            img_file = self.images_dir / f"{json_file.stem}.jpg"
            if img_file.exists():
                self.image_files.append(img_file)
                self.annotation_files.append(json_file)

        logger.info(
            f"Found {len(self.image_files)} image-annotation pairs in GranD dataset"
        )

        # Deterministic 90/10 split via index sampling with a fixed seed
        n_total = len(self.image_files)
        n_train = int(0.9 * n_total)

        # Use torch.Generator for reproducible sampling
        g = torch.Generator()  # Seed already set in main script

        perm = torch.randperm(n_total, generator=g).tolist()
        train_idx = set(perm[:n_train])

        if split == "train":
            self.image_files = [
                self.image_files[i] for i in range(n_total) if i in train_idx
            ]
            self.annotation_files = [
                self.annotation_files[i] for i in range(n_total) if i in train_idx
            ]
        else:  # "validation"
            self.image_files = [
                self.image_files[i] for i in range(n_total) if i not in train_idx
            ]
            self.annotation_files = [
                self.annotation_files[i] for i in range(n_total) if i not in train_idx
            ]

        logger.info(f"GranD {split} split has {len(self.image_files)} images")

    def __len__(self):
        """Return the total number of images in the dataset.

        Returns:
            Number of images with available annotations.
        """
        return len(self.image_files)

    def _decode_rle_mask(self, rle_dict: dict) -> np.ndarray:
        """Decode RLE mask from COCO format to 2D numpy array.

        Args:
            rle_dict: Dictionary with 'size' and 'counts' keys in COCO RLE format.

        Returns:
            Binary mask as 2D numpy array of shape (H, W).
        """
        return maskUtils.decode(rle_dict)

    def _extract_phrase_positions(
        self, dense_caption: dict
    ) -> list[tuple[int, int, list[int]]]:
        """Extract phrase positions and their corresponding object IDs.

        Args:
            dense_caption: Dictionary containing 'caption' and 'details' keys.

        Returns:
            List of tuples (start_pos, end_pos, object_ids) where positions
            are character indices in the caption.
        """
        details = dense_caption.get("details", [])
        phrase_positions = []

        for detail in details:
            start, end = detail["tokens_positive"]
            ids = detail["ids"]
            phrase_positions.append((start, end, ids))

        return phrase_positions

    def __getitem__(self, idx):
        """Load and return a single sample from the dataset.

        Loads an image, its RLE segmentations, caption, and phrase-to-ID mapping.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - image: RGB PIL Image.
                - rle_masks: Dictionary mapping object_id to RLE segmentation dict.
                - caption: Dense caption text (plain string).
                - phrase_positions: List of (start_pos, end_pos, object_ids) tuples.
        """
        image_path = self.image_files[idx]
        annotation_path = self.annotation_files[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotation
        with open(annotation_path) as f:
            annotation_data = json.load(f)

        # Get image key (filename)
        img_key = image_path.name
        img_data = annotation_data.get(img_key, {})

        # Get dense caption
        dense_caption = img_data.get("dense_caption", {})
        caption = dense_caption.get("caption", "")
        phrase_positions = self._extract_phrase_positions(dense_caption)

        # Get objects and store their RLE masks
        objects = img_data.get("objects", [])

        # Create mapping from object_id to RLE segmentation
        rle_masks = {}
        for obj in objects:
            obj_id = obj["id"]
            segmentation = obj.get("segmentation", None)

            if segmentation and "counts" in segmentation and "size" in segmentation:
                rle_masks[obj_id] = segmentation

        return {
            "image": image,
            "rle_masks": rle_masks,
            "caption": caption,
            "phrase_positions": phrase_positions,
        }
