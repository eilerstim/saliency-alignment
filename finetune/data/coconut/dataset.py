import json
import logging
import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# COCONut PanCap mixes two annotation styles. We only keep ``<N: text>``
# (colon, 1-indexed) where the bracket ids match mask values directly;
# the ``<N>text>`` style is 0-indexed and uses a non-standard separator,
# colon-style brackets that reference id 0 would target the mask's
# void/background pixels rather than a real segment, and unannotated
# pure-prose captions contribute nothing to the alignment loss / metric.
_NO_COLON_ANNOTATION_RE = re.compile(r"<\s*[\d,\s]+\s*>")
_COLON_ID_LIST_RE = re.compile(r"<\s*([\d,\s]+)\s*:")


def _has_clean_format(caption: str) -> bool:
    if _NO_COLON_ANNOTATION_RE.search(caption):
        return False
    has_annotation = False
    for m in _COLON_ID_LIST_RE.finditer(caption):
        if any(int(d) == 0 for d in re.findall(r"\d+", m.group(1))):
            return False
        has_annotation = True
    return has_annotation


class COCONutPanCapDataset(Dataset):
    """PyTorch Dataset for COCONut panoptic segmentation with captions.

    Loads COCONut images with corresponding panoptic segmentation masks.
    COCONut provides high-quality panoptic annotations from the COCO dataset.

    All captions and segment metadata are cached in memory at init time to
    avoid per-sample filesystem I/O during training. Only images and masks
    are loaded lazily in ``__getitem__``.

    Masks are expected to be pre-decoded ``.npy`` files (int32 panoptic IDs).
    Run ``preprocess_masks_to_npy`` from ``download.py`` to convert the raw
    RGB PNGs once after downloading.

    Attributes:
        data_cfg: Data configuration containing paths and settings.
        split: Dataset split, either "train" or "validation".
        root_dir: Path to the directory containing images.
        mask_dir: Path to the directory containing panoptic masks.
        captions_dir: Path to the directory containing captions.
        images: List of image information dictionaries.
        captions: Mapping from image ID to caption string.
        id_to_segments: Mapping from image ID to list of (segment_id, category_id) tuples.
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        split: Literal["train", "validation"],
    ):
        """Initialize the COCONut panoptic segmentation dataset.

        Args:
            data_cfg: Configuration containing data directory and annotation file paths.
            split: Dataset split to load, either "train" or "validation".
        """
        self.data_cfg = data_cfg
        self.split = split

        # Build paths — COCONut uses train2017 images from COCO
        self.root_dir = Path(data_cfg.train.images_dir)
        self.mask_dir = Path(data_cfg.coconut.masks_dir)
        self.captions_dir = Path(data_cfg.coconut.captions_dir)

        # Load panoptic annotations from JSON file
        ann_file = data_cfg.coconut.ann_file
        with open(ann_file) as f:
            annotations = json.load(f)

        # Single listdir instead of one stat() per image on the network FS
        available_captions = set(os.listdir(self.captions_dir))

        # Load broken image IDs to exclude (optional config entry)
        broken_ids: set[str] = set()
        broken_file = getattr(data_cfg.coconut, "broken_file", None)
        if broken_file is not None and Path(broken_file).exists():
            with open(broken_file) as f:
                broken_ids = {line.strip() for line in f if line.strip()}

        # Filter images, cache captions in one pass
        self.images: list[dict] = []
        self.captions: dict[int, str] = {}
        skipped_no_caption = 0
        skipped_broken = 0
        skipped_bad_format = 0

        for img in annotations["images"]:
            stem = img["file_name"].rsplit(".", 1)[0]

            if stem in broken_ids:
                skipped_broken += 1
                continue

            caption_filename = f"{stem}.txt"
            if caption_filename not in available_captions:
                skipped_no_caption += 1
                continue

            caption_path = self.captions_dir / caption_filename
            with open(caption_path) as f:
                caption = f.read().strip()
            if not _has_clean_format(caption):
                skipped_bad_format += 1
                continue

            self.captions[img["id"]] = caption
            self.images.append(img)

        if skipped_no_caption or skipped_broken or skipped_bad_format:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                logger.info(
                    f"Skipped {skipped_no_caption} images with no caption, "
                    f"{skipped_broken} images due to broken captions, "
                    f"{skipped_bad_format} images due to non-strict annotation format."
                )

        # Pre-extract segments_info keyed by image ID
        self.id_to_segments: dict[int, list[tuple[int, int]]] = {}
        for ann in annotations["annotations"]:
            self.id_to_segments[ann["image_id"]] = [
                (seg["id"], seg["category_id"]) for seg in ann["segments_info"]
            ]

        # Free the full JSON — we've extracted everything we need
        del annotations

        # Deterministic prefix/suffix split: take the last ``n_val`` images
        # for validation, everything before for training. The COCONut JSON
        # is the same on every machine so the iteration order above is
        # already deterministic — no explicit sort needed.
        n_val = int(data_cfg.coconut.n_val)

        if n_val >= len(self.images):
            raise ValueError(
                f"n_val={n_val} leaves no images for training "
                f"(only {len(self.images)} COCONut images available)."
            )

        if split == "train":
            self.images = self.images[:-n_val]
        else:
            self.images = self.images[-n_val:]

        # Trim caches to only images in this split
        active_ids = {img["id"] for img in self.images}
        self.captions = {k: v for k, v in self.captions.items() if k in active_ids}
        self.id_to_segments = {
            k: v for k, v in self.id_to_segments.items() if k in active_ids
        }

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """Load and return a single sample from the dataset.

        Images and masks are loaded lazily; captions and segment metadata
        are served from the in-memory cache.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - image: RGB PIL Image.
                - mask: Long tensor of shape (H, W) with panoptic segment IDs.
                - caption: Caption string for this image.
                - segments_info: List of (segment_id, category_id) tuples.
        """
        img_info = self.images[idx]
        img_id = img_info["id"]
        stem = img_info["file_name"].rsplit(".", 1)[0]

        # Load image
        image = Image.open(self.root_dir / img_info["file_name"]).convert("RGB")

        # Load pre-decoded panoptic mask (.npy, int32 panoptic IDs)
        mask = torch.from_numpy(np.load(self.mask_dir / f"{stem}.npy")).long()

        return {
            "image": image,
            "mask": mask,
            "caption": self.captions[img_id],
            "segments_info": self.id_to_segments.get(img_id, []),
        }
