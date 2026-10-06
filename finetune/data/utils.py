"""Shared utilities for data collation."""

import torch


def visible_crop_box(
    image_size: tuple[int, int], image_processor
) -> tuple[int, int, int, int]:
    """Region of the original image that survives the processor's center crop.

    HF's LLaVA-1.5 processor (``CLIPImageProcessor``) resizes the shortest
    edge to ``size["shortest_edge"]`` and then center-crops ``crop_size``, so
    for a non-square image the vision encoder never sees the outer margins
    (for 4:3 COCO images, 12.5% of the width on each side). Annotation masks
    that are compared against the patch grid must be cropped to the same
    region, otherwise the grid is stretched over pixels the model cannot
    attend to. This mirrors ``get_resize_output_image_size`` and
    ``center_crop`` from ``transformers.image_transforms`` and maps the crop
    back to original pixel coordinates.

    Args:
        image_size: ``(width, height)`` of the original image (PIL order).
        image_processor: The processor's image processor.

    Returns:
        ``(left, top, right, bottom)`` in original pixel coordinates. The full
        image when the processor does not center-crop.
    """
    width, height = image_size
    if not getattr(image_processor, "do_center_crop", False):
        return 0, 0, width, height

    size = getattr(image_processor, "size", None) or {}
    if getattr(image_processor, "do_resize", True) and "shortest_edge" in size:
        new_short = int(size["shortest_edge"])
        short, long = (width, height) if width <= height else (height, width)
        new_long = int(new_short * long / short)
        new_w, new_h = (
            (new_short, new_long) if width <= height else (new_long, new_short)
        )
    else:
        new_w, new_h = width, height

    crop = image_processor.crop_size
    crop_h, crop_w = int(crop["height"]), int(crop["width"])
    if crop_h >= new_h and crop_w >= new_w:
        return 0, 0, width, height  # nothing is cropped away (padding case)

    top_r = max((new_h - crop_h) // 2, 0)
    left_r = max((new_w - crop_w) // 2, 0)
    bottom_r = min(top_r + crop_h, new_h)
    right_r = min(left_r + crop_w, new_w)

    sx, sy = width / new_w, height / new_h
    left, right = round(left_r * sx), round(right_r * sx)
    top, bottom = round(top_r * sy), round(bottom_r * sy)
    return left, top, right, bottom


def find_sequence(tensor: torch.Tensor, sequence: list[int]) -> int:
    """Find the starting index of a token sequence in a 1D tensor.

    Args:
        tensor: 1D tensor of token IDs to search in.
        sequence: List of token IDs to search for.

    Returns:
        Starting index of the sequence, or -1 if not found.
    """
    seq_len = len(sequence)
    seq_tensor = torch.tensor(sequence, dtype=tensor.dtype, device=tensor.device)
    for i in range(len(tensor) - seq_len + 1):
        if torch.equal(tensor[i : i + seq_len], seq_tensor):
            return i
    return -1
