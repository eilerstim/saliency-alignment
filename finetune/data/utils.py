"""Shared utilities for data collation."""

import torch


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


def processor_view_box(
    height: int, width: int, image_processor
) -> tuple[int, int, int, int]:
    """Region of an ``height x width`` image that the model's vision grid covers.

    CLIP-style processors (e.g. LLaVA-1.5) resize the shortest edge and then
    center-crop, so the patch grid only covers the central square of a
    non-square image. Returns ``(top, bottom, left, right)`` in original pixel
    coordinates; the whole image for processors that do not center-crop.
    """
    full = (0, height, 0, width)
    if not getattr(image_processor, "do_center_crop", False):
        return full
    crop = getattr(image_processor, "crop_size", None) or {}
    crop_h, crop_w = crop.get("height"), crop.get("width")
    if crop_h is None or crop_w is None:
        return full

    # Size after resizing, as transformers' get_resize_output_image_size
    # computes it for {"shortest_edge": s} with default_to_square=False
    size = getattr(image_processor, "size", None) or {}
    if getattr(image_processor, "do_resize", False) and "shortest_edge" in size:
        s = size["shortest_edge"]
        short, long = sorted((height, width))
        new_short, new_long = s, int(s * long / short)
        res_h, res_w = (new_short, new_long) if height <= width else (new_long, new_short)
    else:
        res_h, res_w = height, width
    if crop_h > res_h or crop_w > res_w:
        return full  # the processor pads instead of cropping

    # Crop offsets as in transformers' center_crop, mapped back to the original
    top, left = (res_h - crop_h) // 2, (res_w - crop_w) // 2
    sy, sx = height / res_h, width / res_w
    return (
        round(top * sy),
        round((top + crop_h) * sy),
        round(left * sx),
        round((left + crop_w) * sx),
    )


def crop_to_processor_view(mask: torch.Tensor, image_processor) -> torch.Tensor:
    """Crop an ``(H, W)`` annotation mask to the region the vision grid covers."""
    top, bottom, left, right = processor_view_box(*mask.shape[-2:], image_processor)
    return mask[..., top:bottom, left:right]
