"""The annotation mask must cover the same region as the image the vision tower
sees. LLaVA-1.5's CLIP processor resizes the shortest edge to 336 and
center-crops 336x336, so for non-square images the patch grid only covers the
central square; an uncropped mask would be stretched over the full image.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPImageProcessor

from finetune.data.utils import crop_to_processor_view, processor_view_box

LLAVA_15 = dict(
    size={"shortest_edge": 336},
    crop_size={"height": 336, "width": 336},
    do_center_crop=True,
    do_resize=True,
    resample=3,
    do_normalize=False,
    do_rescale=False,
)


def bright_columns(image_processor, height, width, x0, x1):
    """Columns of the processed image lit by a white stripe at [x0, x1)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, x0:x1] = 255
    out = image_processor(Image.fromarray(img), return_tensors="np")["pixel_values"]
    profile = out[0, 0].mean(axis=0)  # (W_out,)
    return np.flatnonzero(profile > 127)


@pytest.mark.parametrize("height,width", [(480, 640), (427, 640), (640, 480), (500, 333)])
def test_view_box_matches_processor_crop(height, width):
    """A stripe's position in the processed image matches the box mapping."""
    proc = CLIPImageProcessor(**LLAVA_15)
    top, bottom, left, right = processor_view_box(height, width, proc)
    assert bottom - top > 0 and right - left > 0
    if height < width:
        assert (top, bottom) == (0, height)
    else:
        assert (left, right) == (0, width)

    # Stripe inside the crop: lands where the box mapping predicts
    span = right - left
    x0 = left + span // 2 - 20
    cols = bright_columns(proc, height, width, x0, x0 + 40)
    expected = (np.array([x0, x0 + 40]) - left) * 336 / span
    assert abs(cols.min() - expected[0]) <= 2 and abs(cols.max() + 1 - expected[1]) <= 2

    # Stripe just outside the crop (landscape only): not visible at all
    if left >= 20:
        assert bright_columns(proc, height, width, 0, left - 5).size == 0


def test_crop_mask_landscape():
    proc = CLIPImageProcessor(**LLAVA_15)
    mask = torch.arange(640).repeat(480, 1)  # each pixel = its column index
    cropped = crop_to_processor_view(mask, proc)
    assert cropped.shape == (480, 480)
    assert cropped[0, 0] == 80 and cropped[0, -1] == 559


def test_no_crop_processor_keeps_mask():
    proc = CLIPImageProcessor(**{**LLAVA_15, "do_center_crop": False})
    mask = torch.zeros(480, 640)
    assert crop_to_processor_view(mask, proc).shape == (480, 640)
    assert crop_to_processor_view(mask, None).shape == (480, 640)
