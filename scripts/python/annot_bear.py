"""Bear target mask (COCONut segment 5931152) cropped like the saliency maps."""
import numpy as np
from PIL import Image

BOX = (0, 26, 586, 614)  # visible_crop_box of 000000000285.jpg, see crop_bear.py
pan = np.array(Image.open("data/coco/annotations/panoptic_segmentation/val2017/000000000285.png").convert("RGB")).astype(np.int64)
seg = pan[..., 0] + 256 * pan[..., 1] + 256 * 256 * pan[..., 2]
mask = (seg == 5931152)
print("mask pixels", int(mask.sum()), "of", mask.size)
img = Image.open("data/coco/images/val2017/000000000285.jpg").convert("RGB")
m = Image.fromarray((mask * 255).astype(np.uint8)).crop(BOX)
m.save("figs_v2/annot_bear_mask_crop.png")
# teal overlay (DeepTeal 3B82B8 at 55 %) on the cropped photo, outside dimmed
rgb = np.array(img.crop(BOX)).astype(np.float32)
mc = np.array(m).astype(bool)
teal = np.array([0x3B, 0x82, 0xB8], dtype=np.float32)
out = rgb * 0.45
out[mc] = 0.45 * rgb[mc] + 0.55 * teal
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save("figs_v2/annot_bear_overlay_crop.png")
print("saved", m.size)
