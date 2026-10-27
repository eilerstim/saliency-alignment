import sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.data.utils import visible_crop_box
img = Image.open("figs_v2/inputs/6f506bf9db.jpg").convert("RGB")
proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
box = visible_crop_box(img.size, proc.image_processor)
crop = img.crop(box)
crop.save("figs_v2/input_bear_crop.png")
print("image", img.size, "crop box", box, "crop size", crop.size)
