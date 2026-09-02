import argparse
import csv
import hashlib
import os
import re
from urllib.parse import urlparse

import requests
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from vl_saliency import Saliency
from vl_saliency.select import regex

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", help="CSV file with columns: word, prompt, response, image_url")
parser.add_argument("--output_dir", default="figs", help="Directory to save figures")
parser.add_argument(
    "--models",
    nargs="+",
    default=["base=llava-hf/llava-1.5-7b-hf"],
    metavar="NAME=PATH",
    help="Models to visualize as NAME=PATH (repeatable). Default: base only.",
)
parser.add_argument("--dpi", type=int, default=200, help="Resolution of the saved maps")
args = parser.parse_args()

models_to_run = [tuple(spec.split("=", 1)) for spec in args.models]
os.makedirs(args.output_dir, exist_ok=True)

transformers.utils.logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"


def url_slug(url: str) -> str:
    """Stable, collision-free file stem for an image URL.

    Uses the URL's basename (e.g. the COCO image id) when it is informative and
    appends a short hash so two URLs with the same basename never collide.
    """
    stem = os.path.splitext(os.path.basename(urlparse(url).path))[0]
    stem = re.sub(r"[^a-zA-Z0-9]", "_", stem)[:24] or "img"
    return f"{stem}_{hashlib.md5(url.encode()).hexdigest()[:6]}"


def save_clean_map(fig, path: str, dpi: int) -> None:
    """Save only the image axes of a saliency figure (no title, colorbar, margins).

    The grid figure in the paper needs the bare overlay; cropping the rendered
    PNG by hand is fragile because the axes position depends on the image aspect.
    """
    fig.canvas.draw()
    ax = fig.axes[0]
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=extent, pad_inches=0, dpi=dpi)


with open(args.csv_file, newline="") as f:
    reader = csv.DictReader(f, delimiter=";")
    rows = list(reader)

manifest_path = os.path.join(args.output_dir, "manifest.csv")
manifest_exists = os.path.exists(manifest_path)
manifest = open(manifest_path, "a", newline="")
manifest_writer = csv.writer(manifest, delimiter=";")
if not manifest_exists:
    manifest_writer.writerow(["row", "model", "word", "image_url", "map", "clean_map"])

for model_type, model_path in models_to_run:
    print(f"Loading {model_type} model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    for row_idx, row in enumerate(rows):
        word = row["word"]
        prompt = row["prompt"]
        response = row["response"]
        image_url = row["image_url"]

        print(f"\nProcessing word='{word}', image_url='{image_url}'")

        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        with Saliency(model, backend="torch_eager"):
            out = model(**inputs)

        sal = out.saliency.view(
            image=image, processor=processor, input_ids=inputs.input_ids
        )

        try:
            fig = sal.plot(regex(word), alpha=0.8, cmap="inferno", title=f"Saliency Map for `{word}` ({model_type})")
            stem = f"{model_type}_{word}_{url_slug(image_url)}"
            map_path = os.path.join(args.output_dir, f"map_{stem}.png")
            clean_path = os.path.join(args.output_dir, f"clean_{stem}.png")
            fig.savefig(map_path, dpi=args.dpi)
            save_clean_map(fig, clean_path, args.dpi)
            manifest_writer.writerow([row_idx, model_type, word, image_url, map_path, clean_path])
            manifest.flush()
            print(f"  Saved saliency map for '{word}' ({model_type})")
        except Exception as e:
            print(f"  Skipping word '{word}' ({model_type}) — could not find it in the tokens.")
            print(f"  Available tokens: {sal.decoded_gen_tokens}")
            print(f"  Error: {e}")

    del model
    del processor
    torch.cuda.empty_cache()

manifest.close()
