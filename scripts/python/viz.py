import argparse
import csv
import hashlib
import os
import re
import time
from io import BytesIO
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
parser.add_argument(
    "--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
    help="Model dtype (float32 matches the paper; bfloat16 halves memory for CPU runs)",
)
parser.add_argument(
    "--offload_dir", default=None,
    help="Enable accelerate disk offload into this folder for machines whose RAM "
         "cannot hold the whole model (CPU-only inference).",
)
parser.add_argument(
    "--max_cpu_mem", default="10GiB",
    help="RAM budget for the weights when --offload_dir is set (rest is streamed from disk)",
)
args = parser.parse_args()

models_to_run = [tuple(spec.split("=", 1)) for spec in args.models]
os.makedirs(args.output_dir, exist_ok=True)

transformers.utils.logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = getattr(torch, args.dtype)


def url_slug(url: str) -> str:
    """Stable, collision-free file stem for an image URL.

    Uses the URL's basename (e.g. the COCO image id) when it is informative and
    appends a short hash so two URLs with the same basename never collide.
    """
    stem = os.path.splitext(os.path.basename(urlparse(url).path))[0]
    stem = re.sub(r"[^a-zA-Z0-9]", "_", stem)[:24] or "img"
    return f"{stem}_{hashlib.md5(url.encode()).hexdigest()[:6]}"


def load_image(url: str, cache_dir: str) -> Image.Image:
    """Fetch an image once into ``cache_dir`` (keyed by URL hash) and reuse it.

    Hosts such as Wikimedia rate-limit repeated fetches, and each CSV row would
    otherwise be downloaded once per model; caching also makes re-runs offline.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, hashlib.md5(url.encode()).hexdigest()[:10] + ".jpg")
    if not os.path.exists(path):
        headers = {"User-Agent": "saliency-alignment-viz/0.1 (academic research)"}
        for attempt in range(5):
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.ok and resp.headers.get("content-type", "").startswith("image/"):
                break
            time.sleep(5 * (attempt + 1))
        resp.raise_for_status()
        Image.open(BytesIO(resp.content)).convert("RGB").save(path, quality=95)
    return Image.open(path).convert("RGB")


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
    load_kwargs = dict(dtype=dtype, attn_implementation="eager")
    if args.offload_dir:
        # Keep up to --max_cpu_mem of weights resident and stream the remainder
        # from disk each forward pass; the model must not be .to()-moved afterwards.
        os.makedirs(args.offload_dir, exist_ok=True)
        load_kwargs.update(
            device_map="auto",
            max_memory={"cpu": args.max_cpu_mem},
            offload_folder=args.offload_dir,
        )
    model = LlavaForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    if not args.offload_dir:
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    for row_idx, row in enumerate(rows):
        word = row["word"]
        prompt = row["prompt"]
        response = row["response"]
        image_url = row["image_url"]

        print(f"\nProcessing word='{word}', image_url='{image_url}'")

        image = load_image(image_url, os.path.join(args.output_dir, "inputs"))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]

        # Render the chat template to text and hand the processor the local
        # image, so nothing is re-fetched from the URL at run time.
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)

        # The eager backend reads attention weights through hooks; no gradients
        # are needed, and skipping autograd keeps activation memory small.
        with Saliency(model, backend="torch_eager"), torch.no_grad():
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
