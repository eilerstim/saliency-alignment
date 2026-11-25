"""Assemble the qualitative saliency grid for the paper appendix.

Reads the prompts CSV (with optional ``group`` and ``label`` columns) and the
``manifest.csv`` written by ``viz.py``, and lays the examples out as column
groups, each showing Input | <model 1> | <model 2> ... per example row, with the
supervised token printed under every example.

Example:
    python scripts/python/make_saliency_grid.py scripts/python/prompts.csv \
        --maps_dir figs/appendix --models base=Base aligned=Aligned \
        --out figures/saliency_grid.pdf
"""

import argparse
import csv
import hashlib
import os
from collections import OrderedDict
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
# Embed text as TrueType (Type 42) rather than Type 3, which some submission
# systems flag and which renders poorly when zoomed.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", help="prompts CSV (';'-separated) used for viz.py")
parser.add_argument("--maps_dir", default="figs", help="viz.py output dir containing manifest.csv")
parser.add_argument(
    "--models",
    nargs="+",
    default=["base=Base", "aligned=Aligned"],
    metavar="NAME=DISPLAY",
    help="Model names as used in viz.py --models, with the column header to display",
)
parser.add_argument("--out", default="figures/saliency_grid.pdf")
parser.add_argument("--cell", type=float, default=1.0, help="Panel size in inches")
parser.add_argument("--crop", choices=["square", "none"], default="square",
                    help="Center-crop panels to squares for a uniform grid")
args = parser.parse_args()

models = [tuple(spec.split("=", 1)) for spec in args.models]

with open(args.csv_file, newline="") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

# manifest: (model, word, image_url) -> clean map path. Keyed by content rather
# than CSV row index so rows can be reordered or swapped without re-running viz.py.
manifest = {}
with open(os.path.join(args.maps_dir, "manifest.csv"), newline="") as f:
    for m in csv.DictReader(f, delimiter=";"):
        manifest[(m["model"], m["word"], m["image_url"])] = m["clean_map"]

# group rows in CSV order; rows without a group form one unnamed group
groups: "OrderedDict[str, list]" = OrderedDict()
for idx, row in enumerate(rows):
    groups.setdefault(row.get("group", "") or "", []).append((idx, row))

cache_dir = os.path.join(args.maps_dir, "inputs")
os.makedirs(cache_dir, exist_ok=True)


def load_input(url: str) -> Image.Image:
    # Same cache layout as viz.py (<maps_dir>/inputs/<md5(url)[:10]>.jpg), so the
    # grid reuses the exact images the maps were computed from.
    path = os.path.join(cache_dir, hashlib.md5(url.encode()).hexdigest()[:10] + ".jpg")
    if not os.path.exists(path):
        headers = {"User-Agent": "saliency-alignment-viz/0.1 (academic research)"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        Image.open(BytesIO(resp.content)).convert("RGB").save(path, quality=95)
    return Image.open(path).convert("RGB")


def square(im: Image.Image) -> Image.Image:
    if args.crop == "none":
        return im
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s))


n_groups = len(groups)
n_rows = max(len(g) for g in groups.values())
per_group = 1 + len(models)  # input + one column per model
gap = 0.35  # extra column between groups (in cell units)

header_in, label_in = 0.55, 0.28  # inches reserved for group headers / last row's token label
grid_h = args.cell * n_rows * 1.28  # panels plus the inter-row gap that holds the token labels
fig_w = args.cell * (n_groups * per_group + gap * (n_groups - 1))
fig_h = grid_h + header_in + label_in
fig = plt.figure(figsize=(fig_w, fig_h))

width_ratios = []
for g in range(n_groups):
    width_ratios += [1] * per_group
    if g < n_groups - 1:
        width_ratios.append(gap)
gs = fig.add_gridspec(
    n_rows, len(width_ratios), width_ratios=width_ratios,
    left=0.005, right=0.995, top=1 - header_in / fig_h, bottom=label_in / fig_h,
    wspace=0.06, hspace=0.30,
)

col_headers = ["Input"] + [disp for _, disp in models]
for g, (gname, examples) in enumerate(groups.items()):
    col0 = g * (per_group + 1)
    for r in range(n_rows):
        if r >= len(examples):
            continue
        idx, row = examples[r]
        panels = [square(load_input(row["image_url"]))]
        for name, _ in models:
            path = manifest.get((name, row["word"], row["image_url"]))
            panels.append(square(Image.open(path).convert("RGB")) if path else None)
        for c, im in enumerate(panels):
            ax = fig.add_subplot(gs[r, col0 + c])
            ax.set_axis_off()
            if im is not None:
                ax.imshow(im)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6)
            if r == 0:
                ax.set_title(col_headers[c], fontsize=8, pad=3)
        # token label centred under the example's panels
        label = row.get("label") or row["word"]
        mid = fig.add_subplot(gs[r, col0:col0 + per_group], frameon=False)
        mid.set_axis_off()
        mid.text(0.5, -0.06, f"“{label}”", ha="center", va="top",
                 fontsize=8, transform=mid.transAxes)
    # group header spanning its columns
    if gname:
        head = fig.add_subplot(gs[0, col0:col0 + per_group], frameon=False)
        head.set_axis_off()
        head.text(0.5, 1.22, gname, ha="center", va="bottom", fontsize=9.5,
                  fontweight="bold", transform=head.transAxes)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=300)
print(f"Saved {args.out}  ({n_groups} groups x {n_rows} rows, {len(models)} models)")
