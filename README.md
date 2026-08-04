# Saliency Alignment

This repository fine-tunes Vision-Language Models (VLMs) with a **Saliency Alignment Loss**: a KL-divergence term that pulls the model's per-token attention toward panoptic segmentation masks of the image regions each token refers to. It contains the full training, evaluation, and analysis pipeline for our paper, which uses this loss as a controlled intervention to ask whether improving attention localization *alone* improves downstream visual grounding.

The headline findings the code reproduces:

1. **Attention localization is directly supervisable.** Training LLaVA-1.5-7B with the alignment loss improves chance-normalized attention-mask recall (AMR) from 1.25 to 8.44 (6.8x) with a monotone dose–response in training length, while validation cross-entropy also improves.
2. **Localization gains do not transfer downstream.** Across a suite of grounding-sensitive benchmarks (counting, spatial reasoning, hallucination), aligned models are statistically indistinguishable from a matched lambda=0 control.
3. **The change is compactly encoded.** It is carried by the language model rather than the projector, concentrated in a stable set of attention heads, and reachable through rank-4 LoRA adapters.

## Installation

To install the necessary dependencies, we recommend using [uv](https://docs.astral.sh/uv/installation).

```bash
uv sync
```

However, if you prefer not to use `uv`, you can manually install the dependencies listed in `pyproject.toml` using pip:

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv/Scripts/activate 
pip install -e .[dev]
```

## Data Setup

Training expects the COCONut-PanCap data under `$PROJECT_DIR/data/coco/`. Download and prepare it once before fine-tuning:

```bash
uv run -m finetune.data.download
```

This fetches the COCO 2017 images and annotations, the COCONut panoptic masks (`xdeng77/coconut_s`), and the grounded captions, and arranges them in the layout described in [Data Directory Structure](#data-directory-structure). On a SLURM cluster, submit `scripts/cscs/data.sh` instead.

## Quick Start

With the data in place, fine-tune with the default configuration (LLaVA-1.5-7B, KL alignment loss at weight 0.5):

```bash
uv run -m finetune
```

Everything is configured through [Hydra](https://hydra.cc/), so any parameter can be overridden from the command line:

```bash
# CE-only control (lambda = 0): same data and recipe, alignment loss disabled
uv run -m finetune loss=default loss.weight=0

# Different model and batch size
uv run -m finetune model=qwen2.5-vl-7b data.dataloader_kwargs.batch_size=4
```

Model configs live in `configs/model/` (LLaVA-1.5 7B/13B, Qwen2.5-VL-7B, Gemma-3-4B); each carries the architecture's freeze list so only the intended submodules train. Loss configs live in `configs/loss/` (`kl` for the alignment loss, `alignment` for the MSE variant, `default` for CE-only).

## Repository Structure

```
finetune/               # Training package (Lightning + FSDP)
├── criterion/          # Auxiliary losses: kl.py (paper), alignment.py (MSE), zero.py (CE-only)
├── data/               # COCONut-PanCap download + per-token segment supervision
├── model.py            # Model/processor construction, freezing, LoRA, eager attention
└── finetune.py         # Entry point (python -m finetune)
configs/                # Hydra configs (model/, loss/, data/, optim/, trainer, lora, ...)
scripts/
├── cscs/               # SLURM pipeline: arr_train.sh, arr_align_eval.sh, arr_eval.sh, arr_setup.sh
│   └── experiments/    # Paper experiment launchers (see below)
└── python/             # Analysis: aggregate_results.py, compare_drift.py, viz.py
eval/lmms_eval/tasks/   # Custom lmms-eval task definitions (e.g. VLMs-are-Biased)
```

## Reproducing the Paper Experiments

Each launcher in `scripts/cscs/experiments/` submits training plus both evaluations (attention alignment and downstream) as dependent SLURM jobs. Runs are identified by a deterministic `RUN_ID`, so launchers are idempotent and can be resubmitted after failures.

| Launcher | Paper section | What it runs |
|---|---|---|
| `sweep_lambda.sh` | Objective appendix | Loss-weight sweep (lambda knee at 0.5) |
| `sweep_lr_length.sh` | Results (dose–response) | Training length 200–2400 steps, plus lambda=0 controls and LR grid |
| `run_mse.sh` | Objective appendix | MSE (`alignment`) variant of the loss |
| `sweep_lora.sh` | Analysis (locus) | LoRA ranks 4/16/128 vs. full fine-tuning |
| `sweep_models.sh` | Cross-architecture | LLaVA-1.5-13B, Qwen2.5-VL-7B, Gemma-3-4B, each with a lambda=0 control |

Component ablations (`lm_only`, `proj_only`, `lm_proj`) are direct `arr_train.sh` invocations with the corresponding freeze overrides.

Two evaluation jobs follow each training run:

- **`arr_align_eval.sh`** — intrinsic attention metrics (AMR, AP, NSS) on a held-out split of COCONut-PanCap (the last 10k samples, deterministically excluded from training).
- **`arr_eval.sh`** — downstream benchmarks via [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) on vLLM: CountBench, CV-Bench 2D/3D, MMStar, MMVP, SalBench O3, POPE, and VLMs-are-Biased.

Note that saliency extraction requires **eager attention** (`attn_implementation="eager"`); SDPA and FlashAttention do not expose per-head attention weights, which would silently collapse the alignment loss to zero. `finetune/model.py` enforces this.

## Analysis Tooling

- **`scripts/python/aggregate_results.py`** — collects alignment and lmms-eval results across runs into a single `summary.csv` (the source of the paper's tables).
- **`scripts/python/compare_drift.py`** — parameter-drift analysis between checkpoints: component-, layer-, and head-level deltas, plus Jaccard overlap of the most-changed attention heads across seeds/lengths.
- **`scripts/python/viz.py`** — qualitative saliency maps via [vl-saliency](https://pypi.org/project/vl-saliency/) for prompts in a CSV; compare checkpoints with `--models base=llava-hf/llava-1.5-7b-hf aligned=models/<run_id>`.

## LoRA Fine-tuning

The framework supports both full fine-tuning (default) and parameter-efficient
fine-tuning via [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation),
powered by [PEFT](https://github.com/huggingface/peft). LoRA settings live in
`configs/lora.yaml`. When LoRA is enabled, the model's `freeze` / `unfreeze` 
settings are ignored: PEFT freezes the base model and trains only the adapter 
weights.

Training writes only the adapter weights to `models/<run_id>/`. The
SLURM training script (`scripts/cscs/arr_train.sh`) immediately follows
with a merge step that materializes a full HF checkpoint at
`models/<run_id>-merged/`.

## Logging and Monitoring

We use [Weights & Biases](https://wandb.ai/) for logging and monitoring the finetuning process. Make sure to set up your W&B account and configure the API key before starting the finetuning.

The project name for W&B logging can be set in the configuration file or overridden from the command line:

```bash
uv run -m finetune wandb.project=my-project-name
```

## Dataset

This repository uses **COCONut-PanCap**, which combines COCO 2017 images with panoptic segmentation masks and detailed captions whose phrases are linked to segment IDs. During preprocessing, each caption token is mapped to the segments it mentions, giving the per-token binary masks the alignment loss trains against.

The dataset abstraction is flexible: any dataset providing a per-token mask of the image regions each token refers to can be plugged in through `configs/data/`.

## Defining a Custom Loss

Auxiliary losses live in `finetune/criterion/` and are selected via the config group `configs/loss/`. A custom loss inherits from `finetune.criterion.Criterion` and implements a single method:

```python
def compute_loss(self, attn, mask):
    """attn: (S, H, W) softmax-normalized attention maps for supervised tokens.
    mask: (S, H, W) boolean segmentation masks for the same tokens.
    Returns a scalar loss."""
```

The base class handles everything else: it receives `(labels, segment_ids, preds, saliency, masks)` per batch, extracts per-token attention grids from the `vl_saliency.SaliencyGrid`, upsamples them to mask resolution, normalizes them to distributions, builds the per-token binary masks from segment IDs, and skips tokens without segment annotations. `KL(target || attn)` in `kl.py` is the paper's loss; `alignment.py` implements the MSE variant.

To use a custom loss, add a config to `configs/loss/`:

```yaml
_target_: finetune.criterion.my_custom_loss.MyCustomLoss
weight: 0.5  # Defined for all criterion classes, default is 1.0
```

and select it with `uv run -m finetune loss=my_custom_loss`.

## Data Directory Structure

After the download step, the data is organized as follows:

```
$PROJECT_DIR/data/coco/
│
├── images/
│   ├── train2017/                        # 118,287 training images
│   │   ├── 000000000009.jpg
│   │   └── ...
│   └── val2017/                          # 5,000 validation images
│       ├── 000000000139.jpg
│       └── ...
│
└── annotations/
    ├── panoptic_segmentation/            # COCONut panoptic masks (xdeng77/coconut_s)
    │   ├── train2017/
    │   │   ├── 000000000009.png
    │   │   └── ...
    │   └── val2017/
    ├── panoptic_train2017.json           # COCONut panoptic annotations
    ├── panoptic_val2017.json
    ├── instances_train2017.json          # COCO instance segmentation
    ├── instances_val2017.json
    ├── captions_train2017.json           # COCO captions
    ├── captions_val2017.json
    └── ...                               # grounded captions with segment annotations
```
