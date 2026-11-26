# Saliency Alignment

This repository fine-tunes Vision-Language Models (VLMs) with a **Saliency Alignment Loss**: a KL-divergence term that pulls the model's per-token attention toward panoptic segmentation masks of the image regions each token refers to. It contains the full training, evaluation, and analysis pipeline for our paper, which uses this loss as a controlled intervention to ask whether improving attention localization *alone* improves downstream visual grounding.

The headline findings the code reproduces:

1. **Attention localization is directly supervisable.** Training LLaVA-1.5-7B with the alignment loss raises the chance-normalized Attention Mass Ratio (AMR) from 1.52 for the base model to 15.2 after 800 steps and 17.8 after 2,400 steps (roughly 12x), with a monotone dose–response in training length.
2. **Localization gains do not transfer downstream.** Across a suite of grounding-sensitive benchmarks (counting, spatial reasoning, hallucination), aligned models show no consistent downstream gains over a matched lambda=0 control.
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
align_eval/             # Intrinsic localization metrics (AMR, AP, NSS) and the counting analysis
configs/                # Hydra configs (model/, loss/, data/, optim/, trainer, lora, ...)
scripts/
├── cscs/               # SLURM pipeline: arr_train.sh, arr_align_eval.sh, arr_eval.sh, arr_setup.sh
│   └── experiments/    # Paper experiment launchers (see below)
└── python/             # Analysis and figures: aggregate_results.py, compare_drift.py, viz.py, ...
tests/                  # Unit tests: collator alignment, mask cropping, token/attention-row pairing
eval/lmms_eval/tasks/   # Custom lmms-eval task definitions (e.g. VLMs-are-Biased)
```

## Reproducing the Paper Experiments

Each launcher in `scripts/cscs/experiments/` submits training plus both evaluations (attention alignment and downstream) as dependent SLURM jobs. Runs are identified by a deterministic `RUN_ID`, so launchers are idempotent and can be resubmitted after failures.

Section, figure, table and appendix numbers below refer to the camera-ready version of the paper.

| Launcher | Paper | What it runs |
|---|---|---|
| `reproduce_lm_only.sh` | Sections 5.1–5.2 (Figure 3, Table 1); Base and LM-only rows of Table 2 | LM-only training at lambda=0.5 and lambda=0 for 200, 800, 1,600 and 2,400 steps, each with the alignment evaluation and, from 800 steps on, the downstream suite; both evaluations of the base model |
| `reproduce_rest.sh` | Section 6 (Table 2); Appendices C–F (Tables 3, 4 and 8) | Component ablation (projector only, LM + projector), loss-weight sweep at 200 steps, LoRA ranks 4/16/128, two more seeds of the 800-step pair, parameter drift |
| `sweep_lambda.sh` | Appendix C, Loss-Weight Sensitivity (Table 3) | Loss-weight sweep at 200 steps, lambda 0 to 5 (0.5 is the paper's operating point) |
| `sweep_lr_length.sh` | Section 5.1 (two of the training lengths in Figure 3); the lr 5e-6 runs are not reported in the paper | lambda=0.5 at 800 and 1,600 steps, at learning rates 2e-5 and 5e-6 |
| `sweep_lora.sh` | Appendix F, LoRA Rank Sweep (Table 8) | LoRA ranks 4/16/128 vs. full fine-tuning |
| `run_mse.sh` | Not reported in the paper | MSE (`alignment`) variant of the loss |

`reproduce_lm_only.sh` and `reproduce_rest.sh` together submit the training runs behind Figure 3 and Tables 1–4 and 8; the `sweep_*.sh` launchers are the earlier per-axis sweeps. The component ablation uses `arr_train.sh` with the `lm_only`, `proj_only` and `lm_proj` freeze overrides.

Two evaluation jobs follow each training run:

- **`arr_align_eval.sh`** — intrinsic attention metrics (AMR, AP, NSS) on a held-out split of COCONut-PanCap (the last 10k samples, deterministically excluded from training).
- **`arr_eval.sh`** — downstream benchmarks via [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) on vLLM: CountBench, CV-Bench 2D/3D, MMStar, MMVP, SalBench O3, POPE, and VLMs-are-Biased.

Two further jobs run single analyses: `arr_counting.sh` (counting correlation, Section 5.3 and Appendix D; needs the question file from the separate `cat_bench` repository, see `CAT_BENCH_DATA`) and `arr_drift.sh` (parameter drift, Appendix E).

Note that saliency extraction requires **eager attention** (`attn_implementation="eager"`); SDPA and FlashAttention do not expose per-head attention weights, which would silently collapse the alignment loss to zero. `finetune/model.py` enforces this.

### Attention read-out position

For a grounded caption token at position *t*, the loss and the localization metrics read the attention of decoder position *t* itself, where the token is the input. This is the setting of the paper's main experiments (Section 3.2).

Appendix G repeats the experiments at the preceding position *t − 1*, whose hidden state predicts the token. That variant is not a config option. To reproduce it, change the slice that pairs tokens with attention rows in both files below, and the constant the unit tests check them against:

| File | Main experiments (current code) | Appendix G |
|---|---|---|
| `finetune/criterion/base.py` | `attn = attn[-gen_len:]` | `attn = attn[-gen_len - 1 : -1]` |
| `align_eval/metrics.py` | `attn = attn[-seg_ids.shape[0] :]` | `attn = attn[-seg_ids.shape[0] - 1 : -1]` |
| `tests/test_criterion_alignment.py` | `QUERY_SHIFT = 0` | `QUERY_SHIFT = 1` |

Train and evaluate with the same setting. Run IDs do not encode the read-out position, and `arr_train.sh` skips a run whose checkpoint already exists, so give the second setting its own `RUN_TAG` (honoured by `reproduce_lm_only.sh` and `reproduce_rest.sh`) or a fresh `PROJECT_DIR`; otherwise checkpoints trained at the other position are reused. The one-batch diagnostic `scripts/python/check_alignment.py` copies the criterion's slice and should be changed with it. The launchers are the same for both settings; at *t − 1* they produce Tables 9–11 instead of the main-text results. The counting analysis (`align_eval/counting.py`, Section 5.3) does not depend on this choice: it always reads the position that predicts the count, since the answer is not part of the input.

## Analysis Tooling

- **`scripts/python/aggregate_results.py`** — collects alignment and lmms-eval results across runs into a single `summary.csv` (the source of the paper's tables).
- **`scripts/python/compare_drift.py`** — parameter-drift analysis between checkpoints: component-, layer-, and head-level deltas, plus Jaccard overlap of the most-changed attention heads across seeds/lengths.
- **`scripts/python/viz.py`** — qualitative saliency maps via [vl-saliency](https://pypi.org/project/vl-saliency/) for prompts in a CSV; compare checkpoints with `--models base=llava-hf/llava-1.5-7b-hf aligned=models/<run_id>`. Maps are drawn on the center crop of the image that the model sees.
- **`scripts/python/make_saliency_grid.py`** — assembles the qualitative saliency grid (Figure 2) from the maps and the `manifest.csv` that `viz.py` writes. The prompt sets are `scripts/python/prompts_appendix.csv` (COCO val2017 images) and `prompts_appendix_commons.csv` (an alternative set).
- **`scripts/python/plot_metrics_over_time.py`** — AMR, AP and NSS against training steps for the aligned runs and the lambda=0 controls (Figure 3), from `summary.csv`.
- **`scripts/python/seed_table.py`** — mean and standard deviation over seeds for the 800-step aligned/control pair (run-to-run variation, Appendix D), from `summary.csv`.
- **`scripts/python/paired_tests.py`** — paired McNemar tests between aligned and control runs on POPE and CountBench, from per-sample lmms-eval logs.
- **`scripts/python/make_report.py`** — Markdown report that sets a re-run next to the numbers of the original submission.

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
