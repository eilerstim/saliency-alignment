# Saliency Alignment

This project fine-tunes Vision-Language Models (VLMs) with custom saliency alignment losses. The framework trains models to better align their attention with semantic annotations in images, improving visual grounding capabilities.


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

## Supported Models

The framework currently supports the following Vision-Language Models:
- **LLaVA 1.5 7B** (`llava-hf/llava-1.5-7b-hf`)
- **LLaVA v1.6 Mistral 7B** (`llava-hf/llava-v1.6-mistral-7b-hf`)

Additional vision-language models can be easily added through the configuration system.

## Quick Start

To fine-tune a model with the default configuration:

```bash
uv run -m finetune
```

The training script will automatically download and prepare the COCONut dataset on the first run.

## Configuration

The finetuning process can be customized using the configuration files located in the `configs` directory. You can modify parameters such as learning rate, batch size, and even the model used for finetuning. This repository uses [Hydra](https://hydra.cc/) for configuration management, allowing for easy experimentation with different settings. Thus, you can override any configuration parameter directly from the command line. For example:

```bash
uv run -m finetune model=llava-v1.6-mistral-7b-hf data.dataloader_kwargs.batch_size=4
```

## Logging and Monitoring

We use [Weights & Biases](https://wandb.ai/) for logging and monitoring the finetuning process. Make sure to set up your W&B account and configure the API key before starting the finetuning.

The project name for W&B logging can be set in the configuration file or overridden from the command line:

```bash
uv run -m finetune wandb.project=my-project-name
```

## Dataset

This repository uses the **COCONut dataset**, which combines COCO 2017 images with panoptic segmentation masks and detailed captions containing segment annotations.

The framework supports a flexible dataset structure that allows models to learn from token-to-region alignments. All that is needed is a per-token mask indicating which image segments each token refers to.


## Defining a Custom Loss

Auxiliary losses can be defined in the `finetune/criterion/` directory. To use a custom loss function, specify it in the configuration file by setting the `criterion._target_` parameter to point to your custom loss class, which is instantiated with any required arguments.

The custom loss class should inherit from `finetune.criterion.Criterion` and implement the `compute_loss` method.

For example, if you have a custom loss class `MyCustomLoss` defined in `finetune/criterion/my_custom_loss.py`, you can set it up in the configuration as follows:

```yaml
_target_: finetune.criterion.my_custom_loss.MyCustomLoss
weight: 0.5  # Defined for all criterion classes, default is 1.0
other_param: value
```

A criterion takes as input the following parameters:
- `labels (batch_size, seq_len)`: The ground truth labels from the dataset.
- `preds (batch_size, seq_len, vocab_size)`: The model prediction logits.
- `attentions (list(batch_size, num_heads, seq_len, seq_len))`: The attention weights from the model.
- `annotation_ids (batch_size, seq_len, max_annotations)`: Annotation IDs per token (-1 for padding).
- `masks (batch_size, H, W)`: Segmentation masks with pixel values as segment IDs.
- `segments_infos (batch_size, max_segments, 2)`: (segment_id, category_id) pairs, padded with -1.

## Data Directory Structure

After running the training script with the COCONut dataset for the first time, the data will be automatically downloaded and organized as follows:

```
$PROJECT_DIR/saliency-alignment/data/coco/
│
├── train/                                 # Training images
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ... (118,287 training images)
│
├── validation/                            # Validation images
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ... (5,000 validation images)
│
├── annotations/                           # COCO and COCONut annotations
│   ├── instances_train2017.json          # COCO instance segmentation (train)
│   ├── instances_val2017.json            # COCO instance segmentation (val)
│   ├── captions_train2017.json           # COCO captions (train)
│   ├── captions_val2017.json             # COCO captions (val)
│   ├── panoptic_train2017.json           # COCONut panoptic annotations
│   └── ... (other COCO annotation files)
│
├── panoptic_train2017_masks/             # Panoptic segmentation masks
│   ├── 000000000009.png                  # From HuggingFace (xdeng77/coconut_s)
│   ├── 000000000025.png
│   └── ... (~50K images with panoptic annotations)
│
└── panoptic_train2017_captions/          # Annotated captions
    └── ... (caption files with segment annotations)
```
