"""Collate functions for PNG-COCO dataset with segment ID tracking."""

import torch
from transformers import ProcessorMixin

from finetune.data.png.tokenization import build_clean_caption, tokenize_png_caption


def train_collate_fn(examples: list[dict], processor: ProcessorMixin) -> dict:
    """Collate function for training with PNG-COCO segment ID tracking.

    This function processes a batch of examples from the PNG-COCO dataset, creating
    clean captions for the model while separately tracking which tokens correspond
    to which segment IDs for custom loss computation.

    Args:
        examples: List of dictionaries, each containing:
            - image: PIL Image object
            - caption: Full caption string
            - segments: List of segment dicts with utterance and segment_ids
            - panoptic_mask: Panoptic segmentation mask tensor
        processor: Vision-language model processor
            that handles both image and text processing.

    Returns:
        dict:
            - input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - labels (torch.Tensor): Label IDs for training (same as input_ids with -100 for padding)
            - segment_ids (torch.Tensor): Per-token segment IDs [batch_size, seq_len, max_segments], padded with -1
            - panoptic_masks (list[torch.Tensor]): Panoptic masks [batch_size, H, W] containing segment IDs
            - **batch: Additional fields provided by the processor
    """
    images = []
    texts = []
    all_segments = []
    panoptic_masks = []

    for example in examples:
        image = example["image"]
        segments = example["segments"]
        panoptic_mask = example["panoptic_mask"]

        # Build clean caption from segments
        clean_caption = build_clean_caption(segments)

        if not clean_caption.strip():
            return None  # Skip empty captions

        # Build chat messages with clean caption
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the image in detail."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": clean_caption},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        images.append(image)
        texts.append(prompt)
        all_segments.append(segments)
        panoptic_masks.append(panoptic_mask)

    # Standard processing with clean captions
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=False,
        max_length=None,
        return_tensors="pt",
    )

    input_ids = batch["input_ids"]

    # Labels: ignore padding tokens
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # First pass: tokenize all captions and find max_segments
    batch_size, seq_len = input_ids.shape
    tokenized_segments = []
    max_segments = 1  # At least 1 to avoid empty dimension

    for segments in all_segments:
        _, cap_segment_ids = tokenize_png_caption(
            segments, processor.tokenizer, add_special_tokens=False
        )
        tokenized_segments.append(cap_segment_ids)
        for seg_ids in cap_segment_ids:
            if len(seg_ids) > max_segments:
                max_segments = len(seg_ids)

    # Create segment_ids tensor, padded with -1
    segment_ids_tensor = torch.full(
        (batch_size, seq_len, max_segments), -1, dtype=torch.long
    )

    # Collect indices and values for vectorized assignment
    batch_idx, token_idx, seg_idx, values = [], [], [], []

    for i, cap_segment_ids in enumerate(tokenized_segments):
        # Find where the caption starts in labels
        valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue

        caption_start = valid_positions[0].item()
        caption_len = min(len(cap_segment_ids), seq_len - caption_start)

        for j, seg_ids in enumerate(cap_segment_ids[:caption_len]):
            for k, seg_id in enumerate(seg_ids):
                batch_idx.append(i)
                token_idx.append(caption_start + j)
                seg_idx.append(k)
                values.append(seg_id)

    # Single vectorized assignment
    if values:
        segment_ids_tensor[batch_idx, token_idx, seg_idx] = torch.tensor(
            values, dtype=torch.long
        )

    return {
        **batch,
        "input_ids": input_ids,
        "labels": labels,
        "segment_ids": segment_ids_tensor,
        "panoptic_masks": panoptic_masks,
    }


def eval_collate_fn(examples: list[dict], processor: ProcessorMixin) -> dict:
    """Collate function for evaluation and inference with PNG-COCO.

    This function processes a batch of examples for evaluation by feeding only
    the user prompt to the model (without the assistant response), allowing the
    model to generate captions. Returns ground truth data for comparison.

    Args:
        examples: List of dictionaries, each containing:
            - image: PIL Image object
            - caption: Ground truth caption string
            - segments: List of segment dicts with utterance and segment_ids
            - panoptic_mask: Panoptic segmentation mask tensor
        processor: Vision-language model processor
            that handles both image and text processing.

    Returns:
        dict: A dictionary containing:
            - input_ids (torch.Tensor): Token IDs for prompts [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - answers (list[str]): List of ground truth caption strings
            - segments (list[list[dict]]): Ground truth segments with segment_ids
            - panoptic_masks (list[torch.Tensor]): Panoptic masks [batch_size, H, W]
            - **batch: Additional fields provided by the processor
    """
    images = []
    texts = []
    answers = []
    all_segments = []
    panoptic_masks = []

    for example in examples:
        image = example["image"]
        caption = example["caption"]
        segments = example["segments"]
        panoptic_mask = example["panoptic_mask"]

        images.append(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the image in detail."},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(prompt)
        answers.append(caption)
        all_segments.append(segments)
        panoptic_masks.append(panoptic_mask)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]

    # Labels: ignore padding tokens
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        **batch,
        "input_ids": input_ids,
        "labels": labels,
        "answers": answers,
        "segments": all_segments,
        "panoptic_masks": panoptic_masks,
    }
